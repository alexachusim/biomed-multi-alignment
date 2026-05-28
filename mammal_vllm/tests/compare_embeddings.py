"""
tests/test_embedding_comparison.py
-----------------------------------
Compare embeddings from vLLM plugin vs direct MAMMAL model.
This test requires GPU and both vllm-mammal-plugin and mammal packages installed.

Optional: Set COMPARE_ONLINE=true to also compare with online vLLM server.
To use online comparison, start the server first:
    vllm serve ibm-research/biomed.omics.bl.sm.ma-ted-458m \
        --runner pooling \
        --trust-remote-code \
        --skip_tokenizer_init \
        --gpu_memory_utilization 0.4 \
        --enforce_eager \
        --no-enable-prefix-caching
"""

import os
import time

import numpy as np
import torch
from mammal.keys import (
    ENCODER_INPUTS_ATTENTION_MASK,
    ENCODER_INPUTS_TOKENS,
)
from mammal.model import Mammal
from vllm import LLM
from vllm.inputs import TokensPrompt

from examples.example_prompts import (
    GENE_BRCA1,
    GENE_MALAT1,
    PROTEIN_CALMODULIN,
    PROTEIN_FLUORESCENT,
    SMILES_ASPIRIN,
    SMILES_CAFFEINE,
    SMILES_ETHER,
)
from vllm_mammal_plugin.tokenization import (
    get_mammal_tokenizer,
    tokenize_mammal,
    tokenize_mammal_with_attention_mask,
)

MODEL_NAME = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def get_vllm_embeddings(
    prompts: list[str], tokenizer_op=None
) -> tuple[list, float, float]:
    """Get embeddings using vLLM plugin with utility functions.

    Args:
        prompts: List of text prompts to embed
        tokenizer_op: Optional shared tokenizer instance

    Returns:
        Tuple of (embeddings, initialization_time, inference_time)
    """
    # Time model initialization
    init_start = time.time()
    llm = LLM(
        model=MODEL_NAME,
        runner="pooling",  # use the pooling / embedding runner
        trust_remote_code=True,  # MAMMAL uses custom tokenizer code
        skip_tokenizer_init=True,  # Skip vLLM's tokenizer - we use MAMMAL's custom one
        gpu_memory_utilization=0.4,  # Reduce GPU memory usage to fit in available memory
        enforce_eager=True,  # Disable CUDA graphs to avoid device-side assert errors
        enable_prefix_caching=False,  # Disable prefix/KV caching
    )
    init_time = time.time() - init_start

    # Create tokenizer if not provided
    if tokenizer_op is None:
        tokenizer_op = get_mammal_tokenizer()

    # Time inference
    inference_start = time.time()

    # Tokenize all prompts and create batch
    token_prompts: list[TokensPrompt] = [
        {"prompt_token_ids": tokenize_mammal(prompt, tokenizer_op)}
        for prompt in prompts
    ]

    # Get embeddings for all prompts in a single batch
    outputs = llm.embed(token_prompts)

    # Extract embeddings from outputs
    embeddings = [np.array(output.outputs.embedding) for output in outputs]

    inference_time = time.time() - inference_start

    return embeddings, init_time, inference_time


def get_online_vllm_embeddings(
    prompts: list[str], tokenizer_op=None, base_url: str = "http://localhost:8000/v1"
) -> tuple[list, float]:
    """Get embeddings using online vLLM server via OpenAI-compatible API.

    Args:
        prompts: List of text prompts to embed
        tokenizer_op: Optional shared tokenizer instance
        base_url: Base URL for the vLLM server

    Returns:
        Tuple of (embeddings, inference_time)
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package is required for online comparison. Install with: pip install openai"
        )

    # Create tokenizer if not provided
    if tokenizer_op is None:
        tokenizer_op = get_mammal_tokenizer()

    client = OpenAI(base_url=base_url, api_key="EMPTY")

    # Time inference
    inference_start = time.time()

    # Tokenize all prompts using MAMMAL's custom tokenizer
    token_ids_batch = [tokenize_mammal(prompt, tokenizer_op) for prompt in prompts]

    # Send all tokenized prompts in a single batch request
    response = client.embeddings.create(model=MODEL_NAME, input=token_ids_batch)

    # Extract embeddings from batch response
    embeddings = [np.array(data.embedding) for data in response.data]

    inference_time = time.time() - inference_start

    return embeddings, inference_time


def get_mammal_embeddings(
    model_name: str, prompts: list[str], tokenizer_op=None
) -> tuple[list, float, float]:
    """Get embeddings using direct MAMMAL model.

    Args:
        model_name: Name of the MAMMAL model to load
        prompts: List of text prompts to embed
        tokenizer_op: Optional shared tokenizer instance

    Returns:
        Tuple of (embeddings, initialization_time, inference_time)
    """
    # Time model initialization
    init_start = time.time()
    # Load model
    model = Mammal.from_pretrained(
        pretrained_model_name_or_path=model_name, allow_config_mismatch=True
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)
    init_time = time.time() - init_start

    # Create tokenizer if not provided
    if tokenizer_op is None:
        tokenizer_op = get_mammal_tokenizer()

    # Time inference
    inference_start = time.time()
    embeddings = []

    with torch.no_grad():
        for prompt in prompts:
            # Tokenize using utility function with shared tokenizer
            token_ids, attention_mask = tokenize_mammal_with_attention_mask(
                prompt, tokenizer_op
            )

            # Convert to tensors and add batch dimension, then move to device
            input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
            attention_mask_tensor = torch.tensor(attention_mask).unsqueeze(0).to(device)

            # Create batch_dict with required keys for _calculate_inputs_embeddings
            batch_dict = {
                ENCODER_INPUTS_TOKENS: input_ids,
                ENCODER_INPUTS_ATTENTION_MASK: attention_mask_tensor,
            }

            # Get input embeddings using the model's internal method
            input_embeddings = model._calculate_inputs_embeddings(batch_dict)

            # Pass through encoder
            encoder_output = model.t5_model.encoder(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask_tensor,
            )

            # Use mean pooling over sequence length (excluding padding)
            last_hidden_state = (
                encoder_output.last_hidden_state
            )  # [batch, seq_len, hidden_dim]
            attention_mask_expanded = attention_mask_tensor.unsqueeze(
                -1
            )  # [batch, seq_len, 1]

            # Mean pooling
            masked_hidden_state = last_hidden_state * attention_mask_expanded
            sum_hidden_state = masked_hidden_state.sum(dim=1)  # [batch, hidden_dim]
            sum_mask = attention_mask_expanded.sum(dim=1)  # [batch, 1]
            pooled_output = sum_hidden_state / sum_mask  # [batch, hidden_dim]

            # Convert to numpy and remove batch dimension
            embedding = pooled_output.squeeze(0).cpu().numpy()

            # Normalize the embedding (L2 normalization)
            embedding_norm = np.linalg.norm(embedding)
            if embedding_norm > 0:
                embedding = embedding / embedding_norm

            embeddings.append(embedding)

    inference_time = time.time() - inference_start

    return embeddings, init_time, inference_time


class TestEmbeddingComparison:
    """Compare embeddings from vLLM plugin vs direct MAMMAL model."""

    def test_embedding_similarity(self):
        """Test that vLLM and MAMMAL produce similar embeddings."""

        # Check if online comparison is requested
        compare_online = os.environ.get("COMPARE_ONLINE", "").lower() in (
            "true",
            "1",
            "yes",
        )

        # Create a single tokenizer instance to be shared across all tokenization calls
        print("\n" + "=" * 70)
        print("Creating shared tokenizer...")
        tokenizer_op = get_mammal_tokenizer(MODEL_NAME)

        prompts = [
            PROTEIN_CALMODULIN,
            SMILES_ASPIRIN,
            SMILES_CAFFEINE,
            PROTEIN_FLUORESCENT,
            SMILES_ETHER,
            GENE_MALAT1,
            GENE_BRCA1,
        ]
        names = [
            "Calmodulin (protein)",
            "Aspirin (SMILES)",
            "Caffeine (SMILES)",
            "Fluorescent (protein)",
            "Ether (SMILES)",
            "Malat1 (gene)",
            "BRCA1 (gene)",
        ]

        print("\n" + "=" * 70)
        print("Getting embeddings from vLLM plugin (offline)...")
        vllm_embeddings, vllm_init_time, vllm_inference_time = get_vllm_embeddings(
            prompts, tokenizer_op
        )
        print(f"  Initialization time: {vllm_init_time:.3f}s")
        print(f"  Inference time: {vllm_inference_time:.3f}s")
        print(f"  Total time: {vllm_init_time + vllm_inference_time:.3f}s")

        print("\n" + "=" * 70)
        print("Getting embeddings from direct MAMMAL model...")
        mammal_embeddings, mammal_init_time, mammal_inference_time = (
            get_mammal_embeddings(MODEL_NAME, prompts, tokenizer_op)
        )
        print(f"  Initialization time: {mammal_init_time:.3f}s")
        print(f"  Inference time: {mammal_inference_time:.3f}s")
        print(f"  Total time: {mammal_init_time + mammal_inference_time:.3f}s")

        # Optionally get online vLLM embeddings
        online_embeddings = None
        online_inference_time = None
        if compare_online:
            print("\n" + "=" * 70)
            print("Getting embeddings from online vLLM server...")
            try:
                online_embeddings, online_inference_time = get_online_vllm_embeddings(
                    prompts, tokenizer_op
                )
                print(f"  Inference time: {online_inference_time:.3f}s")
                print("✓ Successfully retrieved online embeddings")
            except Exception as e:
                print(f"⚠ Warning: Could not get online embeddings: {e}")
                print("  Continuing with offline comparison only...")

        print("\n" + "=" * 70)
        print("Embedding Comparison Results")
        print("=" * 70)

        # Compare embeddings
        for i, name in enumerate(names):
            vllm_emb = vllm_embeddings[i]
            mammal_emb = mammal_embeddings[i]

            # Calculate approximate equality
            approximate_equality = np.allclose(vllm_emb, mammal_emb, atol=1e-3)

            # Calculate cosine similarity
            similarity = cosine_similarity(vllm_emb, mammal_emb)

            # Calculate L2 distance
            l2_distance = np.linalg.norm(vllm_emb - mammal_emb)

            print(f"\n{name}:")
            print("  Offline vLLM comparison:")
            print(f"  vLLM shape:             {vllm_emb.shape}")
            print(f"  MAMMAL shape:           {mammal_emb.shape}")
            print(f"  Approximate equality:   {approximate_equality}")
            print(f"  Cosine similarity:      {similarity:.6f}")
            print(f"  L2 distance:            {l2_distance:.6f}")

            # If online embeddings are available, compare them too
            if online_embeddings is not None:
                online_emb = online_embeddings[i]
                online_approximate_equality = np.allclose(
                    online_emb, mammal_emb, atol=1e-3
                )
                online_similarity = cosine_similarity(online_emb, mammal_emb)
                online_l2 = np.linalg.norm(online_emb - mammal_emb)

                print("\n  Online vLLM comparison:")
                print(f"  vLLM shape:             {online_emb.shape}")
                print(f"  MAMMAL shape:           {mammal_emb.shape}")
                print(f"  Approximate equality:   {online_approximate_equality}")
                print(f"  Cosine similarity:      {online_similarity:.6f}")
                print(f"  L2 distance:            {online_l2:.6f}")

                # Assert online embeddings are also similar
                assert (
                    online_similarity > 0.95
                ), f"Online embeddings for {name} are not similar enough: {online_similarity:.6f}"

            # Assert high similarity (should be very close, > 0.95)
            assert (
                similarity > 0.95
            ), f"Embeddings for {name} are not similar enough: {similarity:.6f}"
            assert (
                vllm_emb.shape == mammal_emb.shape
            ), f"Embedding shapes don't match for {name}"

        print("\n" + "=" * 70)
        print("✓ All embedding comparisons passed!")

        # Display benchmark summary
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"Number of prompts: {len(prompts)}")
        print()

        # Create benchmark table
        print(
            f"{'Method':<25} {'Init Time (s)':<15} {'Inference Time (s)':<20} {'Total Time (s)':<15} {'Time per prompt (ms)':<20}"
        )
        print("-" * 95)

        # vLLM offline
        vllm_total = vllm_init_time + vllm_inference_time
        vllm_per_prompt = (vllm_inference_time / len(prompts)) * 1000
        print(
            f"{'vLLM (offline)':<25} {vllm_init_time:<15.3f} {vllm_inference_time:<20.3f} {vllm_total:<15.3f} {vllm_per_prompt:<20.2f}"
        )

        # Direct MAMMAL
        mammal_total = mammal_init_time + mammal_inference_time
        mammal_per_prompt = (mammal_inference_time / len(prompts)) * 1000
        print(
            f"{'Direct MAMMAL':<25} {mammal_init_time:<15.3f} {mammal_inference_time:<20.3f} {mammal_total:<15.3f} {mammal_per_prompt:<20.2f}"
        )

        # Online vLLM (if available)
        if online_inference_time is not None:
            online_per_prompt = (online_inference_time / len(prompts)) * 1000
            print(
                f"{'vLLM (online)':<25} {'N/A':<15} {online_inference_time:<20.3f} {online_inference_time:<15.3f} {online_per_prompt:<20.2f}"
            )

        print()
        print("Speedup Analysis:")
        print("-" * 95)

        # Calculate speedups (inference only, since init is one-time cost)
        if mammal_inference_time > 0:
            vllm_speedup = mammal_inference_time / vllm_inference_time
            print(
                f"  vLLM offline vs Direct MAMMAL: {vllm_speedup:.2f}x {'faster' if vllm_speedup > 1 else 'slower'}"
            )

        if online_inference_time is not None and mammal_inference_time > 0:
            online_speedup = mammal_inference_time / online_inference_time
            print(
                f"  vLLM online vs Direct MAMMAL:  {online_speedup:.2f}x {'faster' if online_speedup > 1 else 'slower'}"
            )

        if online_inference_time is not None and vllm_inference_time > 0:
            online_vs_offline = vllm_inference_time / online_inference_time
            print(
                f"  vLLM online vs vLLM offline:   {online_vs_offline:.2f}x {'faster' if online_vs_offline > 1 else 'slower'}"
            )

        print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run the test directly
    test = TestEmbeddingComparison()
    test.test_embedding_similarity()
