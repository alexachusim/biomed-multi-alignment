"""
tests/test_embedding_comparison.py
-----------------------------------
Compare embeddings from vLLM plugin vs direct MAMMAL model.
This test requires GPU and both vllm-mammal-plugin and mammal packages installed.
"""

import numpy as np
import pytest
import torch
from vllm.inputs import TokensPrompt

from mammal.keys import (
    ENCODER_INPUTS_TOKENS,
    ENCODER_INPUTS_ATTENTION_MASK,
)
from mammal.model import Mammal
from vllm_mammal_plugin.mammal_utils import (
    tokenize_mammal,
    tokenize_mammal_with_attention_mask,
    get_vllm_mammal_model,
    get_mammal_tokenizer
)
from examples.example_prompts import (
    PROTEIN_CALMODULIN,
    PROTEIN_FLUORESCENT,
    SMILES_ASPIRIN,
    SMILES_CAFFEINE,
    SMILES_ETHER,
    GENE_MALAT1,
    GENE_BRCA1, 
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def get_vllm_embeddings(prompts: list[str], tokenizer_op=None):   
    """Get embeddings using vLLM plugin with utility functions.
    
    Args:
        prompts: List of text prompts to embed
        tokenizer_op: Optional shared tokenizer instance
    """
    # Use the utility function to get the model
    llm = get_vllm_mammal_model()

    # Create tokenizer if not provided
    if tokenizer_op is None:
        tokenizer_op = get_mammal_tokenizer()

    embeddings = []
    for prompt in prompts:
        token_ids = tokenize_mammal(prompt, tokenizer_op)
        token_prompt: TokensPrompt = {"prompt_token_ids": token_ids}
        
        # Get embedding for single prompt
        outputs = llm.embed([token_prompt])
        embedding = np.array(outputs[0].outputs.embedding)
        embeddings.append(embedding)
    
    return embeddings


def get_mammal_embeddings(model_name: str, prompts: list[str], tokenizer_op=None):
    """Get embeddings using direct MAMMAL model.    
       
    Args:
        model_name: Name of the MAMMAL model to load
        prompts: List of text prompts to embed
        tokenizer_op: Optional shared tokenizer instance
    """
    # Load model
    model = Mammal.from_pretrained(pretrained_model_name_or_path=model_name, allow_config_mismatch=True)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    model = model.to(device=device)
    
    # Create tokenizer if not provided
    if tokenizer_op is None:
        tokenizer_op = get_mammal_tokenizer()
    
    embeddings = []
    
    with torch.no_grad():
        for prompt in prompts:
            # Tokenize using utility function with shared tokenizer
            token_ids, attention_mask = tokenize_mammal_with_attention_mask(prompt, tokenizer_op)            
            
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
            last_hidden_state = encoder_output.last_hidden_state  # [batch, seq_len, hidden_dim]
            attention_mask_expanded = attention_mask_tensor.unsqueeze(-1)  # [batch, seq_len, 1]
            
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

    return embeddings


@pytest.mark.gpu
@pytest.mark.slow
class TestEmbeddingComparison:
    """Compare embeddings from vLLM plugin vs direct MAMMAL model."""
    
    def test_embedding_similarity(self):
        """Test that vLLM and MAMMAL produce similar embeddings."""
        model_name = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"
        
        # Create a single tokenizer instance to be shared across all tokenization calls
        print("\n" + "=" * 70)
        print("Creating shared tokenizer...")
        tokenizer_op = get_mammal_tokenizer(model_name)        
        
        prompts = [PROTEIN_CALMODULIN, SMILES_ASPIRIN, SMILES_CAFFEINE, PROTEIN_FLUORESCENT, SMILES_ETHER, GENE_MALAT1, GENE_BRCA1]
        names = ["Calmodulin (protein)", "Aspirin (SMILES)", "Caffeine (SMILES)", "Fluorescent (protein)", "Ether (SMILES)", "Malat1 (gene)", "BRCA1 (gene)"]
                       
        print("\n" + "=" * 70)
        print("Getting embeddings from vLLM plugin...")
        vllm_embeddings = get_vllm_embeddings(prompts, tokenizer_op)
        
        print("\n" + "=" * 70)
        print("Getting embeddings from direct MAMMAL model...")
        mammal_embeddings = get_mammal_embeddings(model_name, prompts, tokenizer_op)
        
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
            print(f"  vLLM embedding shape:   {vllm_emb.shape}")
            print(f"  MAMMAL embedding shape: {mammal_emb.shape}")
            print(f"  Approximate equality:   {approximate_equality}")
            print(f"  Cosine similarity:      {similarity:.6f}")
            print(f"  L2 distance:            {l2_distance:.6f}")
            
            # Assert high similarity (should be very close, > 0.99)
            assert similarity > 0.95, f"Embeddings for {name} are not similar enough: {similarity:.6f}"
            assert vllm_emb.shape == mammal_emb.shape, f"Embedding shapes don't match for {name}"        
        
        print("\n" + "=" * 70)
        print("✓ All embedding comparisons passed!")
        print("=" * 70)


if __name__ == "__main__":
    # Run the test directly
    test = TestEmbeddingComparison()
    test.test_embedding_similarity()
