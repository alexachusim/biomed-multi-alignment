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

from mammal.keys import *
from mammal.model import Mammal
from vllm_mammal_plugin.mammal_utils import tokenize_mammal, tokenize_mammal_with_attention_mask, get_mammal_model


# Test sequences
PROTEIN_CALMODULIN = (
    "<@TOKENIZER-TYPE=AA>"
    "<MOLECULAR_ENTITY><MOLECULAR_ENTITY_OF_TYPE_PROTEIN>"
    "MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMISELDQDGFIDKEDLHDGDGKISFEEFLNLVNK"
    "EMTADVDGDGQVNYEEFVTMMTSK"
    "<EOS>"
)

SMILES_ASPIRIN = (
    "<@TOKENIZER-TYPE=SMILES>"
    "<MOLECULAR_ENTITY><MOLECULAR_ENTITY_OF_TYPE_SMALL_MOL>"
    "CC(=O)Oc1ccccc1C(=O)O"
    "<EOS>"
)

SMILES_CAFFEINE = (
    "<@TOKENIZER-TYPE=SMILES>"
    "<MOLECULAR_ENTITY><MOLECULAR_ENTITY_OF_TYPE_SMALL_MOL>"
    "Cn1cnc2c1c(=O)n(c(=O)n2C)C"
    "<EOS>"
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def get_vllm_embeddings(prompts: list[str]):
    """Get embeddings using vLLM plugin with utility functions."""
    # Use the utility function to get the model
    llm = get_mammal_model()

    # Prepare prompts with token IDs using the utility tokenizer
    token_prompts: list[TokensPrompt] = [
        {"prompt_token_ids": tokenize_mammal(prompt)}
        for prompt in prompts
    ]

    # Get embeddings
    outputs = llm.embed(token_prompts)
    embeddings = [np.array(output.outputs.embedding) for output in outputs]
    
    return embeddings


def get_mammal_embeddings(model_name: str, prompts: list[str]):
    """Get embeddings using direct MAMMAL model.
    
    Uses tokenize_mammal_with_attention_mask() from mammal_utils.py for tokenization.
    """
    # Load model
    model = Mammal.from_pretrained(pretrained_model_name_or_path=model_name, allow_config_mismatch=True)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    embeddings = []
    
    with torch.no_grad():
        for prompt in prompts:
            # Tokenize using utility function
            token_ids, attention_mask = tokenize_mammal_with_attention_mask(prompt)
            
            # Convert to tensors and add batch dimension
            input_ids = torch.tensor(token_ids).unsqueeze(0)  # [1, seq_len]
            attention_mask_tensor = torch.tensor(attention_mask).unsqueeze(0)  # [1, seq_len]
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask_tensor = attention_mask_tensor.cuda()
            
            # Get encoder output
            input_embeddings = model.t5_model.get_input_embeddings()(input_ids)
            
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
            
            embedding = pooled_output.squeeze(0).cpu().numpy()  # Remove batch dimension
            embeddings.append(embedding)
    
    return embeddings


@pytest.mark.gpu
@pytest.mark.slow
class TestEmbeddingComparison:
    """Compare embeddings from vLLM plugin vs direct MAMMAL model."""
    
    def test_embedding_similarity(self):
        """Test that vLLM and MAMMAL produce similar embeddings."""
        model_name = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"
        
        prompts = [PROTEIN_CALMODULIN, SMILES_ASPIRIN, SMILES_CAFFEINE]
        labels = ["Calmodulin (protein)", "Aspirin (SMILES)", "Caffeine (SMILES)"]
        
        print("\n" + "=" * 70)
        print("Getting embeddings from vLLM plugin...")
        vllm_embeddings = get_vllm_embeddings(prompts)
        
        print("Getting embeddings from direct MAMMAL model...")
        mammal_embeddings = get_mammal_embeddings(model_name, prompts)
        
        print("\n" + "=" * 70)
        print("Embedding Comparison Results")
        print("=" * 70)
        
        # Compare embeddings
        for i, label in enumerate(labels):
            vllm_emb = vllm_embeddings[i]
            mammal_emb = mammal_embeddings[i]
            
            # Calculate similarity
            similarity = cosine_similarity(vllm_emb, mammal_emb)
            
            # Calculate L2 distance
            l2_distance = np.linalg.norm(vllm_emb - mammal_emb)
            
            print(f"\n{label}:")
            print(f"  vLLM embedding shape:   {vllm_emb.shape}")
            print(f"  MAMMAL embedding shape: {mammal_emb.shape}")
            print(f"  Cosine similarity:      {similarity:.6f}")
            print(f"  L2 distance:            {l2_distance:.6f}")
            
            # Assert high similarity (should be very close, > 0.99)
            assert similarity > 0.95, f"Embeddings for {label} are not similar enough: {similarity:.6f}"
            assert vllm_emb.shape == mammal_emb.shape, f"Embedding shapes don't match for {label}"
        
        print("\n" + "=" * 70)
        print("Cross-comparison: vLLM embeddings")
        print("=" * 70)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                sim = cosine_similarity(vllm_embeddings[i], vllm_embeddings[j])
                print(f"  {labels[i]} ↔ {labels[j]}: {sim:.4f}")
        
        print("\n" + "=" * 70)
        print("Cross-comparison: MAMMAL embeddings")
        print("=" * 70)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                sim = cosine_similarity(mammal_embeddings[i], mammal_embeddings[j])
                print(f"  {labels[i]} ↔ {labels[j]}: {sim:.4f}")
        
        print("\n" + "=" * 70)
        print("✓ All embedding comparisons passed!")
        print("=" * 70)


if __name__ == "__main__":
    # Run the test directly
    test = TestEmbeddingComparison()
    test.test_embedding_similarity()
