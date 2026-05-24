"""
Offline example to get the embeddings of MAMMAL biomedical foundation model.

Usage:
    python examples/offline_mammal_usage.py
"""

import numpy as np
from vllm_mammal_plugin.mammal_utils import *
from examples.example_prompts import (
    PROTEIN_CALMODULIN,
    SMILES_ASPIRIN,
    GENE_BRCA1,
)
from vllm.inputs import TokensPrompt


def main():
    model = get_vllm_mammal_model()
       
    names = ["Calmodulin (protein)", "Aspirin (SMILES)", "BRCA1 (gene)"]
    prompts: list[TokensPrompt] = [
        {"prompt_token_ids": tokenize_mammal(PROTEIN_CALMODULIN)},
        {"prompt_token_ids": tokenize_mammal(SMILES_ASPIRIN)},
        {"prompt_token_ids": tokenize_mammal(GENE_BRCA1)},
    ]

    outputs = model.embed(prompts)

    print("=" * 60)
    print(f"{'Sequence':<30}  {'Embedding dim':>14}")
    print("=" * 60)

    embeddings = []
    for name, output in zip(names, outputs):
        emb = np.array(output.outputs.embedding)
        embeddings.append(emb)
        print(f"{name:<30}  {emb.shape[0]:>14}")
        #print (f"Embedding: {name:<30} {emb}")    
    

if __name__ == "__main__":
    main()
