"""
Offline example to get the embeddings of MAMMAL biomedical foundation model.

Usage:
    python examples/offline_mammal_usage.py
"""

import numpy as np
from vllm import LLM
from vllm.inputs import TokensPrompt

from examples.example_prompts import (
    GENE_BRCA1,
    PROTEIN_CALMODULIN,
    SMILES_ASPIRIN,
)
from vllm_mammal_plugin.tokenization import *


def main():
    model_name = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"
    model = LLM(
        model=model_name,
        runner="pooling",  # use the pooling / embedding runner
        trust_remote_code=True,  # MAMMAL uses custom tokenizer code
        skip_tokenizer_init=True,  # Skip vLLM's tokenizer - we use MAMMAL's custom one
        gpu_memory_utilization=0.4,  # Reduce GPU memory usage to fit in available memory
        enforce_eager=True,  # Disable CUDA graphs to avoid device-side assert errors
        enable_prefix_caching=False,  # Disable prefix/KV caching
    )

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
        # print (f"Embedding: {name:<30} {emb}")


if __name__ == "__main__":
    main()
