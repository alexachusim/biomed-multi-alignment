"""
Offline example to get the embeddings of MAMMAL biomedical foundation model.

Usage:
    python examples/offline_mammal_usage.py
"""

import numpy as np
from vllm_mammal_plugin.mammal_utils import *
from vllm.inputs import TokensPrompt


# ---------------------------------------------------------------------------
# Pre-formatted MAMMAL prompt strings
# ---------------------------------------------------------------------------
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
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    model = get_mammal_model()
       
    labels = ["Calmodulin (protein)", "Aspirin (SMILES)", "Caffeine (SMILES)"]
    prompts: list[TokensPrompt] = [
        {"prompt_token_ids": tokenize_mammal(PROTEIN_CALMODULIN)},
        {"prompt_token_ids": tokenize_mammal(SMILES_ASPIRIN)},
        {"prompt_token_ids": tokenize_mammal(SMILES_CAFFEINE)},
    ]

    outputs = model.embed(prompts)

    print("=" * 60)
    print(f"{'Sequence':<30}  {'Embedding dim':>14}")
    print("=" * 60)

    embeddings = []
    for label, output in zip(labels, outputs):
        emb = np.array(output.outputs.embedding)
        embeddings.append(emb)
        print(f"{label:<30}  {emb.shape[0]:>14}")

    print()
    print("Pairwise cosine similarities:")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"  {labels[i]}  ↔  {labels[j]}: {sim:.4f}")


if __name__ == "__main__":
    main()
