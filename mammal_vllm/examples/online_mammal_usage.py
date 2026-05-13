"""
Online example to get the embeddings of MAMMAL biomedical foundation model.

This example uses the OpenAI-compatible /v1/embeddings endpoint.

Start the server first:

    vllm serve ibm-research/biomed.omics.bl.sm.ma-ted-458m \\
        --runner pooling \\
        --trust-remote-code \\
        --max-model-len 1024 \\
        --dtype float16

Then run this script:

    python examples/online_mammal_usage.py
"""

import numpy as np
from openai import OpenAI


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
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    model_name = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"
    
    labels = ["Calmodulin (protein)", "Aspirin (SMILES)", "Caffeine (SMILES)"]
    texts = [PROTEIN_CALMODULIN, SMILES_ASPIRIN, SMILES_CAFFEINE]

    response = client.embeddings.create(model=model_name, input=texts)

    print("=" * 60)
    print(f"{'Sequence':<30}  {'Embedding dim':>14}")
    print("=" * 60)

    embeddings = []
    for label, item in zip(labels, response.data):
        emb = np.array(item.embedding)
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
