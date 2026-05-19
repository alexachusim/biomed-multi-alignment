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
from vllm_mammal_plugin.mammal_prompts import (
    PROTEIN_CALMODULIN,
    SMILES_ASPIRIN,
    SMILES_CAFFEINE,
)


def main():
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    model_name = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"
    
    names = ["Calmodulin (protein)", "Aspirin (SMILES)", "Caffeine (SMILES)"]
    texts = [PROTEIN_CALMODULIN, SMILES_ASPIRIN, SMILES_CAFFEINE]

    response = client.embeddings.create(model=model_name, input=texts)

    print("=" * 60)
    print(f"{'Sequence':<30}  {'Embedding dim':>14}")
    print("=" * 60)

    embeddings = []
    for name, item in zip(names, response.data):
        emb = np.array(item.embedding)
        embeddings.append(emb)
        print(f"{name:<30}  {emb.shape[0]:>14}")
        #print (f"Embedding: {name:<30} {emb}")

   
if __name__ == "__main__":
    main()
