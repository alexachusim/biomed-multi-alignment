# vllm-mammal

A **vLLM plugin** that exposes the [IBM MAMMAL biomedical foundation model](https://huggingface.co/ibm-research/biomed.omics.bl.sm.ma-ted-458m) as an **encoder-only embedding model** inside vLLM's pooling runner.

MAMMAL is a 458M-parameter T5-style encoder-decoder model trained on over 2 billion biological samples across proteins, small molecules (SMILES), and single-cell gene expression data. This plugin uses **only the encoder stack** and applies mean-pooled, L2-normalized hidden states as dense embedding vectors.

## Key Features

- 🧬 **Multi-modal biomedical embeddings**: Proteins, small molecules (SMILES), and gene expression
- ⚡ **High-performance inference**: Leverages vLLM's optimized pooling runner
- 🔌 **Easy integration**: Auto-discovered plugin via Python entry points
- 🎯 **Encoder-only**: Uses only the T5 encoder stack for efficient embeddings
- 📊 **Normalized embeddings**: L2-normalized for direct similarity comparisons

---

## Installation

### From source (recommended)

```bash
# Navigate to the mammal_vllm directory
cd /path/to/biomed-multi-alignment/mammal_vllm

# Install with all dependencies
pip install -e .
```

### Verification

After installation, verify the plugin is registered:

```python
from vllm_mammal_plugin import register_mammal_model
print("Plugin successfully installed!")
```

vLLM auto-discovers the plugin via the `entry_points` mechanism defined in `pyproject.toml`.

---

## Usage

### Offline Inference (Python API)

Use vLLM's Python API directly for embedding generation. See the complete example:

**📄 [`examples/offline_mammal_usage.py`](examples/offline_mammal_usage.py)**

Key points:
- Initialize LLM with `runner="pooling"` and `skip_tokenizer_init=True`
- Use `tokenize_mammal()` from `vllm_mammal_plugin.tokenization` to tokenize inputs
- Call `model.embed()` to generate embeddings

### Online Serving (OpenAI-Compatible API)

Start a vLLM server and use the OpenAI-compatible client. See the complete example:

**📄 [`examples/online_mammal_usage.py`](examples/online_mammal_usage.py)**

Server command:
```bash
vllm serve ibm-research/biomed.omics.bl.sm.ma-ted-458m \
    --runner pooling \
    --trust-remote-code \
    --skip_tokenizer_init \
    --gpu_memory_utilization 0.4 \
    --enforce_eager \
    --no-enable-prefix-caching
```

Key points:
- Pre-tokenize inputs using `tokenize_mammal()` before sending to API
- Use OpenAI client with the `/v1/embeddings` endpoint
- Embeddings are automatically normalized

---

## MAMMAL Input Format

See [`examples/example_prompts.py`](examples/example_prompts.py) for pre-formatted example prompts.

---

## Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests/test_plugin.py -v
```

### Run Embedding Comparison Tests

Compare embeddings from vLLM plugin vs direct MAMMAL model:

```bash
# Basic test (offline vLLM vs direct MAMMAL)
python tests/test_embedding_comparison.py

# Include online vLLM server comparison (requires server running)
COMPARE_ONLINE=true python tests/test_embedding_comparison.py
```

See **📄 [`tests/test_embedding_comparison.py`](tests/test_embedding_comparison.py)** for details.

---

## Troubleshooting

### "Tokenizer not available when skip_tokenizer_init=True"

**Cause**: Inputs must be pre-tokenized when using `skip_tokenizer_init=True`.

**Solution**: Use `tokenize_mammal()` before sending to the API:

```python
from vllm_mammal_plugin.tokenization import tokenize_mammal

token_ids = tokenize_mammal(your_text)
response = client.embeddings.create(model=model_name, input=[token_ids])
```

### Low cosine similarity between embeddings

**Cause**: Embeddings may not be normalized.

**Solution**: The plugin automatically normalizes embeddings. If comparing with external embeddings, ensure they are also L2-normalized:

```python
import numpy as np

embedding_norm = np.linalg.norm(embedding)
if embedding_norm > 0:
    embedding = embedding / embedding_norm
```

### Out of memory errors

**Solution**: Reduce GPU memory utilization:

```bash
vllm serve ... --gpu_memory_utilization 0.3
```

---

## Project Structure

```
mammal_vllm/
├── vllm_mammal_plugin/
│   ├── __init__.py          # Plugin registration
│   ├── mammal.py            # Model implementation
│   └── tokenization.py      # Tokenization utilities
├── examples/
│   ├── example_prompts.py   # Pre-formatted example prompts
│   ├── offline_mammal_usage.py  # Offline inference example
│   └── online_mammal_usage.py   # Online serving example
├── tests/
│   ├── test_plugin.py       # Unit tests
│   └── test_embedding_comparison.py  # Integration tests
├── pyproject.toml           # Project configuration
└── README.md                # This file
```
