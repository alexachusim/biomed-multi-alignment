# vllm-mammal

A **vLLM plugin** that exposes the [IBM MAMMAL biomedical foundation model](https://huggingface.co/ibm-research/biomed.omics.bl.sm.ma-ted-458m)
as an **encoder-only embedding model** inside vLLM's pooling runner.

MAMMAL is a 458M-parameter T5-style encoder-decoder model trained on over 2 billion biological samples across proteins, small molecules (SMILES), and single-cell gene expression data.  
This plugin uses **only the encoder stack** and applies mean-pooled, L2-normalised hidden states as dense embedding vectors.

`MammalEncoderModel` wraps HuggingFace's `T5EncoderModel`. Decoder weights are silently skipped during `load_weights` so only the encoder is loaded into GPU memory. Hidden states are mean-pooled over non-padding tokens and L2-normalised.
---

## Installation

```bash
# 1. Install vLLM (≥ 0.6.0 required for the pooling runner)
pip install vllm

# 2. Install the MAMMAL library from IBM (includes the modular tokenizer)
pip install git+https://github.com/BiomedSciAI/biomed-multi-alignment.git

# 3. Install this plugin
pip install -e .
```

vLLM auto-discovers the plugin via Python's `entry_points` mechanism once installed — no `VLLM_PLUGINS` environment variable needed.

---

## Quick Start

### Offline inference

See example at examples/offline_mammal_usage.py

### Online serving

See example at examples/offline_mammal_usage.py

---

## MAMMAL Prompt Syntax

| Modality | Prompt prefix |
|---|---|
| Protein (amino acid) | `<@TOKENIZER-TYPE=AA><MOLECULAR_ENTITY><MOLECULAR_ENTITY_OF_TYPE_PROTEIN><EOS>` |
| Small molecule (SMILES) | `<@TOKENIZER-TYPE=SMILES><MOLECULAR_ENTITY><MOLECULAR_ENTITY_OF_TYPE_SMALL_MOL><EOS>` |
| Gene expression | No prefix — pass the raw token sequence |

---

## Requirements

| Package | Version |
|---|---|
| Python | ≥ 3.9 |
| vllm | ≥ 0.6.0 |
| transformers | ≥ 4.40.0 |
| mammal (BiomedSciAI) | latest from GitHub |

---

## Citation

```bibtex
@misc{shoshan2024mammalmolecularaligned,
  title         = {MAMMAL -- Molecular Aligned Multi-Modal Architecture and Language},
  author        = {Yoel Shoshan and Moshiko Raboh and Michal Ozery-Flato and others},
  year          = {2024},
  eprint        = {2410.22367},
  archivePrefix = {arXiv},
  primaryClass  = {q-bio.QM},
  url           = {https://arxiv.org/abs/2410.22367},
}
```

## License

Apache 2.0
