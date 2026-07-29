# MAMMAL Cell-Line Drug Response: Fine-Tuning Guide

This guide walks through fine-tuning MAMMAL to predict **IC50** (drug sensitivity) from a **drug SMILES string** and a **tumor/cell-line gene expression profile**. It covers setup, training, where outputs are saved, and how to run inference afterward.

---

## What this task does

MAMMAL takes two inputs and predicts one output:

| Input | Description |
|---|---|
| **Drug** | SMILES string (e.g. `"CC(=O)NCCC1=CNc2c1cc(OC)cc2"`) |
| **Tumor / cell line** | Gene expression profile (ranked list of genes by expression level) |
| **Output** | **IC50** — lower values generally mean the drug is more potent against that tumor |

This example uses the **GDSC** (Genomics of Drug Sensitivity in Cancer) benchmark via [Therapeutics Data Commons (TDC)](https://tdcommons.ai/multi_pred_tasks/drugres/).

**Useful links:**

- [MAMMAL paper (arXiv)](https://arxiv.org/abs/2410.22367)
- [Base pretrained model on Hugging Face](https://huggingface.co/ibm-research/biomed.omics.bl.sm.ma-ted-458m)
- [GDSC drug response task (TDC)](https://tdcommons.ai/multi_pred_tasks/drugres/)
- [Main repo README](../../../README.md)

---

## Which files do what

You do **not** run `task.py` directly. It defines the task logic and is loaded automatically by the training framework.

| File | Purpose |
|---|---|
| `task.py` | Task definition (data formatting, metrics) — loaded via config |
| `config.yaml` | Training settings (dataset, epochs, output path) |
| `dataset.py` | Loads GDSC data from TDC |
| `pl_data_module.py` | PyTorch Lightning data pipeline |
| **`main_finetune.py`** (in `mammal/`) | **Entry point for training** |
| **`main_infer.py`** | **Entry point for inference** after training |

Flow:

```
config.yaml  →  main_finetune.py  →  uses task.py internally  →  best_epoch.ckpt
                                                                      ↓
                                                            main_infer.py  →  IC50 prediction
```

---

## Prerequisites

- **Python** >= 3.10
- **PyTorch** >= 2.0 ([install guide](https://pytorch.org/get-started/locally/))
- A GPU is recommended for full training; CPU works for small smoke tests

---

## Step 1: Set up the environment

Clone the repo (if you have not already):

```bash
git clone git@github.com:BiomedSciAI/biomed-multi-alignment.git
cd biomed-multi-alignment
```

Create and activate a conda environment:

```bash
conda create -n mammal python=3.10 -y
conda activate mammal
```

Install PyTorch for your system (example — adjust for your hardware):

```bash
# macOS (CPU or Apple Silicon)
pip install torch

# Linux + NVIDIA GPU (example)
# conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install MAMMAL and example dependencies from the repo root.

> **zsh note:** Square brackets are special in zsh. Always quote the package name:

```bash
pip install -e ".[examples]"
```

Verify the install:

```bash
python -c "import hydra; import mammal; print('Install OK')"
python -c "from mammal.examples.cell_line_drug_response.task import CellLineDrugResponseTask; print('Examples OK')"
```

---

## Step 2: Understand what gets downloaded automatically

| Asset | When | Source |
|---|---|---|
| Base MAMMAL weights (~458M params) | During fine-tuning | [Hugging Face](https://huggingface.co/ibm-research/biomed.omics.bl.sm.ma-ted-458m) |
| GDSC training data | During fine-tuning | [PyTDC / TDC](https://tdcommons.ai/multi_pred_tasks/drugres/) |
| Fine-tuned checkpoint | **After you finish training** | Saved locally (see Step 5) |

There is **no pre-built GDSC IC50 checkpoint** published by IBM on Hugging Face (unlike some other MAMMAL tasks such as drug–target binding or protein solubility). You fine-tune once locally, then reuse the checkpoint for all future predictions.

**Other published MAMMAL fine-tunes (for reference, not this task):**

- [Drug–target binding (pKd)](https://huggingface.co/ibm-research/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd)
- [Protein solubility](https://huggingface.co/ibm-research/biomed.omics.bl.sm.ma-ted-458m.protein_solubility)
- [All IBM MAMMAL fine-tunes](https://huggingface.co/models?other=base_model:finetune:ibm-research/biomed.omics.bl.sm.ma-ted-458m)

---

## Step 3: Configure training

Edit `mammal/examples/cell_line_drug_response/config.yaml`.

Key settings:

```yaml
name: mammal_cldr_gdsc2_seed_1234   # output folder name
model_dir: ${root}/${name}          # where checkpoints are saved

task:
  data_module_kwargs:
    dataset_name: "GDSC2"           # or "GDSC1"
    limit_samples: null             # set to 100 for a quick smoke test

trainer:
  max_epochs: 100                   # set to 2 for a quick smoke test
```

### Smoke test (recommended first)

Before a full run, confirm the pipeline works with a small subset:

```yaml
limit_samples: 100
max_epochs: 2
```

### Full training

Restore production settings:

```yaml
limit_samples: null
max_epochs: 100
```

---

## Step 4: Run fine-tuning

Run from the **repo root** with your `mammal` environment active:

```bash
conda activate mammal
cd /path/to/biomed-multi-alignment

python mammal/main_finetune.py \
  --config-name config.yaml \
  --config-path examples/cell_line_drug_response
```

What happens during training:

1. Downloads the base MAMMAL model from Hugging Face
2. Downloads GDSC2 (or GDSC1) via PyTDC
3. Fine-tunes on drug + gene expression → IC50 pairs
4. Saves the best checkpoint based on validation MSE

**You only need to train once.** After that, reuse the same checkpoint for unlimited inference runs.

---

## Step 5: Where the model is saved

Output directory (relative to repo root):

```
./mammal_cldr_gdsc2_seed_1234/
```

| File / folder | Description |
|---|---|
| **`best_epoch.ckpt`** | **Your fine-tuned model** — use this for inference |
| `tokenizer/` | Tokenizer saved alongside the model |
| `config.json` | Model configuration |
| Other `.ckpt` files | Possible per-epoch checkpoints |
| Logs | Training logs from PyTorch Lightning / Hydra |

Check that training completed successfully:

```bash
ls -lh mammal_cldr_gdsc2_seed_1234/
```

You should see `best_epoch.ckpt` (typically hundreds of MB to ~1 GB).

Nothing is uploaded automatically — the model stays on your machine unless you copy or publish it yourself.

---

## Step 6: Run inference

After fine-tuning, predict IC50 for any tumor + drug combination using `main_infer.py`.

### Option A: GDSC cell line by name

```bash
python mammal/examples/cell_line_drug_response/main_infer.py \
  --model_path ./mammal_cldr_gdsc2_seed_1234/best_epoch.ckpt \
  --cell_line_name A549 \
  --drug_smiles "CC(=O)NCCC1=CNc2c1cc(OC)cc2" \
  --drug_name "Melatonin"
```

### Option B: Custom tumor expression (`.h5ad` file)

Prepare an AnnData file with:

- **`X`**: one row per tumor, columns = expression values
- **`var_names`**: gene symbols (e.g. `TP53`, `EGFR`)

```bash
python mammal/examples/cell_line_drug_response/main_infer.py \
  --model_path ./mammal_cldr_gdsc2_seed_1234/best_epoch.ckpt \
  --cell_line_h5ad_file /path/to/my_tumor.h5ad \
  --drug_smiles "CC(CCl)OC(C)CCl" \
  --device cuda
```

Use `--device cpu` if you do not have a GPU.

### Screen multiple drugs in Python

```python
import scanpy as sc
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from mammal.model import Mammal
from mammal.examples.cell_line_drug_response.main_infer import cell_line_drug_infer

model = Mammal.from_pretrained("./mammal_cldr_gdsc2_seed_1234/best_epoch.ckpt")
model.eval()
tokenizer = ModularTokenizerOp.from_pretrained("ibm/biomed.omics.bl.sm.ma-ted-458m")

adata = sc.read_h5ad("my_tumor.h5ad")
drugs = {
    "Melatonin": "CC(=O)NCCC1=CNc2c1cc(OC)cc2",
    "Example drug": "CC(CCl)OC(C)CCl",
}

for name, smiles in drugs.items():
    ic50 = cell_line_drug_infer(model, tokenizer, adata, smiles, device="cpu")
    print(f"{name}: IC50 = {ic50:.4f}")
```

Lower predicted IC50 → model expects higher drug sensitivity.

---

## Step 7: Evaluate on the test set (optional)

After fine-tuning, evaluate on the held-out GDSC test split:

```bash
python mammal/main_finetune.py \
  --config-name config.yaml \
  --config-path examples/cell_line_drug_response \
  evaluate=True \
  model.pretrained_kwargs.pretrained_model_name_or_path=./mammal_cldr_gdsc2_seed_1234/best_epoch.ckpt
```

---

## Troubleshooting

### `zsh: no matches found: biomed-multi-alignment[examples]`

Quote the package name:

```bash
pip install -e ".[examples]"
```

### `ModuleNotFoundError: No module named 'hydra'`

Dependencies are not installed. Run from the repo root:

```bash
pip install -e ".[examples]"
```

Make sure you are in the correct conda environment (`conda activate mammal`), not `(base)`.

### `ModuleNotFoundError: No module named 'mammal.examples.cell_line_drug_response'`

Install the package in editable mode from the repo root:

```bash
pip install -e ".[examples]"
```

### Checkpoint path not found

The path `./mammal_cldr_gdsc2_seed_1234/best_epoch.ckpt` only exists **after** fine-tuning completes. You cannot download this checkpoint from the repo — you must train first (or obtain a checkpoint from someone who has trained one).

### Training is slow on CPU

Use the smoke-test settings (`limit_samples: 100`, `max_epochs: 2`) to verify the pipeline, then run full training on a GPU machine if available.

---

## When to retrain

| Situation | Action |
|---|---|
| First time setup | Fine-tune once on GDSC |
| Predicting new tumors / drugs | Reuse existing `best_epoch.ckpt` — no retraining |
| Want better accuracy | Retrain with full data and `max_epochs: 100` |
| Have your own lab IC50 data | Retrain (or fine-tune further) on your labeled data |
| Switch GDSC1 ↔ GDSC2 | Retrain with updated `dataset_name` in config |

---

## Recommended workflow

1. **Install** — `pip install -e ".[examples]"` from repo root
2. **Smoke test** — `limit_samples: 100`, `max_epochs: 2`
3. **Full train** — `limit_samples: null`, `max_epochs: 100` (once)
4. **Inference** — reuse `best_epoch.ckpt` for all tumor + drug predictions
5. **Optional** — evaluate on test set; retrain only if you add new data or need better performance

---

## Further reading

- [Main README — Cell Line Drug Response section](../../../README.md)
- [Advanced: create a new MAMMAL task](../../../tutorials/advanced_create_new_task.ipynb)
- [MAMMAL MCP server (agent integration)](../../../mammal_mcp/README.md)
