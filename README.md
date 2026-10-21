# QORH: Learning Query-Specific Ordered Rule Hypergraphs for Inductive Knowledge Graph Link Prediction

This repository accompanies our paper **"QORH: Learning Query-Specific Ordered Rule Hypergraphs for Inductive Knowledge Graph Link Prediction"** 

## Overview

**QORH (Query-specific Ordered Rule Hypergraph)** jointly represents rule heads and ordered rule bodies as query-conditioned hyperedges. By modeling higher-order dependencies among multiple relations, QORH learns transferable structural patterns for inductive link prediction. Experiments on entity-inductive and fully inductive benchmarks demonstrate the effectiveness of rule hypergraph modeling.

## Framework

![Overview of the QORH framework](Fig_methodology_overview_01.png)

## Dependencies

Use **Python 3.10**, **PyTorch 2.5.1**, and **torch-scatter 2.1.2**. Run from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
python -m pip install -r requirements.txt
```

## Quick Start

Datasets are stored in `data/`:

- **Entity-inductive:** paired `<DATASET>_train_ind/` and `<DATASET>_ind/` directories, each containing `train.txt`, `valid.txt`, and `test.txt`.
- **Fully inductive:** a `<DATASET>_ind/` directory containing `train.txt`, `msg.txt`, `valid.txt`, and `test.txt`.

Run the following examples from the repository root. The final argument selects GPU `0`; training defaults to 200 epochs with early stopping patience of 10.

### Entity-inductive

```bash
bash my_script/train_tradiInduc.sh fb237_v1 0
bash my_script/test_tradiInduc.sh fb237_v1 0
```

### Fully inductive

```bash
bash my_script/train.sh FB-25 0
bash my_script/test.sh FB-25 0
```

Best checkpoints are selected by validation MRR and saved to `saved_models/`. Each test script loads the corresponding checkpoint and reports filtered MRR, Hits@1, and Hits@10 on `test.txt`.