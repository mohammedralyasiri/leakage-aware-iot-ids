# Leakage-aware evaluation of hybrid deep-feature + boosting IoT intrusion detectors

Code accompanying the article *"Leakage-Aware Evaluation of Cross-Fitted CNN–XGBoost Stacking for IoT Intrusion Detection"* (M. R. Jasim).

The repository reproduces every table and figure of the article:

* a feature-space leakage audit of ToN-IoT (network Train_Test subset) and Edge-IIoTset;
* six evaluation protocols: P1 record-level random, P2 5-tuple group, P3 feature-space-deduplicated random, P4 source-host-disjoint (fixed host→fold map), P5 unseen attack family and P6 class-stratified temporal split;
* ten models (LR, RF, LightGBM, XGBoost-all, XGBoost-MI, MLP, 1D-CNN, in-sample CNN→XGBoost hybrid and two variants of the proposed cross-fitted CF-MI-CNN-XGB);
* robustness analyses (flow- and host-level bootstrap, dns_query ablation, repeated unseen-family runs, CNN input-order permutation).

## Data

The datasets are **not** redistributed here. Download them from their authors and place them under `data/`:

| File expected | Source |
|---|---|
| `data/Train_Test_Network.csv` | ToN-IoT, network Train_Test subset (UNSW Canberra Cyber Range) |
| `data/edge_iiotset_ml.csv` | Edge-IIoTset, `ML-EdgeIIoT-dataset.csv` (Ferrag et al., 2022) |

Paths can be overridden with the environment variables `TON_CSV` and `EDGE_CSV`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All experiments in the article were run on a two-core CPU without a GPU.

## Usage

Single run:

```bash
python run.py <P1|P2|P3|P4|P6> <bin|multi> <seed> [--models RF,XGB_all,...]
python zeroday.py <held-out family> <seed>          # P5
DATASET=edge python run.py P1 multi 0               # Edge-IIoTset
```

Full reproduction (several hours on CPU):

```bash
./run_all.sh
```

Useful environment variables: `RESULTS_DIR` (output folder), `DATASET` (`ton` or `edge`), `DROP_FEATURES` (comma-separated attributes to remove), `MIN_CLASS_N` (drop classes with fewer distinct flows), `SAVE_PRED` (store test probabilities), `PERM_SEED` (permute CNN input columns), `FAST` / `NO_CNN` (skip stacked or neural models in `zeroday.py`).

`p4_host_folds.json` fixes the source-host→fold assignment of P4, so that P4 is identical across scikit-learn versions (older releases produce degenerate folds with `StratifiedGroupKFold(shuffle=True)`).

## Files

| File | Purpose |
|---|---|
| `common.py` | data loading, feature-space deduplication, protocols, leakage-free encoder, MI knee selection, models, metrics |
| `run.py` | one (protocol, task, seed) run of all models |
| `zeroday.py` | unseen-attack-family protocol |
| `aggregate.py`, `agg2.py` | aggregation, statistics, figures |
| `scripts/host_bootstrap.py` | host-level cluster bootstrap of the pooled P4 MCC |
| `scripts/edge_audit.py` | Edge-IIoTset duplication and conflict audit |
| `notebook/` | self-contained Kaggle notebook (ToN-IoT) |

## License and citation

MIT License. If you use this code, please cite the article (see `CITATION.cff`).
