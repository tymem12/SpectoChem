# Learning Light: Predicting Transition Metal Complex Absorption Spectra with GNNs

Official code repository for the **"Learning Light"** paper on predicting UV-Vis absorption properties and electronic excitation spectra of transition metal complexes (TMCs) using 3D-aware Graph Neural Networks (GNNs).

The repository implements training, evaluation, and LaTeX result aggregation across four GNN backbones (**SchNet**, **GINE**, **GAT**, **GCN**) and a statistical **Null Model** baseline for:
1. **Binary classification**: predicting whether a TMC absorbs in the visible window (350–650 nm with oscillator strength $f > 0.01$).
2. **Excitation energy regression**: predicting the first 10 excited-state transition energies $S_1\text{–}S_{10}$ (in eV).
3. **Oscillator strength regression**: predicting the corresponding 10 oscillator strengths as $\log_{10}(\max(f, 10^{-5}))$.
4. **Absorption spectrum reconstruction**: combining the 10 predicted $(E_i, f_i)$ pairs into Gaussian-broadened spectra ($\sigma = 0.25\text{ eV}$) and evaluating spectral similarity metrics (JSD, Wasserstein, SID, STMSE, SMSE, SRMSE).

---

## Repository Structure

```text
.
├── config/                        # Hydra YAML configs (backbones, datasets, experiments, training)
│   ├── backbone/                  # Architecture configs: schnet, gine, gat, gcn
│   ├── dataset/                   # Task & dataset configs (binary, lambda/f regressors, etc.)
│   ├── exp/                       # Experiment presets linking dataset + training defaults
│   ├── model/                     # Model wrapper config (supervised_graph_level)
│   └── training/                  # Optimizer, scheduler, early stopping, checkpointing, logging
├── experiments/
│   ├── scripts/
│   │   ├── train_graph_level.py   # Main Hydra entry point for training and evaluation
│   │   └── parse_results.py       # Generates Null Model baselines, spectral metrics, Excel/JSON summaries
│   └── training_utils.py          # PyTorch Lightning Trainer, callbacks, and loggers
├── learning_light/                # Core Python package
│   ├── config.py                  # Pydantic v2 configuration schemas
│   ├── datasets/                  # TMQMGStarDataset (PyG) & GraphLevelDataModule (splits & standardization)
│   ├── metrics/                   # Classification and regression torchmetrics collections
│   ├── models/                    # SupervisedGraphLevelGNN, GNN backbones, MLP predictors & losses
│   ├── transforms/                # Graph construction transforms (radius graph & Euclidean distances)
│   └── utils/                     # Config, chemistry parsing, LR scheduler, and spectral loss utilities
├── notebooks/                     # Exploratory data analysis and external TMPHOTCAT benchmark notebooks
├── scripts/                       # Data download and d-block splitting scripts
│   ├── download_data.sh           # Downloads raw datasets and triggers d-block splitting
│   └── split_tmc_blocks.py        # Splits complexes into 3d, 4d, and 5d CSV files
│   └── generate_latex.py          # Builds publication-ready LaTeX tables from parsed JSON results
├── slurm_train_submit_scripts/    # SLURM batch scripts for reproducing full paper sweeps
├── script_train_executer.py       # CLI runner orchestrating training jobs and state loops (0–9)
├── Makefile                       # Environment installation and code quality targets
└── requirements*.txt              # Pinned Python dependencies (base, CPU, and CUDA 12.8 GPU)
```

---

## Installation & Dependencies

* **Python**: `3.11.6`
* **Core ML & Graph Stack**:
  * `torch==2.7.1`, `torch-geometric==2.6.1` (+ `pyg-lib`, `torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`) — deep learning and 3D molecular graph operations.
  * `pytorch-lightning==2.5.2` — training loop, checkpointing, early stopping, and reproducibility (`seed_everything`).
* **Configuration & Validation**:
  * `hydra-core==1.3.2`, `omegaconf==2.3.0`, `pydantic==2.11.7` — modular YAML configuration and strict schema validation.
* **Chemistry, Scientific Computing & Evaluation**:
  * `rdkit==2025.3.4`, `scikit-learn==1.7.0`, `pandas==2.3.1`, `seaborn==0.13.2` — molecular utilities, stratified/grouped splitting, metric evaluation, and plotting.
* **Experiment Tracking & Dev Tools**:
  * `tensorboard==2.19.0`, `wandb==0.21.0`, `tqdm==4.67.1`, `jupyterlab==4.4.4`, `mypy==1.17.0`, `pre-commit==4.2.0`.

### Setup

Create and activate a Python `3.11.3` virtual environment, then install either GPU (CUDA 12.8) or CPU dependencies via `Makefile`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate

# GPU (CUDA 12.8)
make install_gpu

# Or CPU / macOS
make install_cpu
```

---

## Data Preparation

Raw data files are expected under `data/datasets/TMQM_SPECTO/raw/`:

```text
data/datasets/TMQM_SPECTO/raw/
├── tmqmg_star.csv           # TD-DFT excited-state wavelengths & oscillator strengths (tmQMg*)
├── tmqm_all.csv             # 3D geometries, stoichiometries, SMILES, and metadata for all d-blocks
└── tmc_blocks/
    ├── 3d.csv               # 3d transition metal complexes (generated by split_tmc_blocks.py)
    ├── 4d.csv               # 4d transition metal complexes (generated by split_tmc_blocks.py)
    └── 5d.csv               # 5d transition metal complexes (generated by split_tmc_blocks.py)
```

* **`tmqmg_star.csv`**: Downloaded from [`uiocompcat/tmQMg_star`](https://github.com/uiocompcat/tmQMg_star/blob/main/tmQMg*.csv).
* **`tmqm_all.csv`**: Full tmQM geometry/metadata table (*TODO: add exact source URL*).
* **`tmc_blocks/{3d,4d,5d}.csv`**: Generated from the raw CSVs using `scripts/split_tmc_blocks.py`.

Run the preparation script from the repository root:

```bash
bash scripts/download_data.sh
```

On the first training run, `TMQMGStarDataset` automatically merges `tmqm_all.csv` (or `tmc_blocks/3d.csv`) with `tmqmg_star.csv` on `CSD_code == id`, applies whole-compound outlier removal ($\lambda > 1327.95\text{ nm}$), constructs 3D radius graphs ($r = 10.0\text{ \AA}$, max 64 neighbors), and caches processed PyTorch Geometric tensors in `data/datasets/TMQM_SPECTO/processed/`.

---

## Reproducing the Paper

### Seeds & Data Splitting

All experiments are repeated across **3 random seeds** (`2137`, `42`, `1234`). Each seed simultaneously controls:
1. **Model initialization & training determinism** (`seed_everything(seed)` in `experiments/scripts/train_graph_level.py`).
2. **Dataset partitioning** (80% train / 10% validation / 10% test in `learning_light/datasets/graph_level.py`).

To prevent data leakage, all splits are **grouped by molecular isomers** (`group_by_isomers: true`, grouping by sorted atomic composition so identical isomer families never span train/val/test) and verified for **atom-type coverage** (ensuring no element appears in validation/test without appearing in training). Binary classification splits are additionally **stratified** at the isomer-group level.

Key implementation files for splitting, training, and evaluation:
* `learning_light/datasets/graph_level.py` — implements isomer-grouped and per-block splitting, saves/verifies the 3d test `CSD_code` lists in `data/datasets/TMQM_SPECTO/raw/block_3d_test_per_seed/{binary_stratified,regression_non_stratified}/{seed}.txt`, and enforces leakage checks.
* `experiments/scripts/train_graph_level.py` — handles training, automatic checkpoint discovery for `3test` evaluation, and metric export (`metrics.json`).
* `experiments/scripts/parse_results.py` — reconstructs the exact splits per seed to generate the **Null Model** baseline (`dummy`), computes physical-unit raw MAE and spectral reconstruction metrics across all 10 states, and aggregates results into Excel and JSON.
* `slurm_train_submit_scripts/submit_{binary,f,lambdach}.sh` — SLURM submission scripts used to run the benchmark sweeps.

### Training Regimes

Models are evaluated under three d-block training/evaluation regimes controlled by `--block_3_split`:

| Scenario | `--block_3_split` | Trained on | Tested on | Output dir |
| :--- | :--- | :--- | :--- | :--- |
| **(a)** | `345` | 3d + 4d + 5d | 3d + 4d + 5d | `block_3_345` |
| **(b)** | `3test` | 3d + 4d + 5d (checkpoint from `345`) | 3d test | `block_3_3test` |
| **(c)** | `none` | 3d only | same 3d test complexes | `block_3_none` (binary) / `block_3_null` (regression) |

* **Important note on Scenario (b) (`3test`)**: Scenario **(b) does not perform any training**. When `block_3_split_mode` is set to `"3test"`, `train_graph_level.py` automatically sets `ONLY_TEST=True`, loads the best validation checkpoint saved during the corresponding **`345`** run (scenario **(a)**), and evaluates it exclusively on the 3d test complexes saved by scenario **(a)**. Consequently, **scenario (a) (`345`) must be completed before running scenario (b) (`3test`)**.
* **Consistency between (b) and (c)**: In scenario **(a)**, each d-block (`3d`, `4d`, `5d`) is split independently with the same seed and grouping logic, so the 3d test subset evaluated in **(b)** is identical to the 3d test subset generated when training on 3d complexes only in **(c)**.

---

## Running Experiments

### 1. Local Execution (`script_train_executer.py`)

`script_train_executer.py` is the unified runner for all tasks, models (`schnet`, `gine`, `gat`, `gcn`), seeds (`2137`, `42`, `1234`), and regimes (`345`, `3test`, `none`). Runs that already contain a finished `metrics.json` are skipped automatically.

* **Binary classification** (visible absorption):
  ```bash
  # Scenario (a): train & test on 3d+4d+5d
  PYTHONPATH=. python script_train_executer.py \
      --model-name schnet --exp binary --block_3_split 345 --seed 2137

  # Scenario (b): evaluate the 345 checkpoint on the 3d test set only (no training)
  PYTHONPATH=. python script_train_executer.py \
      --model-name schnet --exp binary --block_3_split 3test --seed 2137

  # Scenario (c): train & test on 3d complexes only
  PYTHONPATH=. python script_train_executer.py \
      --model-name schnet --exp binary --block_3_split none --seed 2137
  ```

* **Excitation energy regression** ($S_1\text{–}S_{10}$ in eV — automatically loops over states `0..9`):
  ```bash
  PYTHONPATH=. python script_train_executer.py \
      --model-name schnet --exp lambda_regression --block_3_split 345 --seed 2137
  ```

* **Oscillator strength regression** ($\log_{10}(\max(f, 10^{-5}))$ — automatically loops over states `0..9`):
  ```bash
  PYTHONPATH=. python script_train_executer.py \
      --model-name schnet --exp f_regression --block_3_split 345 --seed 2137
  ```

> **Note:** By default, `script_train_executer.py` uses `--normalize_eV` (`True`), `--f_as_log10` (`True`), and `--no-standarization` (`False`), matching the paper's experiments.

### 2. SLURM Cluster Execution

Adjust partition/module settings in `slurm_train_submit_scripts/submit.sh` for your cluster, then submit full model × seed sweeps:

```bash
# Binary classification
bash slurm_train_submit_scripts/submit_binary.sh

# Excitation energy regression (states 0–9)
bash slurm_train_submit_scripts/submit_lambdach.sh

# Oscillator strength regression (states 0–9)
bash slurm_train_submit_scripts/submit_f.sh
```

> **Tip:** The submission scripts iterate over `MODELS=("schnet" "gine" "gat" "gcn")` and `SEEDS=(2137 42 1234)`. Ensure `345` jobs finish before `3test` jobs execute (or run `345` first, then `3test`), and add `"none"` to `BLOCK_SPLITS` to include scenario **(c)**.

### 3. Parsing Results, Dummy Baselines & LaTeX Tables

Once training and evaluation runs have completed under `data/experiments/supervised/`:

```bash
# 1. Generate Null Model baselines, compute combined spectral metrics, and export Excel + JSON
PYTHONPATH=. python experiments/scripts/parse_results.py

# 2. Generate publication LaTeX tables in TEX_DIR/combined_3_splits/
python generate_latex.py
```

Outputs:
* `experiment_results.xlsx` and `experiment_results.json` — per-seed and per-state metrics for all models and baselines.
* `TEX_DIR/combined_3_splits/{binary_metrics,regression_metrics,spectral_metrics}.tex` — formatted LaTeX tables aggregating means and standard deviations across seeds.

---

## License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.
