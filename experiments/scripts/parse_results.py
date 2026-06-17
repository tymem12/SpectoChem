import os
import json
import glob
import time
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Tuple, List, Dict

import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
)
from omegaconf import OmegaConf
from lightning_fabric import seed_everything

from gjepa.config import GraphLevelExperimentConfig
from gjepa.datasets.graph_level import GraphLevelDataModule
from gjepa.utils.config import resolve_config
from gjepa.utils import spectral_loss

# --- EXPERIMENT CONFIGURATION ---
SEEDS = [2137, 42, 1234]
BASE_MODELS = ["schnet", "gine", "gat", "gcn"]
MODELS = BASE_MODELS + ["dummy"]
BLOCK_SPLITS = ["none", "test"]
BASE_DIR = Path("supervised")

# --- DUMMY BASELINE GENERATOR ---

def get_targets_and_ids(dataloader) -> tuple[np.ndarray, list]:
    """Extracts all target labels/values and origin_ids from a dataloader."""
    targets = []
    origin_ids = []
    for batch in dataloader:
        targets.append(batch.y.flatten())
        if hasattr(batch, 'origin_id'):
            if isinstance(batch.origin_id, (list, tuple)):
                origin_ids.extend(batch.origin_id)
            elif hasattr(batch.origin_id, 'tolist'):
                origin_ids.extend(batch.origin_id.tolist())
            else:
                origin_ids.extend(list(batch.origin_id))
    return torch.cat(targets).numpy(), origin_ids

def create_dummy_baseline(ref_dir: Path):
    """Generates the dummy baseline based on a valid reference experiment directory."""
    parts = list(ref_dir.parts)
    
    # Path manipulation: find the base model name and swap it with 'dummy'
    model_idx = -1
    for m in BASE_MODELS:
        if m in parts:
            model_idx = parts.index(m)
            break
            
    if model_idx == -1:
        print(f"Warning: Could not find base model in path {ref_dir} to replace with 'dummy'.")
        return

    parts[model_idx] = "dummy"
    dummy_dir = Path(*parts)
    
    metrics_path = dummy_dir / "metrics.json"
    if metrics_path.exists():
        return  # Dummy already generated
        
    print(f"Generating dummy baseline at {dummy_dir}...")
    dummy_dir.mkdir(parents=True, exist_ok=True)
    eda_dir = dummy_dir / "EDA"
    eda_dir.mkdir(exist_ok=True)

    # Load Hydra Config from Reference
    config_paths = [ref_dir / ".hydra" / "config.yaml", ref_dir / "hparams.yaml"]
    raw_config = next((OmegaConf.load(p) for p in config_paths if p.exists()), None)

    if not raw_config:
        print(f"Warning: Could not find config in {ref_dir}. Skipping dummy.")
        return

    if "config" in raw_config:
        raw_config = raw_config["config"]

    raw_config = resolve_config(raw_config)
    config = GraphLevelExperimentConfig.from_raw_config(raw_config)
    task_type = config.dataset.task_type

    # Synchronize seed
    seed = config.training.random_seed
    seed_everything(seed)

    datamodule = GraphLevelDataModule(
        dataset_config=config.dataset,
        batch_size=config.training.batch_size,
        pos_enc_path=config.pos_encoding.file if config.pos_encoding else None,
    )
    datamodule.setup(stage="fit")

    y_train, _ = get_targets_and_ids(datamodule.train_dataloader())
    y_val, _ = get_targets_and_ids(datamodule.val_dataloader())
    y_test, test_ids = get_targets_and_ids(datamodule.test_dataloader())
    y_total = np.concatenate([y_train, y_val, y_test])

    # EDA
    stats = {}
    total_samples = len(y_total)
    splits = {"total": y_total, "train": y_train, "val": y_val, "test": y_test}

    for split_name, y_data in splits.items():
        split_stats = {
            "count": len(y_data), 
            "percentage_of_total": len(y_data) / total_samples * 100 if total_samples > 0 else 0
        }
        
        if task_type == "binary":
            unique, counts = np.unique(y_data, return_counts=True)
            counts_dict = dict(zip(unique, counts))
            
            count_0 = int(counts_dict.get(0, 0))
            count_1 = int(counts_dict.get(1, 0))
            split_stats.update({
                "class_0_count": count_0,
                "class_1_count": count_1,
                "class_0_percentage": (count_0 / len(y_data)) * 100 if len(y_data) > 0 else 0.0,
                "class_1_percentage": (count_1 / len(y_data)) * 100 if len(y_data) > 0 else 0.0
            })
            
            plt.figure(figsize=(6, 4))
            sns.countplot(x=y_data, palette="Set2")
            plt.title(f"{split_name.capitalize()} Data Distribution")
            plt.savefig(eda_dir / f"{split_name}.png", bbox_inches="tight")
            plt.close()
            
        else: # regression
            s = pd.Series(y_data).describe().to_dict()
            split_stats.update({k: float(v) for k, v in s.items()})
            
            plt.figure(figsize=(8, 5))
            sns.histplot(y_data, bins=50, kde=True, color="steelblue")
            plt.title(f"{split_name.capitalize()} Target Distribution")
            plt.xlabel("Target Value")
            plt.savefig(eda_dir / f"{split_name}.png", bbox_inches="tight")
            plt.close()

        stats[split_name] = split_stats
    
    with open(eda_dir / "data_distribution.json", "w") as f:
        json.dump(stats, f, indent=4)

    # Evaluation
    metrics = {}
    start_test_time = time.perf_counter()

    if task_type == "binary":
        p_1 = (y_train == 1).mean()
        p_0 = 1.0 - p_1
        preds = np.random.choice([0, 1], size=len(y_test), p=[p_0, p_1])
        
        metrics.update({
            "test_AUROC": float(roc_auc_score(y_test, preds)),
            "test_Accuracy": float(accuracy_score(y_test, preds)),
            "test_F1": float(f1_score(y_test, preds, zero_division=0)),
            "test_Precision": float(precision_score(y_test, preds, zero_division=0)),
            "test_Recall": float(recall_score(y_test, preds, zero_division=0)),
            "test_loss": 0.0
        })

    elif task_type == "regression":
        median_val = np.median(y_train)
        preds = np.full_like(y_test, fill_value=median_val, dtype=float)
        
        metrics.update({
            "test_MAE": float(mean_absolute_error(y_test, preds)),
            "test_MSE": float(mean_squared_error(y_test, preds)),
            "test_R2": float(r2_score(y_test, preds)),
            "test_loss": 0.0
        })

    end_test_time = time.perf_counter()
    metrics.update({
        "debug_train_time_seconds": 0.0,
        "debug_train_epochs": 0,
        "debug_test_time_seconds": end_test_time - start_test_time
    })

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Export Wide CSV
    if not test_ids:
        print("Warning: Could not find 'origin_id' in test dataloader. Skipping wide CSV generation.")
    else:
        df_preds = pd.DataFrame({
            "origin_id": test_ids,
            "target_0": y_test,
            "prediction_0": preds
        })
        csv_path = dummy_dir / "test_predictions_wide.csv"
        df_preds.to_csv(csv_path, index=False)


def ensure_dummy_baselines():
    """Loops through expected grid to discover standard runs and triggers dummy generation."""
    print("Checking for missing dummy baselines...")
    for seed in SEEDS:
        for block in BLOCK_SPLITS:
            # 1. Binary
            for m in BASE_MODELS:
                p = BASE_DIR / str(seed) / "binary_classification" / m / f"block_3_{block}" / "results" / "lightning_logs" / f"version_{seed}"
                if p.exists() and (p / "metrics.json").exists():
                    create_dummy_baseline(p)
                    break
                    
            # 2. Regression
            for state in range(10):
                # F Regressor
                for m in BASE_MODELS:
                    glob_f = f"{BASE_DIR}/{seed}/f_regressor/{state}/{m}/*/*/block_3_{block}/results/lightning_logs/version_{seed}"
                    f_dirs = glob.glob(glob_f)
                    if f_dirs and (Path(f_dirs[0]) / "metrics.json").exists():
                        create_dummy_baseline(Path(f_dirs[0]))
                        break
                        
                # Lambda Regressor
                for m in BASE_MODELS:
                    glob_l = f"{BASE_DIR}/{seed}/lambda_regressor/{state}/{m}/*/*/block_3_{block}/results/lightning_logs/version_{seed}"
                    l_dirs = glob.glob(glob_l)
                    if l_dirs and (Path(l_dirs[0]) / "metrics.json").exists():
                        create_dummy_baseline(Path(l_dirs[0]))
                        break


# --- UTILITY & METRICS EXTRACTION FUNCTIONS ---

def format_metrics(metrics: dict) -> dict:
    formatted = {}
    for k, v in metrics.items():
        clean_k = k.replace("test_", "").replace("debug_", "")
        if isinstance(v, (int, float)):
            formatted[clean_k] = float(v)
        else:
            formatted[clean_k] = v
    return formatted

def compute_raw_mae(exp_dir: Path, target_type: str) -> float:
    std_csv = exp_dir / "test_predictions_wide.csv"
    if not std_csv.exists():
        raise FileNotFoundError(f"Missing prediction CSV in {exp_dir}")

    df = pd.read_csv(std_csv)
    t_col = [c for c in df.columns if c.startswith('target')][0]
    p_col = [c for c in df.columns if c.startswith('prediction')][0]
    
    if target_type == "lambda":
        t_vals = 1239.8419843320026224 / df[t_col].values
        p_vals = 1239.8419843320026224 / df[p_col].values
    elif target_type == "f":
        t_vals = 10 ** df[t_col].values
        p_vals = 10 ** df[p_col].values.clip(max=1)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    return float(np.mean(np.abs(t_vals - p_vals)))

def get_smoothed_spectrum(x_values, osc_strengths, is_wavelengths=False, sigma=0.25, x_min=0, x_max=12.0, n_points=5000, normalize=False):
    x_values = np.array(x_values, dtype=float)
    osc_strengths = np.array(osc_strengths, dtype=float)

    if is_wavelengths:
        energies = 1239.8419843320026224 / x_values
    else:
        energies = x_values

    ev_grid = np.linspace(x_min, x_max, n_points)
    spectrum = np.zeros_like(ev_grid)

    for e, f in zip(energies, osc_strengths):
        spectrum += f * np.exp(-0.5 * ((ev_grid - e) / sigma) ** 2)

    if normalize and spectrum.max() > 0:
        spectrum /= spectrum.max()

    return ev_grid, spectrum

def calculate_spectral_metrics_per_sample(f_pred, lambda_pred, f_true, lambda_true, sigma=0.25, n_points=5000, global_x_min=0.0, global_x_max=12.0, torch_device='cuda'):
    f_true_phys = 10 ** f_true
    f_pred_phys = 10 ** f_pred
    
    energies_true = lambda_true
    energies_pred = lambda_pred

    N = f_pred.shape[0]
    model_spectra_list, target_spectra_list = [], []
    
    for i in range(N):
        _, target_spec = get_smoothed_spectrum(energies_true[i], f_true_phys[i], sigma=sigma, x_min=global_x_min, x_max=global_x_max, n_points=n_points)
        _, model_spec = get_smoothed_spectrum(energies_pred[i], f_pred_phys[i], sigma=sigma, x_min=global_x_min, x_max=global_x_max, n_points=n_points)
        target_spectra_list.append(target_spec.tolist())
        model_spectra_list.append(model_spec)

    normalized_target_list = spectral_loss.pre_normalize_targets(targets=target_spectra_list, threshold=1e-8, torch_device=torch_device, batch_size=min(50, N))
    target_tensor = torch.tensor(normalized_target_list, dtype=torch.float32, device=torch_device)
    model_tensor = torch.tensor(np.array(model_spectra_list), dtype=torch.float32, device=torch_device)
    
    with torch.no_grad():
        metrics = {
            "metric_sid": spectral_loss.sid(model_tensor, target_tensor, torch_device=torch_device).cpu().numpy(),
            "metric_jsd": spectral_loss.jsd(model_tensor, target_tensor, torch_device=torch_device).cpu().numpy(),
            "metric_stmse": spectral_loss.stmse(model_tensor, target_tensor, torch_device=torch_device).cpu().numpy(),
            "metric_srmse": spectral_loss.srmse(model_tensor, target_tensor, torch_device=torch_device).cpu().numpy(),
            "metric_smse": spectral_loss.smse(model_tensor, target_tensor, torch_device=torch_device).cpu().numpy(),
            "metric_wasserstein": spectral_loss.wasserstein(model_tensor, target_tensor, torch_device=torch_device, x_min=global_x_min, x_max=global_x_max).cpu().numpy()
        }
    return metrics

def get_spectral_metrics_for_seed(seed: int, model: str, block: str) -> dict:
    f_true_mat, f_pred_mat = [], []
    l_true_mat, l_pred_mat = [], []
    train_times, test_times, epochs = [], [], []
    reg_accum_f = defaultdict(list)
    reg_accum_l = defaultdict(list)

    for state in range(10):
        f_glob = f"{BASE_DIR}/{seed}/f_regressor/{state}/{model}/*/*/block_3_{block}/results/lightning_logs/version_{seed}"
        l_glob = f"{BASE_DIR}/{seed}/lambda_regressor/{state}/{model}/*/*/block_3_{block}/results/lightning_logs/version_{seed}"
        
        f_dirs = glob.glob(f_glob)
        l_dirs = glob.glob(l_glob)

        if not f_dirs: return {"_Status": f"Missing F Regressor path for state {state}"}
        if not l_dirs: return {"_Status": f"Missing Lambda Regressor path for state {state}"}

        state_dir_f = Path(f_dirs[0])
        state_dir_l = Path(l_dirs[0])

        csv_f, csv_l = state_dir_f / "test_predictions_wide.csv", state_dir_l / "test_predictions_wide.csv"
        metrics_f, metrics_l = state_dir_f / "metrics.json", state_dir_l / "metrics.json"

        if not csv_f.exists() or not csv_l.exists() or not metrics_f.exists() or not metrics_l.exists():
            return {"_Status": f"Missing metrics/CSV for state {state}"}

        df_f = pd.read_csv(csv_f)
        df_l = pd.read_csv(csv_l)
        
        t_col_f = [c for c in df_f.columns if c.startswith('target')][0]
        p_col_f = [c for c in df_f.columns if c.startswith('prediction')][0]
        t_col_l = [c for c in df_l.columns if c.startswith('target')][0]
        p_col_l = [c for c in df_l.columns if c.startswith('prediction')][0]
        
        f_true_mat.append(df_f[t_col_f].values)
        f_pred_mat.append(df_f[p_col_f].values)
        l_true_mat.append(df_l[t_col_l].values)
        l_pred_mat.append(df_l[p_col_l].values)
        
        with open(metrics_f, "r") as f: mf = json.load(f)
        with open(metrics_l, "r") as f: ml = json.load(f)

        reg_accum_f["raw_mae"].append(compute_raw_mae(state_dir_f, "f"))
        reg_accum_l["raw_mae"].append(compute_raw_mae(state_dir_l, "lambda"))

        train_times.append(mf.get("debug_train_time_seconds", 0) + ml.get("debug_train_time_seconds", 0))
        test_times.append(mf.get("debug_test_time_seconds", 0) + ml.get("debug_test_time_seconds", 0))
        epochs.append(mf.get("debug_train_epochs", 0) + ml.get("debug_train_epochs", 0))

        for k, v in mf.items():
            if not k.startswith("debug_") and isinstance(v, (int, float)):
                reg_accum_f[k].append(v)
        for k, v in ml.items():
            if not k.startswith("debug_") and isinstance(v, (int, float)):
                reg_accum_l[k].append(v)

    f_true_mat = np.array(f_true_mat).T
    f_pred_mat = np.array(f_pred_mat).T
    l_true_mat = np.array(l_true_mat).T
    l_pred_mat = np.array(l_pred_mat).T

    raw_spectral = calculate_spectral_metrics_per_sample(f_pred_mat, l_pred_mat, f_true_mat, l_true_mat)
    final_metrics = {k: float(np.mean(v)) for k, v in raw_spectral.items()}
    
    final_metrics["train_time_seconds"] = sum(train_times)
    final_metrics["test_time_seconds"] = sum(test_times)
    final_metrics["train_epochs"] = sum(epochs)
    final_metrics["_Status"] = "OK"

    for metric_name, metric_values in reg_accum_f.items():
        final_metrics[f"F_{metric_name}"] = float(np.mean(metric_values))
    for metric_name, metric_values in reg_accum_l.items():
        final_metrics[f"Lambda_{metric_name}"] = float(np.mean(metric_values))

    return final_metrics

# --- DATA EXTRACTION PIPELINE ---

def extract_experiment_data() -> Tuple[List[Dict], List[Dict]]:
    binary_records = []
    regression_records = []

    for seed in SEEDS:
        for model in MODELS:
            for block in BLOCK_SPLITS:
                # 1. Binary Classification
                bin_path = BASE_DIR / str(seed) / "binary_classification" / model / f"block_3_{block}" / "results" / "lightning_logs" / f"version_{seed}"
                
                record_bin = {
                    "Dataset": "TMQM_SPECTO_BINARY",
                    "Model": model,
                    "Seed": seed,
                    "Block_Split": block
                }
                
                metrics_file = bin_path / "metrics.json"
                if metrics_file.exists():
                    record_bin["_Status"] = "OK"
                    with open(metrics_file, "r") as f:
                        record_bin.update(format_metrics(json.load(f)))
                else:
                    record_bin["_Status"] = "Missing"
                    
                binary_records.append(record_bin)

                # 2. Regression (Iterate 10 states)
                for state in range(10):
                    # F Regressor
                    f_glob = f"{BASE_DIR}/{seed}/f_regressor/{state}/{model}/*/*/block_3_{block}/results/lightning_logs/version_{seed}"
                    f_dirs = glob.glob(f_glob)
                    
                    record_f = {
                        "Dataset": "TMQM_SPECTO_F_REGRESSOR",
                        "Model": model,
                        "Seed": seed,
                        "State": state,
                        "Block_Split": block
                    }
                    if f_dirs and (Path(f_dirs[0]) / "metrics.json").exists():
                        exp_dir = Path(f_dirs[0])
                        record_f["_Status"] = "OK"
                        with open(exp_dir / "metrics.json", "r") as f:
                            record_f.update(format_metrics(json.load(f)))
                        try:
                            record_f["raw_mae"] = compute_raw_mae(exp_dir, "f")
                        except FileNotFoundError:
                            pass
                    else:
                        record_f["_Status"] = "Missing"
                    regression_records.append(record_f)

                    # Lambda Regressor
                    l_glob = f"{BASE_DIR}/{seed}/lambda_regressor/{state}/{model}/*/*/block_3_{block}/results/lightning_logs/version_{seed}"
                    l_dirs = glob.glob(l_glob)
                    
                    record_l = {
                        "Dataset": "TMQM_SPECTO_LAMBDA_REGRESSOR",
                        "Model": model,
                        "Seed": seed,
                        "State": state,
                        "Block_Split": block
                    }
                    if l_dirs and (Path(l_dirs[0]) / "metrics.json").exists():
                        exp_dir = Path(l_dirs[0])
                        record_l["_Status"] = "OK"
                        with open(exp_dir / "metrics.json", "r") as f:
                            record_l.update(format_metrics(json.load(f)))
                        try:
                            record_l["raw_mae"] = compute_raw_mae(exp_dir, "lambda")
                        except FileNotFoundError:
                            pass
                    else:
                        record_l["_Status"] = "Missing"
                    regression_records.append(record_l)

                # 3. Spectral Metrics (Combined 10 states)
                record_spectral = {
                    "Dataset": "TMQM_SPECTO_SPECTRAL",
                    "Model": model,
                    "Seed": seed,
                    "State": None,
                    "Block_Split": block
                }
                
                specto_metrics = get_spectral_metrics_for_seed(seed, model, block)
                record_spectral.update(specto_metrics)
                regression_records.append(record_spectral)

    return binary_records, regression_records

def print_summary(df: pd.DataFrame, task_name: str):
    print(f"\n{'='*10} {task_name.upper()} PARSING SUMMARY {'='*10}")
    if df.empty:
        print("No records found.")
        return
        
    ok_df = df[df['_Status'] == 'OK']
    print(f"Total valid 'OK' experiments extracted: {len(ok_df)} (out of {len(df)} expected valid configurations)")
    
    print("\n--- Experiments by Dataset (Present / Expected) ---")
    for ds in df['Dataset'].unique():
        expected = len(df[df['Dataset'] == ds])
        present = len(ok_df[ok_df['Dataset'] == ds])
        print(f"  - {ds}: {present} / {expected}")

    issues_found = False
    for mod in sorted(df['Model'].unique()):
        mod_df = df[df['Model'] == mod]
        for ds in sorted(mod_df['Dataset'].unique()):
            sub_df = mod_df[mod_df['Dataset'] == ds]
            expected = len(sub_df)
            present = len(sub_df[sub_df['_Status'] == 'OK'])
            
            if present < expected:
                issues_found = True
                print(f"\nModel: {mod} | Dataset: {ds}")
                print(f"  [!] Missing Runs: {present}/{expected} records found.")
                for seed in sub_df['Seed'].unique():
                    seed_df = sub_df[sub_df['Seed'] == seed]
                    seed_expected = len(seed_df)
                    seed_present = len(seed_df[seed_df['_Status'] == 'OK'])
                    if seed_present < seed_expected:
                        missing_splits = seed_df[seed_df['_Status'] != 'OK']['Block_Split'].unique()
                        print(f"      - Seed {seed}: Missing Block Splits: {list(missing_splits)}")

    if not issues_found:
        print("\nAll expected combinations and metrics are fully present! 🎉")
    print("-" * 50)


# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Parse Experiment Results to Excel")
    parser.add_argument("--run-dummy", action=argparse.BooleanOptionalAction, default=True, help="Create dummy metrics automatically for found datasets")
    args = parser.parse_args()

    if args.run_dummy:
        ensure_dummy_baselines()

    print("Scanning directories and scraping experiments...")
    bin_records, reg_records = extract_experiment_data()

    df_bin = pd.DataFrame(bin_records)
    df_reg = pd.DataFrame(reg_records)

    print_summary(df_bin, "Binary")
    print_summary(df_reg, "Regression")

    output_file = "experiment_results.xlsx"
    json_output_file = "experiment_results.json"

    with pd.ExcelWriter(output_file) as writer:
        if not df_bin.empty:
            df_bin.to_excel(writer, sheet_name="Binary", index=False)
        if not df_reg.empty:
            df_reg.to_excel(writer, sheet_name="Regression", index=False)

    print(f"\nSaving raw results to {json_output_file}...")
    
    combined_json_data = {
        "binary_results": json.loads(df_bin.to_json(orient="records")) if not df_bin.empty else [],
        "regression_results": json.loads(df_reg.to_json(orient="records")) if not df_reg.empty else [],
    }
    
    with open(json_output_file, "w") as f:
        json.dump(combined_json_data, f, indent=4)

    print(f"\nDone! Results saved to {output_file} and {json_output_file}")

if __name__ == "__main__":
    main()