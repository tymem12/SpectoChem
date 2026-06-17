import os
import json
import glob
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Tuple, List, Dict

import pandas as pd
import numpy as np
import torch

from gjepa.utils import spectral_loss

# --- EXPERIMENT CONFIGURATION ---
SEEDS = [2137, 42, 1234]
MODELS = ["schnet", "gine", "gat", "gcn"]
BLOCK_SPLITS = ["none", "test"]
BASE_DIR = Path("supervised")

# --- UTILITY FUNCTIONS ---

def format_metrics(metrics: dict) -> dict:
    """Removes 'test_' and 'debug_' prefix from metrics and passes floats."""
    formatted = {}
    for k, v in metrics.items():
        clean_k = k.replace("test_", "").replace("debug_", "")
        if isinstance(v, (int, float)):
            formatted[clean_k] = float(v)
        else:
            formatted[clean_k] = v
    return formatted

def compute_raw_mae(exp_dir: Path, target_type: str) -> float:
    """Reads predictions and computes raw MAE, applying physical conversions."""
    std_csv = exp_dir / "test_predictions_wide.csv"
    
    if not std_csv.exists():
        raise FileNotFoundError(f"Missing prediction CSV in {exp_dir}")

    df = pd.read_csv(std_csv)
    t_col = [c for c in df.columns if c.startswith('target')][0]
    p_col = [c for c in df.columns if c.startswith('prediction')][0]
    
    if target_type == "lambda":
        # Convert eV to Lambda (nm)
        t_vals = 1239.8419843320026224 / df[t_col].values
        p_vals = 1239.8419843320026224 / df[p_col].values
    elif target_type == "f":
        # Convert log10(f) to raw f
        t_vals = 10 ** df[t_col].values
        p_vals = 10 ** df[p_col].values.clip(max=1)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    return float(np.mean(np.abs(t_vals - p_vals)))


# --- SPECTRAL METRICS ---

def get_smoothed_spectrum(
    x_values, osc_strengths, is_wavelengths=False, sigma=0.25,
    x_min=0, x_max=12.0, n_points=5000, normalize=False
):
    """Generates a continuous smoothed spectrum strictly in the eV domain."""
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

def calculate_spectral_metrics_per_sample(
    f_pred, lambda_pred, f_true, lambda_true, 
    sigma=0.25, n_points=5000, global_x_min=0.0, global_x_max=12.0, torch_device='cuda'
):
    """Calculates native spectral metrics using PyTorch."""
    f_true_phys = 10 ** f_true
    f_pred_phys = 10 ** f_pred
    
    energies_true = lambda_true
    energies_pred = lambda_pred

    N = f_pred.shape[0]
    model_spectra_list, target_spectra_list = [], []
    
    for i in range(N):
        _, target_spec = get_smoothed_spectrum(
            energies_true[i], f_true_phys[i], 
            sigma=sigma, x_min=global_x_min, x_max=global_x_max, n_points=n_points
        )
        _, model_spec = get_smoothed_spectrum(
            energies_pred[i], f_pred_phys[i], 
            sigma=sigma, x_min=global_x_min, x_max=global_x_max, n_points=n_points
        )
        target_spectra_list.append(target_spec.tolist())
        model_spectra_list.append(model_spec)

    normalized_target_list = spectral_loss.pre_normalize_targets(
        targets=target_spectra_list, threshold=1e-8, torch_device=torch_device, batch_size=min(50, N)
    )

    target_tensor = torch.tensor(normalized_target_list, dtype=torch.float32, device=torch_device)
    model_tensor = torch.tensor(np.array(model_spectra_list), dtype=torch.float32, device=torch_device)
    
    with torch.no_grad():
        metrics = {
            "metric_sid": spectral_loss.sid(model_tensor, target_tensor, torch_device=torch_device).cpu().numpy(),
            "metric_jsd": spectral_loss.jsd(model_tensor, target_tensor, torch_device=torch_device).cpu().numpy(),
            "metric_stmse": spectral_loss.stmse(model_tensor, target_tensor, torch_device=torch_device).cpu().numpy(),
            "metric_srmse": spectral_loss.srmse(model_tensor, target_tensor, torch_device=torch_device).cpu().numpy(),
            "metric_smse": spectral_loss.smse(model_tensor, target_tensor, torch_device=torch_device).cpu().numpy(),
            "metric_wasserstein": spectral_loss.wasserstein(
                model_tensor, target_tensor, torch_device=torch_device, x_min=global_x_min, x_max=global_x_max
            ).cpu().numpy()
        }

    return metrics


def get_spectral_metrics_for_seed(seed: int, model: str, block: str) -> dict:
    """Calculates aggregated spectral metrics across all 10 states for a given seed/model/block."""
    f_true_mat, f_pred_mat = [], []
    l_true_mat, l_pred_mat = [], []
    train_times, test_times, epochs = [], [], []
    
    reg_accum_f = defaultdict(list)
    reg_accum_l = defaultdict(list)

    for state in range(10):
        # Resolve the dynamic folder structure with wildcards (for std, norm, f_log)
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


# --- CORE EXTRACTION PIPELINE ---

def extract_experiment_data() -> Tuple[List[Dict], List[Dict]]:
    binary_records = []
    regression_records = []

    for seed in SEEDS:
        for model in MODELS:
            for block in BLOCK_SPLITS:
                # ----------------------------------------------------
                # 1. Binary Classification
                # ----------------------------------------------------
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

                # ----------------------------------------------------
                # 2. Regression (Iterate 10 states)
                # ----------------------------------------------------
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

                # ----------------------------------------------------
                # 3. Spectral Metrics (Combined 10 states)
                # ----------------------------------------------------
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
    """Prints a parsing summary showing missing results."""
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
    print("Scanning directories and scraping experiments...")
    bin_records, reg_records = extract_experiment_data()

    df_bin = pd.DataFrame(bin_records)
    df_reg = pd.DataFrame(reg_records)

    print_summary(df_bin, "Binary")
    print_summary(df_reg, "Regression")

    output_file = "experiment_results.xlsx"
    json_output_file = "experiment_results.json"

    # Save to Excel
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