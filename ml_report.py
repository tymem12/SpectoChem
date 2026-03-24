import pandas as pd
import re
import json
import yaml
import joblib
from pathlib import Path

def get_feature_count(n, l):
    species = 14
    count = (l + 1) * (species * n * (species * n + 1)) // 2
    return f"{count/1000:.1f}k"

def generate_report(root_dir="ml_experiments/benchmark_results"):
    results = []
    failures = []
    root = Path(root_dir)
    
    folder_pattern = re.compile(r"r([\d.]+)_n(\d+)_l(\d+)")

    for folder in root.iterdir():
        match = folder_pattern.search(folder.name)
        if not match:
            continue
            
        r_cut, n_max, l_max = float(match.group(1)), int(match.group(2)), int(match.group(3))
        config_label = f"r{r_cut}_n{n_max}_l{l_max}"
        feat_count = get_feature_count(n_max, l_max)
        
        benchmark_dirs = list(folder.glob("benchmark_*"))
        if len(benchmark_dirs) != 1:
            raise RuntimeError(f"Expected 1 benchmark dir in {folder}, found {len(benchmark_dirs)}")
        
        bench_root = benchmark_dirs[0]
        base_path = bench_root / "binary_xgboost_soap_outlierFalse_tuneTrue_seed42"
        
        csv_path = base_path / "results" / "model_metrics_binary.csv"
        json_path = bench_root / "benchmark_summary_intermediate.json"
        yaml_path = base_path / "config.yaml"
        model_path = base_path / "models" / "classifier_has_uvvis_peak.pkl"
        
        if csv_path.exists():
            try:
                # 1. Read Metrics
                df = pd.read_csv(csv_path, index_col=0)
                metrics = df.iloc[0].to_dict()
                
                # 2. Extract Tuned Hyperparameters
                tuned_params_found = {}
                if yaml_path.exists() and model_path.exists():
                    try:
                        with open(yaml_path, 'r') as yf:
                            yaml_config = yaml.safe_load(yf)
                        
                        param_dist = yaml_config.get("tuning", {}).get("xgboost", {}).get("param_distributions", {})
                        search_keys = {k for k, v in param_dist.items() if isinstance(v, list) and len(v) > 1}
                        
                        with open(model_path, 'rb') as mf:
                            model = joblib.load(mf)
                        
                        model_params = model.get_params() if hasattr(model, "get_params") else getattr(model, "__dict__", {})
                        
                        for k_search in search_keys:
                            for k_mod, v_mod in model_params.items():
                                if k_mod == k_search or k_mod.endswith(f"__{k_search}"):
                                    val = round(v_mod, 4) if isinstance(v_mod, float) else v_mod
                                    tuned_params_found[k_search] = val
                                    break
                    except Exception as e:
                        tuned_params_found = {"Error": f"Failed to extract params: {e}"}

                results.append({
                    "Config": config_label, 
                    "Features": feat_count, 
                    "BestParams": tuned_params_found,
                    **metrics
                })
            except Exception as e:
                failures.append((config_label, f"CSV Read Error: {str(e)}"))
        else:
            error_msg = "Config failed (Reason unknown/No files found)"
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        error_msg = data["failed"][0]["error"]
                except (IndexError, KeyError, json.JSONDecodeError, TypeError):
                    error_msg = "Config failed (JSON structure invalid or empty)"
            
            failures.append((config_label, error_msg))

    # --- PROCESS AND PRINT SUCCESS TABLE ---
    if results:
        report_df = pd.DataFrame(results).sort_values(by="f1", ascending=False)
        n_total_configs = len(report_df)
        
        # 3. Aggregate Shared Configs and Max F1 scores
        param_counts = {}
        param_max_f1 = {}
        
        for _, row in report_df.iterrows():
            p_dict = row.get("BestParams", {})
            if p_dict and "Error" not in p_dict:
                p_hash = tuple(sorted(p_dict.items()))
                param_counts[p_hash] = param_counts.get(p_hash, 0) + 1
                param_max_f1[p_hash] = max(param_max_f1.get(p_hash, -1.0), row['f1'])
        
        # 4. Name the Shared Configs Based on Frequencies and F1 scores
        shared_hashes = [ph for ph, count in param_counts.items() if count >= 2]
        # Sort by frequency (desc), then by max_f1 (desc) to break ties deterministically
        shared_hashes.sort(key=lambda ph: (param_counts[ph], param_max_f1[ph]), reverse=True)
        
        shared_configs_info = {}
        counts_seen = {}
        
        for ph in shared_hashes:
            c = param_counts[ph]
            counts_seen[c] = counts_seen.get(c, 0) + 1
            idx = counts_seen[c]
            
            n_pct = (c / n_total_configs) * 100
            pct_str = f"[{c}/{n_total_configs} ({n_pct:.2f}%)]"
            
            name = f"[Shared Config by {pct_str}]" if idx == 1 else f"[Shared Config by {idx} {pct_str}]"
            shared_configs_info[ph] = {
                "name": name,
                "count": c,
                "max_f1": param_max_f1[ph],
                "dict": dict(ph)
            }

        print("\n" + "="*145)
        print(f"{'SOAP CONFIG':<20} | {'FEATS':<8} | {'F1':<6} | {'ROC-AUC':<8} | {'ACC':<6} | {'TUNED HYPERPARAMS'}")
        print("-" * 145)
        
        for _, row in report_df.iterrows():
            p_dict = row.get("BestParams", {})
            p_hash = tuple(sorted(p_dict.items())) if p_dict and "Error" not in p_dict else None
            
            # --- Determine Output Lines for Hyperparams ---
            if p_hash in shared_configs_info:
                # Exact match for a shared config
                p_lines = [shared_configs_info[p_hash]["name"]]
            
            elif p_hash is not None and shared_configs_info:
                # Unshared config: look for closest match in shared configs
                best_cand_name = None
                best_diffs = None
                # criteria: (-diff_count, count, max_f1) -> we want to maximize this
                best_criteria = (-float('inf'), -1, -1.0)
                
                for sh_hash, sh_info in shared_configs_info.items():
                    sh_dict = sh_info["dict"]
                    all_keys = set(p_dict.keys()).union(sh_dict.keys())
                    
                    diffs = {}
                    for k in all_keys:
                        v1 = sh_dict.get(k)
                        v2 = p_dict.get(k)
                        if v1 != v2:
                            diffs[k] = (v1, v2)
                    
                    diff_count = len(diffs)
                    if diff_count <= 3: # Allowed difference threshold
                        criteria = (-diff_count, sh_info["count"], sh_info["max_f1"])
                        if criteria > best_criteria:
                            best_criteria = criteria
                            best_cand_name = sh_info["name"]
                            best_diffs = diffs
                
                if best_cand_name:
                    p_lines = [f"~ {best_cand_name} (diff):"]
                    for k, (v1, v2) in best_diffs.items():
                        p_lines.append(f"  {k}: {v1} -> {v2}")
                else:
                    # More than 3 differences or no shared configs exist
                    p_lines = [f"{k}: {v}" for k, v in p_dict.items()] if p_dict else ["None/Fixed"]
            
            else:
                p_lines = [f"{k}: {v}" for k, v in p_dict.items()] if p_dict else ["None/Fixed"]
            
            # Print the formatted row
            print(f"{row['Config']:<20} | {row['Features']:<8} | {row['f1']:.3f}  |  {row['roc_auc']:.3f}   |  {row['accuracy']:.3f} | {p_lines[0]}")
            for p in p_lines[1:]:
                print(f"{'':<20} | {'':<8} | {'':<6} | {'':<8} | {'':<6} | {p}")
            
            print("-" * 145)
            
        # 5. Print the Shared Config Legend at the Bottom
        if shared_configs_info:
            print("\n" + "--- SHARED HYPERPARAMETER CONFIGS ---")
            for ph, info in shared_configs_info.items():
                print(f"{info['name']} (Best F1: {info['max_f1']:.3f}):")
                for k, v in ph:
                    print(f"    {k}: {v}")
                print()
    
    # --- PRINT FAILURE SECTION ---
    if failures:
        print("\n" + "!"*20 + " FAILED EXPERIMENTS " + "!"*20)
        for config, reason in failures:
            print(f"FAILED: {config}\nREASON: {reason}\n{'-'*50}")

if __name__ == "__main__":
    generate_report()