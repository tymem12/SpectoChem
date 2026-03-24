import os
import yaml
import copy
import argparse
import subprocess

def run_cmd(cmd, description):
    """Helper to execute the command."""
    print(f"\n--- Running: {description} ---")
    subprocess.run(cmd)

def run_binary(r_cut, n_max, l_max, seed, block_3_split):
    # Ensure config directory exists
    os.makedirs("ml_configs", exist_ok=True)
    
    # Load default config
    with open("xgboost_training/config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)

    for min_f_value in [0.01]:
        for metric, metric_mode in [('F1', 'max')]:
            block_3_only = block_3_split == "none"
            split_val = block_3_split if block_3_split != "none" else 'null'
            
            # 1. Define the unique config string based on SOAP params + DS params
            config_str = f"r{r_cut}_n{n_max}_l{l_max}__seed-{seed}_block-3-{block_3_split}"
            
            # 2. Modify the config dict
            config = copy.deepcopy(base_config)
            
            # Update paths
            config['hydra_config_dir'] = "../../../config"
            config['output']['models_dir'] = f"ml_experiments/models/{config_str}"
            config['output']['results_dir'] = f"ml_experiments/results/{config_str}"
            
            # Update model and tuning settings
            config['experiments'][0]['model_type'] = "xgboost"
            config['experiments'][0]['tune'] = True
            
            # Inject SOAP parameters
            config['soap']['r_cut'] = r_cut
            config['soap']['n_max'] = n_max
            config['soap']['l_max'] = l_max

            config['tuning']['xgboost']['cv'] = 5

            # Set n_iter to a huge number to force full Grid Search behavior
            config['tuning']['xgboost']['n_iter'] = 9999999
            config['tuning']['xgboost']["param_distributions"] = {
                'n_jobs': [-1],
                'n_estimators': [200, 500, 1000],
                'max_depth': [3, 5, 8],
                'learning_rate': [0.01, 0.05, 0.3],
                'subsample': [0.7, 0.9],
                'colsample_bytree': [0.3, 0.6, 0.9]
            }

            config['tuning']['xgboost']['n_jobs'] = 1

            # 3. Save the modified config
            config_path = f"ml_configs/{config_str}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # 4. Execute the new command
            cmd = [
                "python", "xgboost_training/benchmark.py",
                "--config", config_path,
                "+exp=TMQM_SPECTO_BINARY", # Assuming you still need the base Hydra experiment
                f"training.random_seed={seed}",
                f"dataset.block_3_split_mode={split_val}",
                f"dataset.additional_loading_params.min_f_value={min_f_value}",
                f"dataset.additional_loading_params.block_3_only={block_3_only}",
                f"dataset.main_metric={metric}",
                f"dataset.metric_mode={metric_mode}"
            ]
            
            run_cmd(cmd, f"XGBOOST | SOAP: {config_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XGBoost benchmark with specific SOAP parameters.")
    parser.add_argument("--r_cut", type=float, required=True, help="SOAP cutoff radius")
    parser.add_argument("--n_max", type=int, required=True, help="Number of radial basis functions")
    parser.add_argument("--l_max", type=int, required=True, help="Maximum degree of spherical harmonics")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--block_3_split", type=str, default="none", help="Block 3 split mode ('none' or 'test')")
    
    args = parser.parse_args()
    
    run_binary(args.r_cut, args.n_max, args.l_max, args.seed, args.block_3_split)