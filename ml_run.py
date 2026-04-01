import os
import yaml
import copy
import argparse
import subprocess

def run_cmd(cmd, description):
    """Helper to execute the command."""
    print(f"\n--- Running: {description} ---")
    subprocess.run(cmd)

def run_binary(model_name: str, search_space: dict[str, list], r_cut, n_max, l_max, seed, block_3_split):
    # Ensure config directory exists
    os.makedirs("ml_configs", exist_ok=True)
    
    # Load default config
    with open("xgboost_training/config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)

    if model_name == "logistic_regression":
        search_space = dict(
            classification=search_space
        )

    if model_name != "xgboost":
        cv_n_jobs = -1
    else:
        cv_n_jobs = 1

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
            config['experiments'][0]['model_type'] = model_name
            config['experiments'][0]['tune'] = False
            
            # Inject SOAP parameters
            config['soap']['r_cut'] = r_cut
            config['soap']['n_max'] = n_max
            config['soap']['l_max'] = l_max

            tuning_params = config['tuning'][model_name]

            tuning_params['cv'] = 5

            # Set n_iter to a huge number to force full Grid Search behavior
            tuning_params['n_iter'] = 9999999
            tuning_params["param_distributions"] = search_space

            tuning_params['n_jobs'] = cv_n_jobs

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
            
            run_cmd(cmd, f"{model_name.upper()} | SOAP: {config_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XGBoost benchmark with specific SOAP parameters.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--r_cut", type=float, required=True, help="SOAP cutoff radius")
    parser.add_argument("--n_max", type=int, required=True, help="Number of radial basis functions")
    parser.add_argument("--l_max", type=int, required=True, help="Maximum degree of spherical harmonics")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--block_3_split", type=str, default="none", help="Block 3 split mode ('none' or 'test')")
    
    args = parser.parse_args()
    
    model_name = args.model_name

    match model_name:
        case "xgboost":
            search_space = {
                'n_jobs': [-1],
                'n_estimators': [4000],
                'max_depth': [15, 20],
                'learning_rate': [0.001, 0.01],
                'subsample': [0.9],
                'colsample_bytree': [0.9]
            }
        case "random_forest":
            search_space = {
                "n_estimators": [50, 200, 500, 1000],
                "max_depth": [3, 5, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 5, 10],
            }
        case "svm":
            search_space = {
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"]
            }
        case "logistic_regression":
            search_space = {
                "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "max_iter": [100, 500, 1000]
            }
        case "mlp":
            search_space = {
                "hidden_layer_sizes": [[32], [64], [128], [32, 32], [64, 64]],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate_init": [0.001, 0.01],
                "solver": ["adam"],
                "early_stopping": [True]
            }
        case _:
            raise ValueError(f"Unsupported model {model_name!r}")

    run_binary(model_name, search_space, args.r_cut, args.n_max, args.l_max, args.seed, args.block_3_split)