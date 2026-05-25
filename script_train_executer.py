from html import parser
import subprocess
import datetime
import argparse
import sys



# 1. Dynamic Timestamp
now = datetime.datetime.now()
data_time = now.strftime("%Y-%m-%d_%H.%M.%S")
int_inter = 0  # Global counter

LAST_EXP = -1

def get_config_string(config):
    """Creates a short folder string representation of the config."""
    parts = []
    for k, v in config.items():
        short_key = k.replace("use_", "").replace("projection", "proj").replace("embedding", "emb").replace("interaction", "inter")
        parts.append(f"{short_key}-{v}")
    return "_".join(parts)

def get_backbone_args(config):
    """Creates the list of command line arguments for the backbone config."""
    return [f"model.backbone.{k}={v}" for k, v in config.items()]

LAMBDA_OUTLIER_THRESHOLD = 1327.9505300000078
OUTLIER_STRATEGY = 'whole-compound-outlier-removal'

def run_cmd(cmd, log_info):
    """Centralized function to execute a command and manage the counter."""
    global int_inter
    if int_inter > LAST_EXP:
        print(f"\n[{int_inter}] STARTING: {log_info}")
        try:
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"Error: Command [{int_inter}] failed with code {result.returncode}")
        except Exception as e:
            print(f"Script Error on [{int_inter}]: {e}")
    else:
        print(f"[{int_inter}] SKIPPING: {log_info}")
    
    int_inter += 1

def run_binary(model_name, seed, block_3_split):
    for min_f_value in [0.01]:
        for metric, metric_mode in [('AUROC', 'max')]:
            # experiment_path = f"supervised/{data_time}/binary_classification/{model_name}/{exp_param_str}/UMA_full_embedding/min_f_value_{min_f_value}/metric_{metric}/330-650/results"
            experiment_path = f"supervised/{seed}/binary_classification/{model_name}/block_3_{block_3_split}/results"
            block_3_only = block_3_split == "none"
            block_3_split = block_3_split if block_3_split != "none" else 'null'
            

            f_outlier_threshold = 'null'
            cmd = [
                "python", "experiments/scripts/train_graph_level.py",
                "+exp=TMQM_SPECTO_BINARY",
                "model=supervised_graph_level",
                f"backbone@model.backbone={model_name}",
                f"training.experiment_name={experiment_path}",
                f"training.random_seed={seed}",
                f"dataset.block_3_split_mode={block_3_split}",
                f"dataset.additional_loading_params.min_f_value={min_f_value}",
                f"dataset.additional_loading_params.block_3_only={block_3_only}",
                f"dataset.main_metric={metric}",
                f"dataset.metric_mode={metric_mode}",
                f"dataset.additional_loading_params.outlier_strategy={OUTLIER_STRATEGY}",
                f"dataset.additional_loading_params.lambda_outlier_threshold={LAMBDA_OUTLIER_THRESHOLD}",
                f"dataset.additional_loading_params.f_outlier_threshold={f_outlier_threshold}",

            ]
            run_cmd(cmd, f"{model_name} | BINARY | f={min_f_value} | {metric}")


def run_lambda_regression(model_name, seed,
                          standarization: bool=True, normalize_eV: bool = True,
                          block_3_split = 'test'):
    exp_type_log = f"LAMBDA_REGRESSION"
    block_3_only = block_3_split == "none"
    block_3_split = block_3_split if block_3_split != "none" else 'null'
    min_f_value = -1

    f_outlier_threshold = 'null'
    sort_by_max_f = False

    f_as_log10 = False

    for num_pair in range(0, 10):
        experiment_path = f"supervised/{seed}/lambda_regressor/{num_pair}/{model_name}/std-{standarization}/norm_to_eV-{normalize_eV}_f-as-log10-{f_as_log10}/block_3_{block_3_split}/results"

        cmd = [
            "python", "experiments/scripts/train_graph_level.py",
            "+exp=TMQM_SPECTO_LAMBDA_REGRESSOR",
            "model=supervised_graph_level",
            f"backbone@model.backbone={model_name}",
            f"training.experiment_name={experiment_path}",
            f"training.random_seed={seed}",
            f"dataset.block_3_split_mode={block_3_split}",
            f"dataset.additional_loading_params.num_states={num_pair}",
            f"dataset.additional_loading_params.min_f_value={min_f_value}",
            f"dataset.additional_loading_params.filter_f_value={min_f_value}",
            f"dataset.additional_loading_params.sort_by_max_f={sort_by_max_f}",
            f"dataset.additional_loading_params.outlier_strategy={OUTLIER_STRATEGY}",
            f"dataset.additional_loading_params.standarize_lambda={standarization}",
            f"dataset.additional_loading_params.lambda_outlier_threshold={LAMBDA_OUTLIER_THRESHOLD}",
            f"dataset.additional_loading_params.f_outlier_threshold={f_outlier_threshold}",
            f"dataset.additional_loading_params.convert_to_ev={normalize_eV}",
            f"dataset.additional_loading_params.f_as_log10={f_as_log10}",
            f"dataset.additional_loading_params.block_3_only={block_3_only}",
            "dataset.additional_loading_params.filter_type=all_samples"
        ]
        
        run_cmd(cmd, f"{model_name} | {exp_type_log} | num_pair={num_pair}")

def run_f_regression(model_name, seed, standarization, f_as_log10: bool, block_3_split):
    exp_type_log = f"F_REGRESSION"
    block_3_only = block_3_split == "none"
    block_3_split = block_3_split if block_3_split != "none" else 'null'

    min_f_value = -1
    f_outlier_threshold = 'null'
    sort_by_max_f = False
    normalize_eV = False    

    for num_pair in range(0, 10):
        experiment_path = f"supervised/{seed}/f_regressor/{num_pair}/{model_name}/std-{standarization}/norm_to_eV-{normalize_eV}_f-as-log10-{f_as_log10}/block_3_{block_3_split}/results"

        cmd = [
            "python", "experiments/scripts/train_graph_level.py",
            "+exp=TMQM_SPECTO_F_REGRESSOR",
            "model=supervised_graph_level",
            f"backbone@model.backbone={model_name}",
            f"training.experiment_name={experiment_path}",
            f"training.random_seed={seed}",
            f"dataset.block_3_split_mode={block_3_split}",
            f"dataset.additional_loading_params.num_states={num_pair}",
            f"dataset.additional_loading_params.min_f_value={min_f_value}",
            f"dataset.additional_loading_params.filter_f_value={min_f_value}",
            f"dataset.additional_loading_params.sort_by_max_f={sort_by_max_f}",
            f"dataset.additional_loading_params.outlier_strategy={OUTLIER_STRATEGY}",
            f"dataset.additional_loading_params.standarize_f={standarization}",
            f"dataset.additional_loading_params.lambda_outlier_threshold={LAMBDA_OUTLIER_THRESHOLD}",
            f"dataset.additional_loading_params.f_outlier_threshold={f_outlier_threshold}",
            f"dataset.additional_loading_params.convert_to_ev={normalize_eV}",
            f"dataset.additional_loading_params.f_as_log10={f_as_log10}",
            f"dataset.additional_loading_params.block_3_only={block_3_only}",
            "dataset.additional_loading_params.filter_type=all_samples"
        ]
        
        run_cmd(cmd, f"{model_name} | {exp_type_log} | num_pair={num_pair}")

# --- MAIN EXECUTION LOOP ---
def main():
    parser = argparse.ArgumentParser(description="Run specific experiments with specific models.")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model (e.g., invariant_projection, geometric_gating, moe)")
    parser.add_argument("--exp", type=str, required=True, help="Experiment to run (e.g., binary, multiclass, individual_pairs_both)")
    parser.add_argument(
        "--block_3_split",
        type=str,
        required=True,
        choices=["test", "val", "none"],
    )
    # Use BooleanOptionalAction to automatically support --outliers and --no-outliers
    
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed (required)"
    )

    parser.add_argument(
        "--standarization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to apply standardization (optional)"
    )

    parser.add_argument(
        "--normalize_eV",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to turn lambdas into eV (optional)"
    )

    parser.add_argument(
        "--f_as_log10",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to turn f into log10(f) (optional)"
    )

    args = parser.parse_args()

    model_name = args.model_name
    exp_to_run = args.exp
    block_3_split = args.block_3_split
    seed = args.seed
    standarization = args.standarization
    normalize_eV = args.normalize_eV
    f_as_log10 = args.f_as_log10
    
    


    if exp_to_run == "binary":
        run_binary(model_name, seed, block_3_split)
    elif exp_to_run == "lambda_regression":
        run_lambda_regression(model_name, seed,
                              standarization, normalize_eV, block_3_split)
    elif exp_to_run == 'f_regression':
        run_f_regression(model_name, seed,
                standarization, f_as_log10, block_3_split)
    else:
        print(f"Error: Unknown experiment '{exp_to_run}'.")
        sys.exit(1)

if __name__ == "__main__":
    main()