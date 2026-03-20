import subprocess
import datetime
import argparse
import sys

# --- CONFIGURATION ---
LAST_EXP = -1

# 1. Dynamic Timestamp
now = datetime.datetime.now()
data_time = now.strftime("%Y-%m-%d_%H.%M.%S")
int_inter = 0  # Global counter

# --- CONFIG GENERATION ---
invariant_projection_base = [
    {'use_mlp_projection': True, 'use_z_embedding': True},
    {'use_mlp_projection': False, 'use_z_embedding': True},
    {'use_mlp_projection': True, 'use_z_embedding': False},
    {'use_mlp_projection': False, 'use_z_embedding': False}
]

invariant_projection_configs = []
for cfg in invariant_projection_base:
    if False and cfg.get('use_mlp_projection'):
        # Duplicate for each hidden dim
        for hidden_dim in [128, 512, 1024]:
            new_cfg = cfg.copy()
            new_cfg['uma_mlp_hidden_dim'] = hidden_dim
            invariant_projection_configs.append(new_cfg)
    else:
        invariant_projection_configs.append(cfg)

model_configs = {
    'invariant_projection': invariant_projection_configs,
    'geometric_gating': [
        {'post_interaction_gating': False},
        {'post_interaction_gating': True}
    ],
    'moe': [
        {'routing_mode': 'soft', 'use_full_uma_for_router': True},
        {'routing_mode': 'hard', 'use_full_uma_for_router': True},
        {'routing_mode': 'soft', 'use_full_uma_for_router': False},
        {'routing_mode': 'hard', 'use_full_uma_for_router': False},
    ]
}

# --- UTILITY FUNCTIONS ---
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


# --- EXPERIMENT RUNNERS ---
def run_binary(model_name, exp_param_str, backbone_args):
    for min_f_value in [0.01, 0.05]:
        for metric, metric_mode in [('loss', 'min'), ('F1', 'max')]:
            experiment_path = f"supervised/{data_time}/binary_classification/{model_name}/{exp_param_str}/UMA_full_embedding/min_f_value_{min_f_value}/metric_{metric}/330-650/results"
            
            cmd = [
                "python", "experiments/scripts/train_graph_level.py",
                "+exp=TMQM_SPECTO_BINARY",
                "model=supervised_graph_level",
                f"backbone@model.backbone={model_name}",
                *backbone_args,
                f"training.experiment_name={experiment_path}",
                f"dataset.additional_loading_params.min_f_value={min_f_value}",
                f"dataset.main_metric={metric}",
                f"dataset.metric_mode={metric_mode}"
            ]
            run_cmd(cmd, f"{model_name} | {exp_param_str} | BINARY | f={min_f_value} | {metric}")

def run_vector(model_name, exp_param_str, backbone_args, is_multilabel=False):
    exp_flag = "+exp=TMQM_SPECTO_BINARY_VECTOR_MULTILABEL" if is_multilabel else "+exp=TMQM_SPECTO_BINARY_VECTOR_MULTICLASS"
    folder_prefix = "binary_vector_multilabel" if is_multilabel else "binary_vector_multiclass"
    exp_type_log = "MULTILABEL" if is_multilabel else "MULTICLASS"

    for min_f_value in [0.01, 0.05]:
        for bucket_size in [1, 5, 10]:
            for metric, metric_mode in [('loss', 'min'), ('F1', 'max')]:
                experiment_path = f"supervised/{data_time}/{folder_prefix}/{model_name}/{exp_param_str}/UMA_full_embedding/min_f_value_{min_f_value}/lambda_bucket_size_{bucket_size}/metric_{metric}/330-650/results"

                cmd = [
                    "python", "experiments/scripts/train_graph_level.py",
                    exp_flag,
                    "model=supervised_graph_level",
                    f"backbone@model.backbone={model_name}",
                    *backbone_args,
                    f"training.experiment_name={experiment_path}",
                    f"dataset.additional_loading_params.min_f_value={min_f_value}",
                    f"dataset.additional_loading_params.filter_f_value={min_f_value}",
                    f"dataset.additional_loading_params.lambda_bucket_size={bucket_size}",
                    f"dataset.main_metric={metric}",
                    f"dataset.out_channels={320 // bucket_size}",
                    f"dataset.metric_mode={metric_mode}"
                ]
                run_cmd(cmd, f"{model_name} | {exp_param_str} | {exp_type_log} | f={min_f_value} | bucket={bucket_size} | {metric}")

def run_lambda_1(model_name, exp_param_str, backbone_args, is_pairs=True, min_states=10):
    exp_flag = "+exp=TMQM_SPECTO_PAIRS" if is_pairs else "+exp=TMQM_SPECTO_ONLY_LAMBDAS"
    folder_prefix = "pairs" if is_pairs else "only_lambdas"
    exp_type_log = f"PAIRS (lambda_1, min_states={min_states})" if is_pairs else "ONLY_LAMBDAS (lambda_1)"

    states_path_part = f"/min_states_{min_states}" if is_pairs else ""
    experiment_path = f"supervised/{data_time}/{folder_prefix}/{model_name}/{exp_param_str}/UMA_full_embedding/lambda_1{states_path_part}/330-650/results"
    
    cmd = [
        "python", "experiments/scripts/train_graph_level.py",
        exp_flag,
        "model=supervised_graph_level",
        f"backbone@model.backbone={model_name}",
        *backbone_args,
        f"training.experiment_name={experiment_path}",
        "training.random_seed=2137"
    ]
    
    if is_pairs:
        cmd.append(f"dataset.additional_loading_params.min_states={min_states}")
        
    run_cmd(cmd, f"{model_name} | {exp_param_str} | {exp_type_log}")


def run_one_visible(model_name, exp_param_str, backbone_args, is_pairs=True, min_states=10):
    exp_flag = "+exp=TMQM_SPECTO_PAIRS" if is_pairs else "+exp=TMQM_SPECTO_ONLY_LAMBDAS"
    folder_prefix = "pairs" if is_pairs else "only_lambdas"
    exp_type_log = f"PAIRS (one_visible, min_states={min_states})" if is_pairs else "ONLY_LAMBDAS (one_visible)"

    for min_f_value in [0.01, 0.05]:
        states_suffix = f"_states_{min_states}" if is_pairs else ""
        experiment_path = f"supervised/{data_time}/{folder_prefix}/{model_name}/{exp_param_str}/UMA_full_embedding/like_multiclass_{min_f_value}{states_suffix}/one_visible_lambda/330-650/results"

        cmd = [
            "python", "experiments/scripts/train_graph_level.py",
            exp_flag,
            "model=supervised_graph_level",
            f"backbone@model.backbone={model_name}",
            *backbone_args,
            f"training.experiment_name={experiment_path}",
            "training.random_seed=2137",
            f"dataset.additional_loading_params.min_f_value={min_f_value}",
            f"dataset.additional_loading_params.filter_f_value={min_f_value}",
            f"dataset.additional_loading_params.filter_type=one_visible_lambda"
        ]
        
        if is_pairs:
            cmd.append(f"dataset.additional_loading_params.min_states={min_states}")
            
        run_cmd(cmd, f"{model_name} | {exp_param_str} | {exp_type_log} | f={min_f_value}")

def run_individual_variable_pairs(model_name, exp_param_str, backbone_args, outliers: bool, num_pairs=10, pair_value_to_predict="lambda"):
    exp_type_log = f"ONLY_LAMBDAS (individual_pairs, predict={pair_value_to_predict})"

    min_f_value = -1

    for num_pair in range(1, num_pairs + 1):
        experiment_path = f"supervised/{data_time}/individual_pairs_{pair_value_to_predict}/{model_name}/{exp_param_str}/UMA_full_embedding/outliers-{outliers}/num_pair_{num_pair}/330-650/results"

        cmd = [
            "python", "experiments/scripts/train_graph_level.py",
            "+exp=TMQM_SPECTO_ONLY_LAMBDAS",
            "model=supervised_graph_level",
            f"backbone@model.backbone={model_name}",
            *backbone_args,
            f"training.experiment_name={experiment_path}",
            "training.random_seed=2137",
            f"dataset.additional_loading_params.num_states={num_pair}",
            f"dataset.additional_loading_params.min_f_value={min_f_value}",
            f"dataset.additional_loading_params.filter_f_value={min_f_value}",
            f"dataset.additional_loading_params.outliers={outliers}",
            f"dataset.additional_loading_params.pair_value_to_predict={pair_value_to_predict}",
            "dataset.additional_loading_params.filter_type=all_samples"
        ]
        
        run_cmd(cmd, f"{model_name} | {exp_param_str} | {exp_type_log} | num_pair={num_pair}")

def run_all_individual_variable_pairs(model_name, exp_param_str, backbone_args, **kwargs):
    """Convenience function to run both 'lambda' and 'f' predictions back-to-back."""
    run_individual_variable_pairs(model_name, exp_param_str, backbone_args, **kwargs, pair_value_to_predict="lambda")
    run_individual_variable_pairs(model_name, exp_param_str, backbone_args, **kwargs, pair_value_to_predict="f")

# --- MAIN EXECUTION LOOP ---
def main():
    parser = argparse.ArgumentParser(description="Run specific experiments with specific models.")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model (e.g., invariant_projection, geometric_gating, moe)")
    parser.add_argument("--exp", type=str, required=True, help="Experiment to run (e.g., binary, multiclass, individual_pairs_both)")
    
    # Use BooleanOptionalAction to automatically support --outliers and --no-outliers
    parser.add_argument("--outliers", action=argparse.BooleanOptionalAction, help="Whether to use outliers. Required for individual_pairs experiments.")
    
    args = parser.parse_args()
    
    model_name = args.model_name
    exp_to_run = args.exp
    
    # Validate the outliers argument if the experiment requires it
    if exp_to_run.startswith("individual_pairs"):
        if args.outliers is None:
            print(f"Error: You must specify either --outliers or --no-outliers for experiment '{exp_to_run}'.")
            sys.exit(1)
            
    # Now it's already a boolean (True if --outliers, False if --no-outliers)
    outliers_bool = args.outliers
    
    if model_name not in model_configs:
        print(f"Error: Model '{model_name}' not found in model_configs. Available models: {list(model_configs.keys())}")
        sys.exit(1)

    # Fetch only the configs for the requested model
    configs = model_configs[model_name]

    for config in configs:
        exp_param_str = get_config_string(config)
        backbone_args = get_backbone_args(config)

        if exp_to_run == "binary":
            run_binary(model_name, exp_param_str, backbone_args)
        elif exp_to_run == "multiclass":
            run_vector(model_name, exp_param_str, backbone_args, is_multilabel=False)
        elif exp_to_run == "multilabel":
            run_vector(model_name, exp_param_str, backbone_args, is_multilabel=True)
        elif exp_to_run == "pairs_lambda_1":
            run_lambda_1(model_name, exp_param_str, backbone_args, is_pairs=True)
        elif exp_to_run == "only_lambdas_lambda_1":
            run_lambda_1(model_name, exp_param_str, backbone_args, is_pairs=False)
        elif exp_to_run == "individual_pairs_f":
            run_individual_variable_pairs(model_name, exp_param_str, backbone_args, outliers=outliers_bool, num_pairs=10, pair_value_to_predict="f")
        elif exp_to_run == "individual_pairs_lambda":
            run_individual_variable_pairs(model_name, exp_param_str, backbone_args, outliers=outliers_bool, num_pairs=10, pair_value_to_predict="lambda")
        elif exp_to_run == "individual_pairs_both":
            run_all_individual_variable_pairs(model_name, exp_param_str, backbone_args, outliers=outliers_bool, num_pairs=10)
        elif exp_to_run == "pairs_one_visible":
            run_one_visible(model_name, exp_param_str, backbone_args, is_pairs=True)
        elif exp_to_run == "only_lambdas_one_visible":
            run_one_visible(model_name, exp_param_str, backbone_args, is_pairs=False)
        else:
            print(f"Error: Unknown experiment '{exp_to_run}'.")
            sys.exit(1)

if __name__ == "__main__":
    main()