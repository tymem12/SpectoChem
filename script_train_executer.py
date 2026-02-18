import subprocess
import datetime

# 1. Dynamic Timestamp
now = datetime.datetime.now()
data_time = now.strftime("%Y-%m-%d_%H.%M.%S")

int_inter = 0
LAST_EXP = -1

# Dictionary mapping model names to their grid search parameters
# Each entry contains a list of dictionaries representing a specific config to test
model_configs = {
    'invariant_projection': [
        # Config 1: MLP Projection + SchNet Embedding (Standard Approach A)
        {
            'use_mlp_projection': True,
            'use_z_embedding': True,
            'uma_mlp_hidden_dim': 256
        },
        # Config 2: Weighted Sum + SchNet Embedding (Simpler)
        {
            'use_mlp_projection': False,
            'use_z_embedding': True,
        },
        # Config 3: Only UMA (No SchNet Embedding) - Pure Geometric
        {
            'use_mlp_projection': True,
            'use_z_embedding': False,
            'uma_mlp_hidden_dim': 256
        }
    ],
    'geometric_gating': [
        # Config 1: Pre-Interaction Gating (Standard)
        {
            'post_interaction_gating': False,
            'num_attention_heads': 4
        },
        # Config 2: Post-Interaction Gating (New logic)
        {
            'post_interaction_gating': True,
            'num_attention_heads': 4
        }
    ],
    'moe': [
        # Config 1: Soft Routing (Weighted Sum) with Full UMA Router
        {
            'routing_mode': 'soft',
            'num_experts': 3,
            'use_full_uma_for_router': True
        },
        # Config 2: Hard Routing (Gumbel) with Full UMA Router
        {
            'routing_mode': 'hard',
            'num_experts': 3,
            'use_full_uma_for_router': True
        },
        # Config 3: Soft Routing with only Scalar Router (Simpler Router)
        {
            'routing_mode': 'soft',
            'num_experts': 3,
            'use_full_uma_for_router': False
        }
    ]
}

# Iterate over models
for model_name, configs in model_configs.items():
    # Iterate over specific grid search configurations for that model
    for config in configs:
        
        # Create a string representation of the config for the folder path
        # e.g., "mlp_proj-True_z_emb-True"
        exp_param_parts = []
        for k, v in config.items():
            # Shorten keys for readability in folder paths if needed
            short_key = k.replace("use_", "").replace("projection", "proj").replace("embedding", "emb").replace("interaction", "inter")
            exp_param_parts.append(f"{short_key}-{v}")
        exp_param_str = "_".join(exp_param_parts)

        # Standard loops from your original script
        for min_f_value in [0.01, 0.05]:
            for metric, metric_mode in [('loss', 'min'), ('F1', 'max')]:

                # Build the command arguments list for the backbone params
                backbone_args = [f"model.backbone.{k}={v}" for k, v in config.items()]

                # EXPERIMENTS FOR BINARY CLASSIFICATION
                cmd = [
                        "python", "experiments/scripts/train_graph_level.py",
                        "+exp=TMQM_SPECTO_BINARY",
                        "model=supervised_graph_level",
                        f"backbone@model.backbone={model_name}",
                        
                        # Add specific backbone params from grid search
                        *backbone_args,

                        f"training.experiment_name=supervised/{data_time}/binary_classification/{model_name}/{exp_param_str}/UMA_full_embedding/min_f_value_{min_f_value}/metric_{metric}/330-650/results",
                        f"dataset.additional_loading_params.min_f_value={min_f_value}",
                        f"dataset.main_metric={metric}",
                        f"dataset.metric_mode={metric_mode}"
                ]

                print("\n", int_inter, f"Running: {model_name} | {exp_param_str}")
                
                if int_inter > LAST_EXP:
                    result = subprocess.run(cmd, check=False)
                    if result.returncode != 0:
                        print(f"Error: Command failed with code {result.returncode}")
                
                int_inter += 1