import subprocess


int_inter = 0
LAST_EXP = -1


data_time = '18.12'

for model_name in ['dummy_model']:
    for min_f_value in [0.01, 0.05]:
        for metric, metric_mode in [('loss', 'min'),('F1', 'max')]:

            # EXPERIMENTS FOR BIANRY CLASSIFICATION
            cmd = [
                    "python", "experiments/scripts/train_graph_level.py",
                    "+exp=TMQM_SPECTO_BINARY",
                    "model=supervised_graph_level",
                    f"backbone@model.backbone={model_name}",
                    "model.backbone.disable_pos=True",
                    f"training.experiment_name=supervised/{data_time}/binary_classification/{model_name}/UMA/min_f_value_{min_f_value}/metric_{metric}/330-650/results",
                    f"dataset.additional_loading_params.min_f_value={min_f_value}",
                    f"dataset.main_metric={metric}",
                    f"dataset.metric_mode={metric_mode}"

            ]
            print("\n", int_inter)
            if int_inter > LAST_EXP:
                result = subprocess.run(cmd, check=False)
                if result.returncode !=  0:
                    print(f"Błąd: komenda zakończyła się kodem {result.returncode}")
            int_inter += 1

    
    for min_f_value in [0.01, 0.05]:
        for bucket_size in [1,5,10]:
            for metric, metric_mode in [('loss', 'min'), ('F1', 'max')]:
            # EXPERIMENTS FOR MULTICLASS CLASSIFICATION
            
                cmd = [
                        "python", "experiments/scripts/train_graph_level.py",
                        "+exp=TMQM_SPECTO_BINARY_VECTOR_MULTICLASS",
                        "model=supervised_graph_level",
                        f"backbone@model.backbone={model_name}",
                        "model.backbone.disable_pos=True",
                        f"training.experiment_name=supervised/{data_time}/binary_vector_multiclass/{model_name}/UMA/min_f_value_{min_f_value}/metric_{metric}/lambda_bucket_size_{bucket_size}/330-650/results",
                        f"dataset.additional_loading_params.min_f_value={min_f_value}",
                        f"dataset.additional_loading_params.filter_f_value={min_f_value}",
                        f"dataset.additional_loading_params.lambda_bucket_size={bucket_size}",
                        f"dataset.main_metric={metric}",
                        f"dataset.out_channels={320 // bucket_size}",
                        f"dataset.metric_mode={metric_mode}"


                ]
                print("\n", int_inter)
                if int_inter > LAST_EXP:
                    result = subprocess.run(cmd, check=False)
                    if result.returncode != 0:
                        print(f"Błąd: komenda zakończyła się kodem {result.returncode}")
                int_inter += 1
                # EXPERIMENTS FOR MULTILABEL CLASSIFICATION

                cmd = [
                        "python", "experiments/scripts/train_graph_level.py",
                        "+exp=TMQM_SPECTO_BINARY_VECTOR_MULTILABEL",
                        "model=supervised_graph_level",
                        f"backbone@model.backbone={model_name}",
                        "model.backbone.disable_pos=True",
                        f"training.experiment_name=supervised/{data_time}/binary_vector_multilabel/{model_name}/UMA/min_f_value_{min_f_value}/metric_{metric}/lambda_bucket_size_{bucket_size}/330-650/results",
                        f"dataset.additional_loading_params.min_f_value={min_f_value}",
                        f"dataset.additional_loading_params.filter_f_value={min_f_value}",
                        f"dataset.additional_loading_params.lambda_bucket_size={bucket_size}",
                        f"dataset.main_metric={metric}",
                        f"dataset.out_channels={320 // bucket_size}",
                        f"dataset.metric_mode={metric_mode}"
                        ]
                print("\n", int_inter)
                if int_inter > LAST_EXP:
                    result = subprocess.run(cmd, check=False)
                    if result.returncode !=  0:
                        print(f"Błąd: komenda zakończyła się kodem {result.returncode}")
                int_inter += 1
    # EXPERIMENTS FOR PAIRS on LAMBDA_1 
    cmd = [
        "python", "experiments/scripts/train_graph_level.py",
        "+exp=TMQM_SPECTO_PAIRS",
        "model=supervised_graph_level",
        f"backbone@model.backbone={model_name}",
        "model.backbone.disable_pos=True",
        f"training.experiment_name=supervised/{data_time}/pairs/{model_name}/UMA/lambda_1/330-650/results",
        "training.random_seed=2137"
        ]

    print("\n", int_inter)
    if int_inter > LAST_EXP:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"Błąd: komenda zakończyła się kodem {result.returncode}")
    int_inter += 1
    
    
    # EXPERIMENTS FOR ONLY_LAMBDA on LAMBDA_1 
    cmd = [
        "python", "experiments/scripts/train_graph_level.py",
        "+exp=TMQM_SPECTO_ONLY_LAMBDAS",
        "model=supervised_graph_level",
        f"backbone@model.backbone={model_name}",
        "model.backbone.disable_pos=True",
        f"training.experiment_name=supervised/{data_time}/only_lambdas/{model_name}/UMA/lambda_1/330-650/results",
        "training.random_seed=2137"
    ]
    print("\n", int_inter)
    if int_inter > LAST_EXP:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"Błąd: komenda zakończyła się kodem {result.returncode}")
    int_inter += 1


for min_f_value in [0.01, 0.05]:
    # EXPERIMENTS SIMMILAR TO MULTICLASS BUT INSTEAD OF VECOTR WITH ONLY ONE "POSITIVE" CLASS WE DO REGRESSION TASK (PAIRS) 
    cmd = [
        "python", "experiments/scripts/train_graph_level.py",
        "+exp=TMQM_SPECTO_PAIRS",
        "model=supervised_graph_level",
        f"backbone@model.backbone=dummy_model",
        "model.backbone.disable_pos=True",
        f"training.experiment_name=supervised/{data_time}/pairs/dummy_model_UMA/like_multiclass_{min_f_value}/one_visible_lambda/330-650/results",
        "training.random_seed=2137",
        f"dataset.additional_loading_params.min_f_value={min_f_value}",
        f"dataset.additional_loading_params.filter_f_value={min_f_value}",
        f"dataset.additional_loading_params.filter_type=one_visible_lambda",

    ]

    print("\n", int_inter)
    if int_inter > LAST_EXP:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"Błąd: komenda zakończyła się kodem {result.returncode}")
    int_inter += 1
    
    # EXPERIMENTS SIMMILAR TO MULTICLASS BUT INSTEAD OF VECOTR WITH ONLY ONE "POSITIVE" CLASS WE DO REGRESSION TASK (LAMBDAS_ONLY) 

    cmd = [
        "python", "experiments/scripts/train_graph_level.py",
        "+exp=TMQM_SPECTO_ONLY_LAMBDAS",
        "model=supervised_graph_level",
        f"backbone@model.backbone=schnet",
        "model.backbone.disable_pos=True",
        f"training.experiment_name=supervised/{data_time}/only_lambdas/dummy_model_UMA/like_multiclass_{min_f_value}/one_visible_lambda/330-650/results",
        "training.random_seed=2137",
        f"dataset.additional_loading_params.min_f_value={min_f_value}",
        f"dataset.additional_loading_params.filter_f_value={min_f_value}",
        f"dataset.additional_loading_params.filter_type=one_visible_lambda",

    ]

    print("\n", int_inter)
    if int_inter > LAST_EXP:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"Błąd: komenda zakończyła się kodem {result.returncode}")
    int_inter += 1
# print("\n=== Running with standarize_lambda = FALSE ===\n")
# subprocess.run(cmd_false, check=True)
