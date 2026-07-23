"""
Launcher for the data-quantity vs data-diversity ablation.

Answers: does adding non-3d data help predicting 3d, and is it the diversity or
just the extra quantity? Every regime is evaluated on the SAME frozen block-3
test set, so only the training data changes.

Regimes
-------
  dump      : run the 3d-only baseline once (1 epoch) just to freeze the block-3
              test set -> writes the holdout isomer-keys file.
  baseline  : train on 3d, test on the frozen 3d holdout                (3d -> 3d)
  full345   : train on 3d(train)+4d+5d, test on the frozen 3d holdout   (345 -> 3d)
  mini345   : same as full345 but training set capped to N graphs       (345mini -> 3d)

IMPORTANT
---------
* Use the SAME --seed for `dump` as the original 3d baseline you already ran, so
  the frozen test set matches it.
* --max-train-graphs N for mini345 must equal the 3d baseline train size (repo
  prints `len of train is X` during setup).
* Run mini345 with several seeds and average (subsampling variance).
* Architecture is identical across regimes (do NOT change the backbone).

Examples
--------
  # 0) freeze the holdout (once, same seed as your 3d baseline):
  python experiments/scripts/run_data_ablation.py --task lambda --model-name schnet \
      --seed 1234 --regime dump

  # 1) baseline / full / mini (test == frozen holdout in all three):
  python experiments/scripts/run_data_ablation.py --task lambda --model-name schnet \
      --seed 1234 --regime baseline
  python experiments/scripts/run_data_ablation.py --task lambda --model-name schnet \
      --seed 1234 --regime full345
  python experiments/scripts/run_data_ablation.py --task lambda --model-name schnet \
      --seed 1234 --regime mini345 --max-train-graphs 18000
"""
import argparse
import subprocess

HOLDOUT_DEFAULT = "data/datasets/TMQM_SPECTO/raw/block3_holdout_keys.json"


def head_params():
    return [256], "silu", 0.2


def base_overrides(task, model_name, seed, num_pair, exp_path):
    exp = "TMQM_SPECTO_LAMBDA_REGRESSOR" if task == "lambda" else "TMQM_SPECTO_F_REGRESSOR"
    hidden, act, drop = head_params()
    normalize_eV = task == "lambda"
    standarize = task == "lambda"
    outlier_strategy = "remove_outlying_transition" if task == "lambda" else "null"
    ov = [
        "python", "experiments/scripts/train_graph_level.py",
        f"+exp={exp}",
        "model=supervised_graph_level",
        f"backbone@model.backbone={model_name}",
        f"training.experiment_name={exp_path}",
        f"training.random_seed={seed}",
        f"dataset.additional_loading_params.num_states={num_pair}",
        "dataset.additional_loading_params.min_f_value=-1",
        "dataset.additional_loading_params.filter_f_value=-1",
        "dataset.additional_loading_params.sort_by_max_f=false",
        f"dataset.additional_loading_params.outlier_strategy={outlier_strategy}",
        "dataset.additional_loading_params.lambda_outlier_threshold=1500",
        "dataset.additional_loading_params.f_outlier_threshold=0.5",
        f"dataset.additional_loading_params.convert_to_ev={normalize_eV}",
        "dataset.additional_loading_params.filter_type=all_samples",
        f"dataset.predictor_kwargs.hidden_channels={hidden}",
        f"dataset.predictor_kwargs.activation={act}",
        f"dataset.predictor_kwargs.dropout={drop}",
    ]
    if task == "lambda":
        ov.append(f"dataset.additional_loading_params.standarize_lambda={standarize}")
    else:
        ov.append("dataset.additional_loading_params.standarize_f=false")
    return ov


def build(task, model_name, seed, regime, holdout, max_train, num_pair):
    tag = {"baseline": "block3_holdout_baseline",
           "full345": "block3_holdout_full345",
           "mini345": "block3_holdout_mini345",
           "dump": "block3_dump"}[regime]
    exp_path = f"supervised/{seed}/{task}_regressor/{num_pair}/{model_name}/{tag}/results"
    ov = base_overrides(task, model_name, seed, num_pair, exp_path)

    if regime == "dump":
        # 3d-only baseline; just materialize the frozen test set, train 1 epoch.
        ov += [
            "dataset.block_3_split_mode=null",
            "dataset.additional_loading_params.block_3_only=True",
            f"dataset.dump_test_keys_file={holdout}",
            "training.max_epochs=1",
        ]
    elif regime == "baseline":
        # 3d -> 3d, but tested on the SAME frozen holdout T (seed-independent test).
        ov += [
            "dataset.block_3_split_mode=train_holdout",
            f"dataset.holdout_test_keys_file={holdout}",
            "dataset.holdout_train_block_3_only=True",
        ]
    elif regime in ("full345", "mini345"):
        ov += [
            "dataset.block_3_split_mode=train_holdout",
            f"dataset.holdout_test_keys_file={holdout}",
        ]
        if regime == "mini345":
            if max_train is None:
                raise SystemExit("mini345 requires --max-train-graphs N (= 3d baseline train size)")
            ov.append(f"dataset.max_train_graphs={max_train}")
    return ov


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["lambda", "f"], required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--regime", choices=["dump", "baseline", "full345", "mini345"], required=True)
    p.add_argument("--holdout", default=HOLDOUT_DEFAULT)
    p.add_argument("--max-train-graphs", type=int, default=None)
    p.add_argument("--num-pairs", type=int, default=10, help="how many excited states (0..N-1)")
    args = p.parse_args()

    pairs = [0] if args.regime == "dump" else range(args.num_pairs)
    for num_pair in pairs:
        cmd = build(args.task, args.model_name, args.seed, args.regime,
                    args.holdout, args.max_train_graphs, num_pair)
        print(f"\n[{args.regime}] task={args.task} seed={args.seed} num_pair={num_pair}")
        print(" ".join(cmd))
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            print(f"  -> FAILED (code {res.returncode})")
            if args.regime == "dump":
                break


if __name__ == "__main__":
    main()
