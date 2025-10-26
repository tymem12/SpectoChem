import math
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import typer

from gjepa.utils.config import load_and_resolve_config
from gjepa.utils.misc import load_json


def main(
    root_dir: Path = typer.Option(...),
    report_metrics: Optional[list[str]] = typer.Option(None),
    main_metric_only: bool = typer.Option(False),
    ignore_errors: bool = typer.Option(False),
) -> None:
    if not main_metric_only ^ ignore_errors:
        raise ValueError("Cannot use main metric with ignore errors, need to load hparams.yaml")

    results: list[dict[str, Any]] = []

    metric_columns = set()
    for metric_file in root_dir.rglob("metrics.json"):
        # load metrics
        metric_data = load_json(metric_file)

        # load info about main metric if flag is set
        if main_metric_only:
            hparams_file = metric_file.parent / "hparams.yaml"
            hparams = load_and_resolve_config(hparams_file)["config"]
            main_metric = f"test_{hparams['dataset']['main_metric']}"
            metric_columns.add(main_metric)
            metric_data = {main_metric: metric_data[main_metric]}
        else:
            metric_columns |= set(m_name for m_name in metric_data.keys())

        # parse experiment name
        *_, experiment_group, experiment_name, _, exp_version, _ = metric_file.parts

        exp_dir = metric_file.parent
        best_epoch = _retrieve_best_epoch(exp_dir / "checkpoints")

        res = {
            "experiment_group": experiment_group,
            "experiment_name": experiment_name,
            "version": exp_version,
            "best_epoch": best_epoch,
        }

        try:
            hparams = load_hparams(exp_dir)
        except FileNotFoundError:
            print(f"hparams.yaml not found for experiments {exp_dir}, skipping...")
            res["dataset"] = experiment_name
        except KeyError:
            if not ignore_errors:
                raise
            res["dataset"] = experiment_name
        else:
            res |= hparams

        res |= metric_data
        results.append(res)

    df = pd.DataFrame(results)
    df = df.dropna(axis=1, how="all").sort_values(by="experiment_name")
    df[list(metric_columns)] = df[list(metric_columns)] * 100

    # Store csv report
    csv_file = root_dir.parent / f"{root_dir.stem}_report.csv"
    df.to_csv(csv_file, index=False, float_format="%.2f")

    # Prepare human-readable markdown report
    group_cols = ["experiment_name", "dataset"]

    if not report_metrics:
        report_metrics = list(metric_columns)

    # hyperparameters are not included into markdown report
    df_agg = df.groupby(group_cols)[report_metrics].agg("mean")
    df_std = df.groupby(group_cols)[report_metrics].agg("std")
    df_count = df.groupby(group_cols).size()
    df_count.name = "#repeats"

    for col in set(df_agg.columns):
        df_agg[col] = df_agg[col].apply(_fmt_float) + " (" + df_std[col].apply(_fmt_float) + ")"
    df_agg = df_agg.join(df_count)

    # Store markdown report
    md_file = root_dir.parent / f"{root_dir.stem}_report.md"
    df_agg.reset_index().to_markdown(md_file, index=False)


def load_hparams(exp_dir: Path) -> dict[str, Any]:
    hparams = load_and_resolve_config(exp_dir / "hparams.yaml")["config"]

    similarity_method = hparams["model"].get("similarity_matrix_file")
    if similarity_method:
        similarity_method = Path(similarity_method).stem

    return {
        "model_type": hparams["model"]["name"],
        "dataset": hparams["dataset"]["name"],
        "num_hops": hparams["dataset"]["num_hops"],
        "backbone": hparams["model"]["backbone"]["gnn_cls"].split(".")[-1],
        "hidden_dim": hparams["model"]["backbone"].get("hidden_channels"),
        "pos_encoding": hparams["pos_encoding"],
        "similarity_method": similarity_method,
        "context_target_overlap_strategy": hparams["model"].get("context_target_overlap_strategy"),
        "num_targets": hparams["model"].get("num_targets"),
        "batch_size": hparams["training"]["batch_size"],
    }


def _retrieve_best_epoch(checkpoints_dir: Path) -> int | None:
    if not checkpoints_dir.is_dir():
        return None

    ckpt_files = [ckpt_f for ckpt_f in checkpoints_dir.iterdir() if ckpt_f.stem.startswith("epoch")]
    assert len(ckpt_files) == 1
    return int(ckpt_files[0].stem.split("-")[0].split("=")[1])


def _fmt_float(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.2f}"


if __name__ == "__main__":
    typer.run(main)
