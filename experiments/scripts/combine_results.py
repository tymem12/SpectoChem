from pathlib import Path

import pandas as pd
import typer


def main(
    report_files: list[Path] = typer.Option(..., "-rf"),
    target_file: Path = typer.Option(...),
) -> None:
    reports = []
    for r_file in report_files:
        df = (
            pd.read_table(r_file, sep="|", header=0, index_col=None, skipinitialspace=True)
            .dropna(axis=1, how="all")
            .iloc[1:]
        )
        df["method"] = r_file.name.replace("_report.md", "")
        cols = [col.strip() for col in df.columns]
        df.columns = cols
        reports.append(df)

    all_reports = pd.concat(reports, axis=0)

    cols.insert(0, cols.pop(cols.index("experiment_name")))
    cols.insert(1, cols.pop(cols.index("method")))
    all_reports = all_reports[cols]

    for exp_name in all_reports["dataset"].unique():
        df = all_reports[all_reports["dataset"] == exp_name]
        with target_file.open("a") as file:
            df.to_markdown(file, index=False)
            file.write("\n\n")


if __name__ == "__main__":
    typer.run(main)
