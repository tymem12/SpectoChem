from pathlib import Path

import typer
import yaml
from dvc.repo import Repo
from rich import print


def main(
    dvc_dir: Path = typer.Option("."),
    dvc_lock: Path = typer.Option("dvc.lock"),
) -> None:
    repo = Repo(dvc_dir)

    stages = set()
    for stage, *_ in repo.index.graph.nodes(data=True):
        if stage.path_in_repo == "dvc.yaml":
            stages.add(stage.name)
        else:
            continue

    with dvc_lock.open() as file:
        lock_file = yaml.safe_load(file)

    lock_stages = set(lock_file["stages"].keys())
    to_remove = lock_stages.difference(stages)

    print(to_remove)
    if typer.confirm("Are you sure you want to delete?"):
        lock_file["stages"] = {
            key: val for key, val in lock_file["stages"].items() if key not in to_remove
        }

        with dvc_lock.open("w") as file:
            yaml.safe_dump(lock_file, file)


if __name__ == "__main__":
    typer.run(main)
