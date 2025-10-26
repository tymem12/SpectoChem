"""
Runs dvc reproduction in parallel stages, on multiple GPU cards
NOTE: GPU queueing is implemented naively and might cause errors!
"""

import json
import subprocess
from collections import deque
from pathlib import Path
from time import sleep
from typing import Optional

import typer
from libtmux.pane import Pane
from libtmux.server import Server
from libtmux.session import Session
from mpire import WorkerPool
from rich import print

server = Server()
print(server)

COMMAND = "SHELL=/bin/bash NUM_WORKERS={workers} CUDA_VISIBLE_DEVICES={cuda} dvc repro -s {stage}"


class Worker:
    def __init__(self, conda: str, data_loader_workers: int, dvc_args: str | None) -> None:
        self.conda = conda
        self.data_loader_workers = data_loader_workers
        self.dvc_args = dvc_args

    def __call__(self, worker_id: int, shared_objs: tuple[Session, deque], stage: str) -> None:
        session, devices = shared_objs
        cuda_device = self._acquire_device(devices)

        print(f"Starting {stage} on worker {worker_id} (CUDA={cuda_device})...")
        window = session.new_window(attach=True, window_name=f"w{worker_id}:{stage}")
        assert isinstance(window.attached_pane, Pane)
        window.attached_pane.send_keys(
            f'bash -c eval "$(conda shell.bash hook)" && conda activate {self.conda}'
        )
        window.attached_pane.send_keys("ulimit -n 10000")

        cmd = COMMAND.format(workers=self.data_loader_workers, cuda=cuda_device, stage=stage)

        if self.dvc_args:
            cmd = f"{cmd} {self.dvc_args}"

        window.attached_pane.send_keys(f"{cmd}; tmux wait -S ping-{worker_id}")
        subprocess.run(["tmux", "wait", f"ping-{worker_id}"]).check_returncode()
        print(f"Finished {stage}.")

        devices.append(cuda_device)

    def _acquire_device(self, queue: deque, max_trials: int = 3) -> int:
        device = None
        trial = 0
        while device is None:
            try:
                device = queue.pop()
            except IndexError:
                trial += 1
                if trial >= max_trials:
                    raise
                sleep(60)
        return device


def main(
    dvc: Path = typer.Option(..., dir_okay=False, file_okay=True),
    target: Optional[list[str]] = typer.Option(None, "--target", "-t"),
    cuda_devices: list[int] = typer.Option(..., "--cuda-device", "-c"),
    data_loader_workers: int = typer.Option(4),
    conda: str = typer.Option(...),
    session_name: str = typer.Option("dvc_runner"),
    dry: bool = typer.Option(False),
    dvc_args: str = typer.Option(None),
) -> None:
    if not dry:
        try:
            session, *_ = server.sessions.filter(session_name=session_name)
        except ValueError:
            session = server.new_session(session_name=session_name)

    result = subprocess.run(["dvc", "status", "--json", dvc], stdout=subprocess.PIPE)
    result.check_returncode()

    stages = sorted(json.loads(result.stdout.decode().replace("\x1b[0m", "")).keys())

    if target is not None:
        stages = [s for s in stages if s.split(":")[-1].split("@")[0] in target]

    devices = deque(cuda_devices)
    n_jobs = len(cuda_devices)

    jobs_per_device = n_jobs / len(set(devices))
    print(f"Total jobs to run: {len(stages)}")
    print(
        f"Running {n_jobs} parallel jobs on {len(set(devices))} CUDA devices: {set(devices)}; "
        f"{jobs_per_device} jobs/device"
    )
    print(stages)

    if not dry:
        worker = Worker(conda, data_loader_workers, dvc_args)
        with WorkerPool(
            n_jobs=n_jobs,
            start_method="threading",
            pass_worker_id=True,
            shared_objects=[session, devices],
        ) as pool:
            pool.map(worker, stages)


if __name__ == "__main__":
    typer.run(main)
