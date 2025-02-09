# [TODO: do backend setting more cleanly]
import os

os.environ["MUJOCO_GL"] = "egl"

from .training import train
from .evaluate import evaluate_wrapper
from ogbench import make_env_and_datasets
from ogbench.impls.utils.datasets import HGCDataset, GCDataset, Dataset
from ogbench.impls.agents import CRLAgent, CMDAgent, GCBCAgent, QRLAgent, HIQLAgent
from .configurations import generate_configurations, get_config_space
import argparse
from datetime import datetime
from pathlib import Path
import json
import git
import toml
import logging
from ml_collections import FrozenConfigDict
import submitit
import signal
import sys
from itertools import product

logger = logging.getLogger(__name__)

AGENT_CLASSES = {
    "CRL": CRLAgent,
    "CMD": CMDAgent,
    "GCBC": GCBCAgent,
    "QRL": QRLAgent,
    "HIQL": HIQLAgent,
}
DATASET_CLASSES = {
    "CRL": GCDataset,
    "CMD": GCDataset,
    "GCBC": GCDataset,
    "QRL": GCDataset,
    "HIQL": HGCDataset,
}


def run_setup(args: argparse.Namespace) -> None:
    args.logdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.logdir / "log.txt", level=logging.INFO)
    logging.info("Set up for later running")

    # Log metadata to file
    metadata = {
        "arguments": args.__dict__,
        "git": {
            "commit": git.Repo(".", search_parent_directories=True).head.object.hexsha,
            "branch": git.Repo(".", search_parent_directories=True).active_branch.name,
        },
        "time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(args.logdir / "info.toml", "w") as f:
        f.write(toml.dumps(metadata))

    # Save configurations
    config_dir = args.logdir / "configurations"
    config_dir.mkdir()

    config_space = get_config_space(args.agent)
    config_space.to_json(config_dir / "configspace.json")

    configurations = generate_configurations(
        args.n_configurations, args.agent, set(args.hyperparameters)
    )
    for i, config in enumerate(configurations):
        with open(config_dir / f"configuration_{i}.json", "w") as f:
            f.write(config.to_json())

    return


def run_config(
    logdir: Path,
    phase: int,
    agent_path: Path | None,
    configuration_index: int,
    seed: int,
) -> None:
    assert phase == 0 or agent_path

    # submitit just bypasses SIGTERM although it should end the job, overwrite that behaviour here
    def handler(signum, frame):
        print(f"Received {signal.Signals(signum).name} ({signum}), stopping!")
        sys.exit(1)

    signal.signal(signal.SIGTERM, handler)

    setup = toml.load(args.logdir / "info.toml")["arguments"]

    run_log_dir = (
        logdir
        / "run_logs"
        / f"configuration_{configuration_index}"
        / f"phase_{phase}"
        / f"seed_{seed}"
    )
    run_log_dir.mkdir(parents=True, exist_ok=False)

    # Log metadata to file
    metadata = {
        "arguments": {
            "logdir": logdir,
            "phase": phase,
            "agent_path": agent_path,
            "configuration_index": configuration_index,
            "seed": seed,
        },
        "git": {
            "commit": git.Repo(".", search_parent_directories=True).head.object.hexsha,
            "branch": git.Repo(".", search_parent_directories=True).active_branch.name,
        },
        "time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(run_log_dir / "info.toml", "w") as f:
        f.write(toml.dumps(metadata))

    env, train_dataset, val_dataset = make_env_and_datasets(setup["dataset"])  # type: ignore

    with open(
        logdir / "configurations" / f"configuration_{configuration_index}.json", "r"
    ) as f:
        configuration = FrozenConfigDict(json.loads(f.read()))

    eval_trajectory = train(
        agent_class=AGENT_CLASSES[setup["agent"]],
        agent_path=agent_path,
        env=env,
        train_dataset=DATASET_CLASSES[setup["agent"]](
            Dataset.create(**train_dataset), configuration
        ),
        val_dataset=DATASET_CLASSES[setup["agent"]](
            Dataset.create(**val_dataset), configuration
        ),
        already_trained_steps=phase,
        eval_at_steps=setup["eval_steps"],
        evaluate=evaluate_wrapper,
        config=configuration,
        save_at_steps=setup["save_steps"]
        if "save_steps" in setup
        else setup["eval_steps"],
        eval_episodes=setup["eval_episodes"],
        log_dir=run_log_dir,
        seed=seed,
    )

    with open(run_log_dir / "eval_trajectory.json", "w") as f:
        f.write(eval_trajectory.to_json())


def submit(args: argparse.Namespace) -> None:
    setup = toml.load(args.logdir / "info.toml")["arguments"]

    configuration_indices = list(range(setup["n_configurations"]))
    seeds = list(range(args.n_seeds))
    arguments = product(
        [args.logdir], [args.phase], [args.agent_path], configuration_indices, seeds
    )

    executor = submitit.AutoExecutor(folder=str(args.logdir / "submitit" / "%j"))
    executor.update_parameters(
        cpus_per_task=10,
        slurm_time=120,
        slurm_gpus_per_node=1,
        slurm_ntasks_per_gpu=5,
        slurm_partition=args.partition,
        slurm_mem="16G",
        slurm_job_name="gcrl_submitit",
        slurm_mail_user="m.toepperwien@stud.uni-hannover.de",
        slurm_mail_type="BEGIN,FAIL,END",
    )
    executor.map_array(run_config, *zip(*arguments))

    return


def run_config_wrapper(args: argparse.Namespace) -> None:
    return run_config(
        args.logdir, args.phase, args.agent_path, args.configuration, args.seed
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    # Setup subcommand parsing
    # Creates configurations and folder to log into
    setup_subparser = subparsers.add_parser("setup")
    setup_subparser.add_argument(
        "--agent", required=True, type=str, choices=list(AGENT_CLASSES.keys())
    )
    setup_subparser.add_argument("--phases", required=True, type=int, nargs="+")
    setup_subparser.add_argument("--dataset", required=True, type=str)
    setup_subparser.add_argument("--n_configurations", required=True, type=int)
    setup_subparser.add_argument(
        "--hyperparameters", required=True, type=str, nargs="+"
    )
    setup_subparser.add_argument("--eval_steps", required=True, type=int, nargs="+")
    setup_subparser.add_argument("--save_steps", required=False, type=int, nargs="+")
    setup_subparser.add_argument("--eval_episodes", required=True, type=int)
    setup_subparser.add_argument("--seed", type=int, default=0)
    setup_subparser.add_argument("--logdir", type=Path, required=True)
    setup_subparser.set_defaults(func=run_setup)

    # Setup slurm parsing
    # Submits all jobs to slurm
    slurm_subparser = subparsers.add_parser("submit")
    slurm_subparser.add_argument("--logdir", type=Path, required=True)
    slurm_subparser.add_argument("--phase", required=True, type=int)
    slurm_subparser.add_argument("--agent_path", required=False, type=Path)
    slurm_subparser.add_argument("--n_seeds", type=int, required=True)
    slurm_subparser.add_argument("--partition", type=str, default="ai")
    slurm_subparser.set_defaults(func=submit)

    # Run subcommand parsing
    # Runs a configuration and a seed
    run_subparser = subparsers.add_parser("run")
    run_subparser.add_argument("--logdir", type=Path, required=True)
    run_subparser.add_argument("--phase", required=True, type=int)
    run_subparser.add_argument("--agent_path", required=False, type=Path)
    run_subparser.add_argument("--configuration", required=True, type=int)
    run_subparser.add_argument("--seed", type=int, required=True)
    run_subparser.set_defaults(func=run_config_wrapper)

    args = parser.parse_args()
    args.func(args)
