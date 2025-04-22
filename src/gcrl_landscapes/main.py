# A lot of the imports are lazy loaded. This is needed to prevent issues from coming up during submitit, as it will run on different hardware configurations

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
from .util.misc import retry_call
import re
from .util.data import get_best_agent_path

logger = logging.getLogger(__name__)

# None to allow for lazy loading of ogbench
AGENT_CLASSES = {
    "CRL": None,
    "CMD": None,
    "GCBC": None,
    "GCIQL": None,
    "GCIVL": None,
    "QRL": None,
    "HIQL": None,
}
DATASET_CLASSES = {
    "CRL": None,
    "CMD": None,
    "GCBC": None,
    "GCIQL": None,
    "GCIVL": None,
    "QRL": None,
    "HIQL": None,
}


def run_setup(args: argparse.Namespace) -> None:
    from .configurations import generate_configurations, get_config_space

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

    def save_info():
        with open(args.logdir / "info.toml", "w") as f:
            f.write(toml.dumps(metadata))

    retry_call(save_info)

    # Save configurations
    config_dir = args.logdir / "configurations"
    config_dir.mkdir()

    config_space = get_config_space(args.agent)
    config_space.to_json(config_dir / "configspace.json")

    configurations = generate_configurations(
        args.n_configurations, args.agent, set(args.hyperparameters), seed=args.seed
    )

    def save_configurations():
        for i, config in enumerate(configurations):
            with open(config_dir / f"configuration_{i}.json", "w") as f:
                f.write(config.to_json())

    retry_call(save_configurations)

    return


def run_config(
    logdir: Path,
    phase: int,
    configuration_index: int,
    seed: int,
    tasks_per_node: int,
) -> None:
    setup = toml.load(args.logdir / "info.toml")["arguments"]

    # submitit just bypasses SIGTERM although it should end the job, overwrite that behaviour here
    def handler(signum, frame):
        print(f"Received {signal.Signals(signum).name} ({signum}), stopping!")
        sys.exit(1)

    signal.signal(signal.SIGTERM, handler)

    # [TODO: do backend setting more cleanly]
    import os
    import numpy as np

    # This will set egl (nvidia) as the backend for mujoco and may lead to failure on other systems
    os.environ["MUJOCO_GL"] = "egl"
    # let jax only pre-allocate a fraction of gpus memory, so that all tasks on node can run
    fraction_gpu_allocation = np.round(1 / (tasks_per_node + 1), 2)
    fraction_only_decimal = f"{fraction_gpu_allocation}".split(".")[1]
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = (
        f".{fraction_only_decimal}"  # one more task to leave some space in gpu ram
    )

    from .training import train
    from .evaluate import evaluate_wrapper
    from ogbench import make_env_and_datasets
    from ogbench.impls.utils.datasets import HGCDataset, GCDataset, Dataset
    from ogbench.impls.agents import (
        CRLAgent,
        CMDAgent,
        GCBCAgent,
        QRLAgent,
        HIQLAgent,
        GCIQLAgent,
        GCIVLAgent,
    )
    import jax
    from .util.misc import jax_has_gpu

    print(f"Default backend: {jax.default_backend()}, running on gpu?: {jax_has_gpu()}")

    # populate agent classes and dataset classes after loading ogbench
    AGENT_CLASSES = {
        "CRL": CRLAgent,
        "CMD": CMDAgent,
        "GCBC": GCBCAgent,
        "GCIQL": GCIQLAgent,
        "GCIVL": GCIVLAgent,
        "QRL": QRLAgent,
        "HIQL": HIQLAgent,
    }
    DATASET_CLASSES = {
        "CRL": GCDataset,
        "CMD": GCDataset,
        "GCBC": GCDataset,
        "GCIQL": GCDataset,
        "GCIVL": GCDataset,
        "QRL": GCDataset,
        "HIQL": HGCDataset,
    }

    # Find best agent for last phase
    last_phase_index = setup["phases"].index(phase) - 1
    already_trained_steps = (
        setup["phases"][last_phase_index] if last_phase_index >= 0 else 0
    )
    agent_path = (
        get_best_agent_path(already_trained_steps, logdir)
        if already_trained_steps > 0
        else None
    )

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

    def save_metadata():
        with open(run_log_dir / "info.toml", "w") as f:
            f.write(toml.dumps(metadata))

    retry_call(save_metadata)

    env, train_dataset, val_dataset = make_env_and_datasets(setup["dataset"])  # type: ignore

    with open(
        logdir / "configurations" / f"configuration_{configuration_index}.json", "r"
    ) as f:
        configuration = FrozenConfigDict(json.loads(f.read()))

    # don't train until end if we don't use final timestep as fitness evaluation
    setup_eval_steps = setup["eval_steps"]
    setup_save_steps = (
        setup["save_steps"] if "save_steps" in setup else setup_eval_steps
    )
    if setup["final_step_is_phase"]:
        eval_steps = [step for step in setup_eval_steps if step <= phase]
        save_steps = [step for step in setup_save_steps if step <= phase]
    else:
        eval_steps = setup_eval_steps
        save_steps = setup_save_steps

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
        already_trained_steps=already_trained_steps,
        eval_at_steps=eval_steps,
        evaluate=evaluate_wrapper,
        config=configuration,
        save_at_steps=save_steps,
        eval_episodes=setup["eval_episodes"],
        log_dir=run_log_dir,
        seed=seed,
    )

    def save_results():
        with open(run_log_dir / "eval_trajectory.json", "w") as f:
            f.write(eval_trajectory.to_json())

    retry_call(save_results)


def submit(args: argparse.Namespace) -> None:
    setup = toml.load(args.logdir / "info.toml")["arguments"]

    configuration_indices = list(range(setup["n_configurations"]))
    seeds = list(range(args.n_seeds))
    array_id: int | None = None
    phases = setup["phases"] if not args.phases else args.phases
    for phase in phases:
        argument_lines = product(
            [args.logdir],
            [phase],
            configuration_indices,
            seeds,
            [args.tasks_per_node],
        )
        argument_columns = list(zip(*argument_lines))
        chunked_arguments = [
            [
                argument_columns[column_idx][i : i + args.tasks_per_node]
                for i in range(
                    0, len(argument_columns[column_idx]), args.tasks_per_node
                )
            ]
            for column_idx in range(len(argument_columns))
        ]

        last_phase_index = setup["phases"].index(phase) - 1
        already_trained_steps = (
            setup["phases"][last_phase_index] if last_phase_index >= 0 else 0
        )

        steps_to_train = (
            max(setup["eval_steps"]) - already_trained_steps
            if not setup["final_step_is_phase"]
            else phase - already_trained_steps
        )
        executor = submitit.AutoExecutor(folder=str(args.logdir / "submitit" / "%j"))
        executor.update_parameters(
            cpus_per_task=3,
            slurm_time=int(
                args.basetime
                + args.min_per_mill_steps
                * (steps_to_train / 1_000_000)
                * args.tasks_per_node
            ),  # this overestimates, keep safety margin
            slurm_gpus_per_node=1,
            tasks_per_node=args.tasks_per_node,
            slurm_mem_per_cpu="3G",
            slurm_array_parallelism=50,
            slurm_partition=args.partition,
            slurm_job_name=args.jobname,
            slurm_mail_user="m.toepperwien@stud.uni-hannover.de",
            slurm_mail_type="FAIL,END",
            slurm_additional_parameters={"dependency": f"afterok:{array_id}"}
            if array_id
            else {},
        )
        jobs = executor.map_array(run_config_slurm_tasks_wrapper, *chunked_arguments)
        # parse job array number/array id without subtaskid
        array_id_match = re.fullmatch(r"^(?P<array_id>\d+)_\d+$", jobs[0].job_id)
        if not array_id_match:
            raise Exception(
                f'Slurm job id does not match expected format "\\d+_\\d+": {jobs[0].job_id}'
            )
        array_id = int(array_id_match.groupdict()["array_id"])

    return


def run_config_slurm_tasks_wrapper(
    logdirs: list[Path],
    phases: list[int],
    configuration_indices: list[int],
    seeds: list[int],
    tasks_per_node: list[int],
):
    assert (
        len(logdirs)
        == len(phases)
        == len(configuration_indices)
        == len(seeds)
        == len(tasks_per_node)
    )
    job_env = submitit.JobEnvironment()
    print(f"There are {job_env.num_tasks} in this job")
    print(f"I'm the task #{job_env.local_rank} on the node {job_env.node}")
    print(f"I'm the task #{job_env.global_rank} in the job")
    r = job_env.local_rank
    if r >= len(logdirs):
        print(f"Not enough jobs ({len(logdirs)}) for this node. Exiting ...")
        sys.exit(0)
    return run_config(
        logdirs[r],
        phases[r],
        configuration_indices[r],
        seeds[r],
        tasks_per_node[r],
    )


def run_config_wrapper(args: argparse.Namespace) -> None:
    return run_config(
        args.logdir,
        args.phase,
        args.configuration,
        args.seed,
        args.tasks_per_node,
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
    setup_subparser.add_argument(
        "--final_step_is_phase",
        action="store_true",
        help="Train configs in phase only till end of phase.",
    )
    setup_subparser.set_defaults(func=run_setup)

    # Setup slurm parsing
    # Submits all jobs to slurm
    slurm_subparser = subparsers.add_parser("submit")
    slurm_subparser.add_argument("--logdir", type=Path, required=True)
    slurm_subparser.add_argument("--n_seeds", type=int, required=True)
    slurm_subparser.add_argument("--partition", type=str, default="ai")
    slurm_subparser.add_argument("--tasks_per_node", type=int, required=True)
    slurm_subparser.add_argument("--jobname", required=True, type=str)
    slurm_subparser.add_argument(
        "--min_per_mill_steps", type=int, default=300, required=False
    )
    slurm_subparser.add_argument(
        "--basetime",
        type=int,
        required=False,
        default=30,
        help="Time allocation per Job independent from training steps",
    )
    slurm_subparser.add_argument("--phases", required=False, nargs="+", type=int)
    slurm_subparser.set_defaults(func=submit)

    # Run subcommand parsing
    # Runs a configuration and a seed
    run_subparser = subparsers.add_parser("run")
    run_subparser.add_argument("--logdir", type=Path, required=True)
    run_subparser.add_argument("--phase", required=True, type=int)
    run_subparser.add_argument("--agent_path", required=False, type=Path)
    run_subparser.add_argument("--configuration", required=True, type=int)
    run_subparser.add_argument("--seed", type=int, required=True)
    run_subparser.add_argument(
        "--tasks_per_node",
        type=int,
        default=1,
        help="Limits jax gpu pre-allocation in case of multiple jobs running",
    )
    run_subparser.set_defaults(func=run_config_wrapper)

    args = parser.parse_args()
    args.func(args)
