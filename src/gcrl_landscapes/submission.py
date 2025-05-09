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
from .util.data import get_best_agent_path, get_phase_results
from .phase_splitting import get_all_phases
from more_itertools import chunked, divide

logger = logging.getLogger(__name__)


SUPPORTED_AGENTS = [
    "CRL",
    "CMD",
    "GCBC",
    "GCIQL",
    "GCIVL",
    "QRL",
    "HIQL",
]
SUPPORTED_DATASETS = [
    "CRL",
    "CMD",
    "GCBC",
    "GCIQL",
    "GCIVL",
    "QRL",
    "HIQL",
]


def run_setup(args: argparse.Namespace) -> None:
    """Create file structure and save common parameters for the whole experiment

    Args:
        args: parsed arguments, look into main or run from commandline to see documentation
    """
    from .configurations import (
        generate_configurations,
        get_config_space,
        get_adapted_default_config,
    )

    args.logdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.logdir / "log.txt", level=logging.INFO)
    logging.info("Set up for later running")

    phases = (
        [
            get_all_phases(
                args.agent,
                args.dataset,
                args.final_performance_percentage,
                args.convergence_zip,
                args.phase_percentages,
                mode="target_ratio",
                interpolation="linear_target",
            )
        ]
        if args.convergence_zip
        else [
            int((phase_percentage / 100) * args.final_phase)
            for phase_percentage in args.phase_percentages
        ]
    )
    # Log metadata to file
    metadata = {
        "arguments": args.__dict__ | {"phases": phases},
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

    if args.n_configurations > 1:
        configurations = generate_configurations(
            args.n_configurations,
            args.agent,
            set(args.hyperparameters),
            seed=args.seed,
            env=args.dataset,
        )
    else:
        configurations = [get_adapted_default_config(args.agent, args.dataset)]

    def save_configurations():
        for i, config in enumerate(configurations):
            with open(config_dir / f"configuration_{i}.json", "w") as f:
                f.write(config.to_json())

    retry_call(save_configurations)

    return


def run_config(
    logdir: Path,
    phase_idx: int,
    configuration_index: int,
    seed: int,
    tasks_per_node_parallel: int,
    clean_checkpoints: bool = False,
    agent_path: Path | None = None,
) -> None:
    """Run given config on GPU and collect data
    This function is mainly organizational to e.g. find the best agent from the previous phase, prepare the environment and save metadata. It calls the train function to do the actual training.

    Args:
        logdir: logdir for current experiment where metadata resides. Will create its own subdirectory
        phase: phase to run
        agent_path: Which agent to load. Will load metadata from last phase if None to find the best agent itself
        configuration_index: configuration to use for training
        seed: seed for training
        tasks_per_node_parallel: how many tasks will run on this node. Needed for environment setup
    """
    assert phase_idx == 0 or agent_path

    setup = toml.load(logdir / "info.toml")["arguments"]
    phase = setup["phases"][phase_idx]

    # submitit just bypasses SIGTERM although it should end the job, overwrite that behaviour here
    def handler(signum, frame):
        print(f"Received {signal.Signals(signum).name} ({signum}), stopping!")
        sys.exit(1)

    signal.signal(signal.SIGTERM, handler)

    # Set GPU training environment variables before loading modules
    import os
    import numpy as np

    # This will set egl (nvidia) as the backend for mujoco and may lead to failure on other systems
    os.environ["MUJOCO_GL"] = "egl"
    # let jax only pre-allocate a fraction of gpus memory, so that all tasks on node can run
    fraction_gpu_allocation = np.round(1 / (tasks_per_node_parallel + 1), 2)
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
    already_trained_steps = setup["phases"][phase_idx - 1] if phase_idx - 1 >= 0 else 0
    if not agent_path and already_trained_steps > 0:
        print("No agent path given, finding best agent")
        phase_results = get_phase_results(already_trained_steps, logdir)
        agent_path = get_best_agent_path(phase_results)
        # delete unneeded checkpoints
        if clean_checkpoints:
            print("Cleaning checkpoints")
            for path in map(Path, phase_results["path"]):
                if str(path) != str(agent_path):
                    path.unlink(missing_ok=True)
    print(f"Loading agent {agent_path}")

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
            "tasks_per_node_parallel": tasks_per_node_parallel,
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

    # Set up training
    env, train_dataset, val_dataset = make_env_and_datasets(setup["dataset"])  # type: ignore
    with open(
        logdir / "configurations" / f"configuration_{configuration_index}.json", "r"
    ) as f:
        configuration = FrozenConfigDict(json.loads(f.read()))

    # don't train until end if we don't use final timestep as fitness evaluation
    setup_eval_steps = sorted(setup["extra_eval_steps"] + setup["phases"])
    setup_save_steps = (
        sorted(setup["extra_save_steps"] + [phase])
        if "extra_save_steps" in setup
        else [phase]
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
    """Submit jobs on the cluster. Does also support local execution to some degree.

    Args:
        args: Look into main argument parser options or run from commandline for documentation
    """
    setup = toml.load(args.logdir / "info.toml")["arguments"]

    # Generate arguments for jobs
    configuration_indices = list(range(setup["n_configurations"]))
    seeds = list(range(args.n_seeds))
    array_id: int | None = None
    phase_indices = (
        list(range(len(setup["phases"])))
        if not args.phase_indices
        else args.phase_indices
    )
    for phase_idx in phase_indices:
        argument_lines = [
            {
                "logdir": args.logdir,
                "phase_idx": phase_idx,
                "configuration_idx": configuration_idx,
                "seed": seed,
                "tasks_per_node_parallel": args.tasks_per_node_parallel,
            }
            for configuration_idx, seed in product(
                configuration_indices,
                seeds,
            )
        ]
        # first chunk tasks by total tasks per node
        chunked_tasks_per_node: list[list[dict]] = list(
            chunked(argument_lines, args.tasks_per_node_total)
        )
        # then create n tasks_per_node_parallel groups for parallel execution
        # resulting dimensions: (node, parallel_groups, sequential_groups)
        chunked_tasks_parallel: list[list[list[dict]]] = [
            [
                list(divided_chunk)
                for divided_chunk in divide(args.tasks_per_node_parallel, task_chunk)
            ]
            for task_chunk in chunked_tasks_per_node
        ]

        already_trained_steps = (
            setup["phases"][phase_idx - 1] if phase_idx - 1 >= 0 else 0
        )

        steps_to_train = (
            max(setup["eval_steps"]) - already_trained_steps
            if not setup["final_step_is_phase"]
            else setup["phases"][phase_idx] - already_trained_steps
        )
        executor = submitit.AutoExecutor(folder=str(args.logdir / "submitit" / "%j"))
        executor.update_parameters(
            cpus_per_task=3,
            slurm_time=int(
                args.basetime
                + args.min_per_mill_steps
                * (steps_to_train / 1_000_000)
                * args.tasks_per_node_total
            ),  # this overestimates, keep safety margin
            slurm_gpus_per_node=1,
            tasks_per_node=args.tasks_per_node_parallel,
            slurm_mem_per_cpu=args.mem_per_cpu,
            slurm_array_parallelism=50,
            slurm_partition=args.partition,
            slurm_job_name=args.jobname,
            slurm_mail_user="m.toepperwien@stud.uni-hannover.de",
            slurm_mail_type="END,FAIL",
            slurm_additional_parameters={"dependency": f"afterok:{array_id}"}
            if array_id
            else {},
        )
        jobs = executor.map_array(
            run_config_chunked_arguments_wrapper, chunked_tasks_parallel
        )
        # parse job array number/array id without subtaskid
        array_id_match = re.fullmatch(r"^(?P<array_id>\d+)_\d+$", jobs[0].job_id)
        if not array_id_match:
            raise Exception(
                f'Slurm job id does not match expected format "\\d+_\\d+": {jobs[0].job_id}'
            )
        array_id = int(array_id_match.groupdict()["array_id"])

    return


def run_config_chunked_arguments_wrapper(
    arguments: list[list[dict]],
):
    """Run chunked arguments. This will typically be executed on a slurm node per task.
    All tasks receive the same chunked arguments and we have to properly index them to pass to correct task.
    For more documentation look at `run_config()` and `submit`
    """
    job_env = submitit.JobEnvironment()
    r = job_env.local_rank
    print(f"There are {job_env.num_tasks} in this job")
    print(f"I'm the task #{job_env.local_rank} on the node {job_env.node}")
    print(f"I'm the task #{job_env.global_rank} in the job")
    print(f"I will run {len(arguments[r])} jobs")
    if r >= len(arguments):
        print(f"Not enough jobs ({len(arguments)}) for this node. Exiting ...")
        sys.exit(0)

    first_job_and_task_in_array = int(job_env.array_task_id) == 0 and r == 0

    results = [
        run_config(
            setup["logdir"],
            setup["phase_idx"],
            setup["configuration_index"],
            setup["seed"],
            setup["tasks_per_node_parallel"],
            clean_checkpoints=first_job_and_task_in_array and seq_idx == 0,
        )
        for seq_idx, setup in enumerate(arguments[r])
    ]
    return results


def run_config_wrapper(args: argparse.Namespace) -> None:
    return run_config(
        args.logdir,
        args.phase_idx,
        args.configuration,
        args.seed,
        args.tasks_per_node_parallel,
        agent_path=args.agent_path,
    )
