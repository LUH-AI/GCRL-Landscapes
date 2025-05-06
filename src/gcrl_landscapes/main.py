import argparse
from pathlib import Path
import logging
from .submission import run_config_wrapper, run_setup, submit, SUPPORTED_AGENTS

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    # Setup subcommand parsing
    # Creates configurations and folder to log into
    setup_subparser = subparsers.add_parser(
        "setup", description="Create folder structure with metadata for experiment."
    )
    setup_subparser.add_argument(
        "--agent",
        required=True,
        type=str,
        choices=SUPPORTED_AGENTS,
        help="Agent to run",
    )
    setup_subparser.add_argument(
        "--phases",
        required=True,
        type=int,
        nargs="+",
        help="Which phases should be run in the course of the experiment",
    )
    setup_subparser.add_argument(
        "--dataset", required=True, type=str, help="Which dataset/environment to run"
    )
    setup_subparser.add_argument(
        "--n_configurations",
        required=True,
        type=int,
        help="How many configurations to run/sample across configuration space",
    )
    setup_subparser.add_argument(
        "--hyperparameters",
        required=True,
        type=str,
        nargs="+",
        help="Which configuration space to use",
    )
    setup_subparser.add_argument(
        "--eval_steps",
        required=True,
        type=int,
        nargs="+",
        help="At what steps to evaluate",
    )
    setup_subparser.add_argument(
        "--save_steps",
        required=False,
        type=int,
        nargs="+",
        help="At what steps to save a checkpoint. Will save checkpoints for best configurations",
    )
    setup_subparser.add_argument(
        "--eval_episodes",
        required=True,
        type=int,
        help="How many evaluation episodes to run per evaluation",
    )
    setup_subparser.add_argument(
        "--seed", type=int, default=0, help="Which seed to use"
    )
    setup_subparser.add_argument(
        "--logdir", type=Path, required=True, help="in which directory to save results"
    )
    setup_subparser.add_argument(
        "--final_step_is_phase",
        action="store_true",
        help="Train configs in phase only till end of phase.",
    )
    setup_subparser.set_defaults(func=run_setup)

    # Setup slurm parsing
    # Submits all jobs to slurm
    slurm_subparser = subparsers.add_parser("submit")
    slurm_subparser.add_argument(
        "--logdir",
        type=Path,
        required=True,
        help="In which directory to save results. Has to be already initialized using setup",
    )
    slurm_subparser.add_argument(
        "--agent_path",
        required=False,
        type=Path,
        help="Which checkpoint to load. Will search best checkpoint itself if not given",
    )
    slurm_subparser.add_argument(
        "--n_seeds", type=int, required=True, help="How many seeds to run"
    )
    slurm_subparser.add_argument(
        "--partition",
        type=str,
        default="ai",
        help="On which slurm partition to schedule the jobs",
    )
    slurm_subparser.add_argument(
        "--mem_per_cpu",
        type=str,
        default="3G",
        required=False,
        help="memory given to node per allocated cpu",
    )
    slurm_subparser.add_argument(
        "--tasks_per_node_total",
        type=int,
        required=True,
        help="How many tasks/configurations should run per node in total",
    )
    slurm_subparser.add_argument(
        "--tasks_per_node_parallel",
        type=int,
        required=True,
        help="How many tasks/configurations should run per node in parallel",
    )
    slurm_subparser.add_argument(
        "--jobname", required=True, type=str, help="How to name the job in slurm"
    )
    slurm_subparser.add_argument(
        "--min_per_mill_steps",
        type=int,
        default=300,
        required=False,
        help="How many minutes should be allocated on slurm as limit per million steps taken",
    )
    slurm_subparser.add_argument(
        "--basetime",
        type=int,
        required=False,
        default=30,
        help="Time allocation per Job independent from training steps",
    )
    slurm_subparser.add_argument(
        "--phases",
        required=False,
        nargs="+",
        type=int,
        help="Which phases to run. Will run all if not given",
    )

    slurm_subparser.set_defaults(func=submit)

    # Run subcommand parsing
    # Runs a configuration and a seed
    run_subparser = subparsers.add_parser("run")
    run_subparser.add_argument(
        "--logdir",
        type=Path,
        required=True,
        help="In which directory to save results. Has to be already initialized using setup",
    )
    run_subparser.add_argument(
        "--phase", required=True, type=int, help="Which phase to run"
    )
    run_subparser.add_argument(
        "--agent_path",
        required=False,
        type=Path,
        help="Which checkpoint to load. Will search best checkpoint if not given",
    )
    run_subparser.add_argument(
        "--configuration",
        required=True,
        type=int,
        help="which configuration (index) to run",
    )
    run_subparser.add_argument(
        "--seed", type=int, required=True, help="Which seed to use for training"
    )
    run_subparser.add_argument(
        "--tasks_per_node_parallel",
        type=int,
        default=1,
        help="Limits jax gpu pre-allocation in case of multiple jobs running. See submit for more details",
    )
    run_subparser.set_defaults(func=run_config_wrapper)

    args = parser.parse_args()
    args.func(args)
