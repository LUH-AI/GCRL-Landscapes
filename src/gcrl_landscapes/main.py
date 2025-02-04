from training import full_phased_run
from evaluate import evaluate_wrapper
import ogbench
from ogbench.impls.utils.datasets import HGCDataset, GCDataset, Dataset
from ogbench.impls.agents import CRLAgent, CMDAgent, GCBCAgent, QRLAgent, HIQLAgent
from configurations import generate_configurations
import argparse
from datetime import datetime
from pathlib import Path
import json
import git
import toml
import logging

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [TODO: add all agents]
    parser.add_argument(
        "--agent", required=True, type=str, choices=list(AGENT_CLASSES.keys())
    )
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--n_configurations", required=True, type=int)
    parser.add_argument("--hyperparameters", type=str, nargs="+")
    parser.add_argument("--phase_steps", required=True, type=int, nargs="+")
    parser.add_argument("--eval_steps", required=True, type=int, nargs="+")
    parser.add_argument("--save_steps", required=False, type=int, nargs="+")
    parser.add_argument("--eval_episodes", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    log_dir = Path("./logs") / Path(
        f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}_{args.agent}_{args.dataset}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=log_dir / "log.txt", level=logging.INFO)

    metadata = {
        "arguments": args.__dict__,
        "git": {
            "commit": git.Repo(".", search_parent_directories=True).head.object.hexsha,
            "branch": git.Repo(".", search_parent_directories=True).active_branch.name,
        },
    }
    with open(log_dir / "info.toml", "w") as f:
        f.write(toml.dumps(metadata))

    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(args.dataset)  # type: ignore
    configurations = generate_configurations(
        args.n_configurations, args.agent, set(args.hyperparameters)
    )

    results_per_phase = full_phased_run(
        phase_steps=args.phase_steps,
        configs=configurations,
        agent_class=AGENT_CLASSES[args.agent],  # type: ignore # types of ogbench are not properly defined
        env=env,
        train_datasets=[
            DATASET_CLASSES[args.agent](Dataset.create(**train_dataset), config)
            for config in configurations
        ],
        val_datasets=[
            DATASET_CLASSES[args.agent](Dataset.create(**val_dataset), config)
            for config in configurations
        ],
        eval_at_steps=args.eval_steps,
        evaluate=evaluate_wrapper,
        save_at_steps=args.save_steps if args.save_steps else args.eval_steps,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        log_dir=log_dir,
    )

    with open(log_dir / "results.json", "w") as f:
        results_saveable = {
            key: item.to_dict() for key, item in results_per_phase.items()
        }
        json.dump(results_saveable, f)
