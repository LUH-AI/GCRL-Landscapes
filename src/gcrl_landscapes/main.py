from training import full_phased_run
from evaluate import evaluate_wrapper
import ogbench
from ogbench.impls.utils.datasets import GCDataset, Dataset
from ogbench.impls.agents import CRLAgent, CMDAgent, SACAgent
from configurations import generate_configurations
import argparse
from datetime import datetime
from pathlib import Path
import json

AGENT_CLASSES = {"CRL": CRLAgent, "CMD": CMDAgent, "SAC": SACAgent}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [TODO: add all agents]
    parser.add_argument(
        "--agent", required=True, type=str, choices=list(AGENT_CLASSES.keys())
    )
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--n_configurations", required=True, type=int)
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

    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(args.dataset)  # type: ignore
    configurations = generate_configurations(args.n_configurations, args.agent)

    results_per_phase = full_phased_run(
        phase_steps=args.phase_steps,
        configs=configurations,
        agent_class=AGENT_CLASSES[args.agent],  # type: ignore # types of ogbench are not properly defined
        env=env,
        train_datasets=[
            GCDataset(Dataset.create(**train_dataset), config)
            for config in configurations
        ],
        val_datasets=[
            GCDataset(Dataset.create(**val_dataset), config)
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
