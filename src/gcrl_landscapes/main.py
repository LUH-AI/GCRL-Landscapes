from training import full_phased_run
from evaluate import evaluate_wrapper
import ogbench
from ogbench.impls.utils.datasets import GCDataset, Dataset
from ogbench.impls.agents.crl import CRLAgent, get_config
from ml_collections import FrozenConfigDict

if __name__ == "__main__":
    # [TODO: make this configurable by argument parser or loop over multiple setups]
    # this is only an exemplary setup to test data collection
    agent_class = CRLAgent
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        "humanoidmaze-large-navigate-v0"
    )  # type: ignore
    train_steps = 1000
    eval_at_steps = [500]
    config = get_config()
    save_at_steps = [500]

    results_per_phase = full_phased_run(
        phase_steps=[10, 20, 30, 40, 100],
        configs=[FrozenConfigDict(config)],
        agent_class=CRLAgent,  # type: ignore # types of ogbench are not properly defined
        env=env,
        train_dataset=GCDataset(Dataset.create(**train_dataset), config),
        val_dataset=GCDataset(Dataset.create(**val_dataset), config),
        eval_at_steps=[10, 20, 30, 40, 50, 100],
        evaluate=evaluate_wrapper,
        save_at_steps=[10, 20, 30, 40, 100],
        eval_episodes=2,
    )
    print(results_per_phase)
