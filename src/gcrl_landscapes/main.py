from training import full_phased_run
from evaluate import evaluate_wrapper
import ogbench
from ogbench.impls.utils.datasets import GCDataset, Dataset
from ogbench.impls.agents.crl import CRLAgent
from configurations import generate_configurations_crl

if __name__ == "__main__":
    # [TODO: make this configurable by argument parser or loop over multiple setups]
    # this is only an exemplary setup to test data collection
    agent_class = CRLAgent
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        "antmaze-medium-navigate-v0"
    )  # type: ignore
    configurations = generate_configurations_crl(128)

    results_per_phase = full_phased_run(
        phase_steps=[50000, 100000, 200000],
        configs=configurations,
        agent_class=CRLAgent,  # type: ignore # types of ogbench are not properly defined
        env=env,
        train_datasets=[
            GCDataset(Dataset.create(**train_dataset), config)
            for config in configurations
        ],
        val_datasets=[
            GCDataset(Dataset.create(**val_dataset), config)
            for config in configurations
        ],
        eval_at_steps=[10000, 50000, 100000, 200000, 500000],
        evaluate=evaluate_wrapper,
        save_at_steps=[10000, 50000, 100000, 200000, 500000],
        eval_episodes=10,
    )
    print(results_per_phase)
