from training import train
from evaluate import evaluate_wrapper
import ogbench
from ogbench.impls.utils.datasets import GCDataset, Dataset
from ogbench.impls.agents.crl import CRLAgent, get_config

if __name__ == "__main__":
    # [TODO: make this configurable by argument parser or loop over multiple setups]
    # this is only an exemplary setup to test data collection
    agent_class = CRLAgent
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets("humanoidmaze-large-navigate-v0")  # type: ignore
    train_steps = 1000
    eval_at_steps = [500]
    config = get_config()
    save_at_steps = [500]

    evaluation_results, models = train(
        agent_class=agent_class,
        agent_path=None,
        env=env,
        train_dataset=GCDataset(Dataset.create(**train_dataset), config),
        val_dataset=GCDataset(Dataset.create(**val_dataset), config),
        config=config,
        train_steps=train_steps,
        eval_at_steps=eval_at_steps,
        evaluate=evaluate_wrapper,
        save_at_steps=save_at_steps,
        eval_episodes=1,
    )
    print(evaluation_results)
    print(models)
