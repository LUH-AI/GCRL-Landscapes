from training import train
import ogbench
from ogbench.impls.utils.datasets import GCDataset, Dataset
from ogbench.impls.agents.crl import CRLAgent, get_config

if __name__ == "__main__":
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets("humanoidmaze-large-navigate-v0")  # type: ignore


    train(
        agent_class=CRLAgent,
        env=env,
        train_dataset=GCDataset(Dataset.create(**train_dataset), get_config()),
        val_dataset=GCDataset(Dataset.create(**val_dataset), get_config()),
        config = get_config(),
        train_steps=1000,
        eval_interval=100,
        evaluate=lambda a,b,c: ([], {}, [], []),
        save_at_steps=[500],
    )
