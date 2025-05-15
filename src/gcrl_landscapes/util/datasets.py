from pathlib import Path
from ogbench import make_env_and_datasets


def mix_datasets(
    base_dataset_name: str, second_dataset_name: str, second_share: int
) -> Path:
    # load datasets
    base_dataset: dict
    base_dataset_val: dict
    second_dataset: dict
    second_dataset_val: dict
    _, base_dataset, base_dataset_val = make_env_and_datasets(base_dataset_name)
    _, second_dataset, second_dataset_val = make_env_and_datasets(second_dataset_name)

    # mix

    # save
