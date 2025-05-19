from pathlib import Path
from ogbench import make_env_and_datasets
import numpy as np
from math import ceil, floor


def mix_datasets(
    base_dataset_name: str,
    second_dataset_name: str,
    target_name: str,
    second_share: int,
    seed: int = 0,
    output_dir: Path = Path("/tmp"),
) -> tuple[Path, Path]:
    # load datasets
    base_dataset: dict[str, np.ndarray]
    base_val_dataset: dict[str, np.ndarray]
    second_dataset: dict[str, np.ndarray]
    second_val_dataset: dict[str, np.ndarray]
    _, base_dataset, base_val_dataset = make_env_and_datasets(base_dataset_name)
    _, second_dataset, second_val_dataset = make_env_and_datasets(second_dataset_name)

    # mix
    rng = np.random.default_rng(seed=seed)

    base_dataset_size = list(base_dataset.values())[0].shape[0]
    second_dataset_size = list(second_dataset.values())[0].shape[0]
    array1_indices = rng.choice(
        base_dataset_size,
        ceil(base_dataset_size * (1 - (second_share / 100))),
        replace=False,
    )
    array2_indices = rng.choice(
        second_dataset_size,
        floor(base_dataset_size * (second_share / 100)),
        replace=False,
    )

    base_val_dataset_size = list(base_val_dataset.values())[0].shape[0]
    second_val_dataset_size = list(second_val_dataset.values())[0].shape[0]
    array1_val_indices = rng.choice(
        base_val_dataset_size,
        ceil(base_val_dataset_size * (1 - (second_share / 100))),
        replace=False,
    )
    array2_val_indices = rng.choice(
        second_val_dataset_size,
        floor(base_val_dataset_size * (second_share / 100)),
        replace=False,
    )

    mixed_dataset = {
        key: np.concatenate(
            (value_base[array1_indices], second_dataset[key][array2_indices])
        )
        for key, value_base in base_dataset.items()
    }
    mixed_dataset_val = {
        key: np.concatenate(
            (
                value_base[array1_val_indices],
                second_val_dataset[key][array2_val_indices],
            )
        )
        for key, value_base in base_val_dataset.items()
    }

    # save
    mixed_dataset_path = output_dir / f"{target_name}.npz"
    mixed_dataset_val_path = output_dir / f"{target_name}-val.npz"
    np.savez(mixed_dataset_path, **mixed_dataset)
    np.savez(mixed_dataset_val_path, **mixed_dataset_val)

    return mixed_dataset_path, mixed_dataset_val_path
