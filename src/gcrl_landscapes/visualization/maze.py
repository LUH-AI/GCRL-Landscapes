from ogbench import make_env_and_datasets
import re
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb as matplotlib_color_to_rgb
from PIL import Image, ImageOps, ImageDraw
import argparse
from functools import reduce
from pathlib import Path

FIGSIZE = (10, 10)
MAX_IMAGE_SIZE = 1000
SUPERSAMPLING_FACTOR = 10


def to_trajectories(
    dataset: dict[str, np.ndarray], batch_size: int | None = None, replace: bool = False
) -> list[dict[str, np.ndarray]]:
    if batch_size is not None and batch_size < 1:
        return []

    assert all(
        [dataset_member in dataset for dataset_member in ["observations", "terminals"]]
    )

    trajectory_ends = np.where(dataset["terminals"] > 0)[0]
    assert isinstance(trajectory_ends, np.ndarray)

    trajectory_starts = np.insert(trajectory_ends[:-1] + 1, 0, [0])
    trajectory_indices = np.column_stack((trajectory_starts, trajectory_ends))

    if not batch_size:
        batch_size = len(trajectory_indices)
    return [
        {key: value[traj_start : traj_end + 1] for key, value in dataset.items()}
        for traj_start, traj_end in trajectory_indices[
            np.random.choice(
                trajectory_indices.shape[0], size=batch_size, replace=replace
            ),
            :,
        ]
    ]


def get_2d_colors(points, min_point, max_point):
    """Get colors corresponding to 2-D points.
    Adapted from OGBench (Park et al. 2025).
    """
    points = np.array(points)
    min_point = np.array(min_point)
    max_point = np.array(max_point)

    colors = (points - min_point) / (max_point - min_point)
    colors = np.hstack((colors, (2 - np.sum(colors, axis=1, keepdims=True)) / 2))
    colors = np.clip(colors, 0, 1)
    colors = np.c_[colors, np.full(len(colors), 0.8)]

    return colors


def xy_to_ij(
    xy: np.ndarray,
    env,  # no type hint, as ogbench creates dynamic types during runtime
) -> np.ndarray:
    # See ogbench MazeEnv
    i = (xy[:, 0] + env._offset_x) / env._maze_unit
    j = (xy[:, 1] + env._offset_y) / env._maze_unit
    return np.column_stack((i, j))


def visualize_trajs(
    trajs: list[dict[str, np.ndarray]], background: Image.Image, env
) -> Image.Image:
    """Visualize x-y trajectories in maze environments."""
    visualized_image = background.copy()
    draw = ImageDraw.Draw(visualized_image)

    colors = [colormaps.get_cmap("tab20")(i) for i in range(len(trajs))]
    for traj, color in zip(trajs, colors):
        ij = xy_to_ij(traj["observations"][:, :2], env)
        draw.line(
            ((ij + 0.5) / env.maze_map.transpose().shape * background.size).tolist(),
            width=2 * SUPERSAMPLING_FACTOR,
            fill=tuple(
                [round(channel * 255) for channel in matplotlib_color_to_rgb(color)]
            ),
        )

    return visualized_image


def sample_trajectories(
    num_dataset_pairs: list[tuple[int, dict[str, np.ndarray]]],
) -> list[dict[str, np.ndarray]]:
    return reduce(
        lambda a, b: a + b,
        [
            to_trajectories(dataset, batch_size=num)
            for num, dataset in num_dataset_pairs
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Which dataset to visualize")
    parser.add_argument(
        "--num_trajectories",
        required=False,
        default=10,
        type=int,
        help="How many trajectories to visualize",
    )
    args = parser.parse_args()
    print(args.dataset)

    output_folder = (
        Path("plots")
        / "visualizations"
        / "datasets"
        / f"{args.num_trajectories}-trajectories"
    )
    output_folder.mkdir(exist_ok=True, parents=True)

    explore_mix_match = re.fullmatch(
        r"^.*explore(?P<explore_share>\d+)(?P<secondtype>[^-]*).*$", args.dataset
    )
    if explore_mix_match:
        base_dataset = re.sub(r"explore\d+", "", args.dataset)
        explore_dataset = re.sub(r"explore\d+[^-]*", "explore", args.dataset)
        explore_share = int(explore_mix_match.groupdict()["explore_share"])
        env, train_base_dataset_raw, val_dataset1_raw = make_env_and_datasets(
            base_dataset
        )  # type: ignore
        _, train_explore_dataset, val_dataset2_raw = make_env_and_datasets(
            explore_dataset
        )  # type: ignore
        num_explore_samples = round(args.num_trajectories * (explore_share / 100))
        trajectories = sample_trajectories(
            [
                (args.num_trajectories - num_explore_samples, train_base_dataset_raw),
                (num_explore_samples, train_explore_dataset),
            ]
        )
    else:
        env, train_dataset_raw, val_dataset_raw = make_env_and_datasets(  # type: ignore
            args.dataset
        )
        trajectories = sample_trajectories([(args.num_trajectories, train_dataset_raw)])

    assert len(trajectories) == args.num_trajectories
    # move goal and ant out of picture
    env.unwrapped.set_goal(goal_xy=(-10, -10))  # type: ignore
    env.unwrapped.set_xy((-10, -10))  # type: ignore
    visualized_env = env.unwrapped.render()  # type: ignore

    background = ImageOps.contain(
        ImageOps.invert(
            Image.fromarray(env.unwrapped.maze_map.astype(np.uint8) * 255)  # type: ignore
        ),
        (MAX_IMAGE_SIZE * SUPERSAMPLING_FACTOR, MAX_IMAGE_SIZE * SUPERSAMPLING_FACTOR),
        method=Image.Resampling.BOX,
    )

    visualized_trajs = visualize_trajs(
        trajectories,
        ImageOps.colorize(background, black="#303e57", white="white"),
        env.unwrapped,
    )

    ImageOps.contain(
        visualized_trajs,
        (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE),
        method=Image.Resampling.LANCZOS,
    ).save(output_folder / f"trajvisualization_{args.dataset}.png")
