import argparse
from gcrl_landscapes.util.datasets import create_env_and_dataset
from gcrl_landscapes.configurations import get_adapted_default_config
from .maze import xy_to_ij
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean
from PIL import Image, ImageOps
import io
from pathlib import Path
import os
import multiprocessing
from functools import partial


DATASET_SAMPLE_SIZE = 100_000


def sample_to_coordinates(sample: dict) -> np.ndarray:
    return sample["observations"][:, :2]


def fig_on_maze(density_plot: Image.Image, background: Image.Image) -> Image.Image:
    """Overlay density onto maze

    Args:
        density_plot: density plot (heatmap)
        background: maze

    Returns:
        Returns density onto maze (image)
    """
    white_bg = Image.new("RGBA", density_plot.size, (255, 255, 255, 255))
    background = ImageOps.contain(
        background.copy(), size=density_plot.size, method=Image.Resampling.BOX
    )
    background.putalpha(background.split()[0].point(lambda p: 0 if p > 240 else 255))
    density_maze_w_alpha = Image.alpha_composite(density_plot, background)
    return Image.alpha_composite(white_bg, density_maze_w_alpha)


def plot_dataset_heatmap(dataset: str, output_folder: Path) -> None:
    # agent class and actor loss do not matter much here, as they influence parts of the dataset we do not use anyways
    env, train_dataset, val_dataset = create_env_and_dataset(
        dataset, "CRL", get_adapted_default_config("CRL", dataset, "awr")
    )

    # Create maze image
    # move goal and ant out of picture
    env.unwrapped.set_goal(goal_xy=(-10, -10))  # type: ignore
    env.unwrapped.set_xy((-10, -10))  # type: ignore
    env.unwrapped.render()
    maze_np = np.array(
        ImageOps.invert(
            Image.fromarray(env.unwrapped.maze_map.astype(np.uint8) * 255)  # type: ignore
        ).convert("RGBA")
    )
    background_color = "#303e57"
    background_color_arr = np.array(
        [int(background_color[i : i + 2], 16) for i in (1, 3, 5)] + [255]
    )
    maze_np[(maze_np == [0, 0, 0, 255]).all(axis=2)] = background_color_arr
    maze = Image.fromarray(maze_np)

    coordinates = (
        xy_to_ij(
            sample_to_coordinates(train_dataset.sample(DATASET_SAMPLE_SIZE)),
            env.unwrapped,
        )
        / env.unwrapped.maze_map.transpose().shape
    )
    fig = plt.figure(figsize=(3, 3))

    # Make plot work as an overlay
    plt.axis("off")
    plt.margins(0, 0)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_position([0, 0, 1, 1])
    plt.gca().invert_yaxis()

    sns.kdeplot(
        x=coordinates[:, 0],
        y=coordinates[:, 1],
        levels=10,
        bw_adjust=0.5,
        fill=True,
        cmap=cmocean.cm.ice_r,
        gridsize=400,
    )

    # Overlay to pillow image
    buf = io.BytesIO()
    fig.savefig(
        buf, dpi=600, format="png", bbox_inches="tight", transparent=True, pad_inches=0
    )
    plt.close(fig)
    buf.seek(0)
    overlay = Image.open(buf).convert("RGBA")

    fig_on_maze(overlay, maze).save(output_folder / f"heatmap_{dataset}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", type=str, required=True)
    parser.add_argument("--no_multiprocessing", action="store_true")
    args = parser.parse_args()

    output_folder = Path("plots") / "visualizations" / "datasets"
    output_folder.mkdir(exist_ok=True, parents=True)

    if not args.no_multiprocessing:
        try:
            thread_count = int(os.environ["SLURM_CPUS_ON_NODE"]) // 2
        except Exception as _:
            thread_count = multiprocessing.cpu_count() // 2
        with multiprocessing.get_context("spawn").Pool(thread_count) as pool:
            pool.map(
                partial(plot_dataset_heatmap, output_folder=output_folder),
                args.datasets,
            )
    else:
        for dataset in args.datasets:
            plot_dataset_heatmap(dataset, output_folder)
