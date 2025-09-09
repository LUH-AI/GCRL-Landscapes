import argparse
from gcrl_landscapes.util.datasets import create_env_and_dataset
from gcrl_landscapes.configurations import get_adapted_default_config
from .maze import xy_to_ij
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps
import io


DATASET_SAMPLE_SIZE = 10_000  # TODO: bigger
MAX_IMAGE_SIZE = 1000
SUPERSAMPLING_FACTOR = 10


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
    white_bg = Image.new("RGBA", background.size, (255, 255, 255, 255))
    background = background.copy()
    background.putalpha(background.split()[0].point(lambda p: 0 if p > 240 else 255))
    density_maze_w_alpha = Image.alpha_composite(density_plot, background)
    return Image.alpha_composite(white_bg, density_maze_w_alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", type=str, required=True)
    args = parser.parse_args()

    for dataset in args.datasets:
        print(f"Creating heatmap for {dataset}")
        # agent class and actor loss do not matter much here, as they influence parts of the dataset we do not use anyways
        env, train_dataset, val_dataset = create_env_and_dataset(
            dataset, "CRL", get_adapted_default_config("CRL", dataset, "awr")
        )

        # Create maze image
        # move goal and ant out of picture
        env.unwrapped.set_goal(goal_xy=(-10, -10))  # type: ignore
        env.unwrapped.set_xy((-10, -10))  # type: ignore
        visualized_env = env.unwrapped.render()  # type: ignore
        maze = ImageOps.contain(
            ImageOps.invert(
                Image.fromarray(env.unwrapped.maze_map.astype(np.uint8) * 255)  # type: ignore
            ),
            (
                MAX_IMAGE_SIZE * SUPERSAMPLING_FACTOR,
                MAX_IMAGE_SIZE * SUPERSAMPLING_FACTOR,
            ),
            method=Image.Resampling.BOX,
        ).convert("RGBA")

        coordinates = (
            xy_to_ij(
                sample_to_coordinates(train_dataset.sample(DATASET_SAMPLE_SIZE)),
                env.unwrapped,
            )
            / env.unwrapped.maze_map.transpose().shape
        )
        fig = plt.figure(figsize=(10, 10))

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
            bw_adjust=0.3,
            fill=True,
            cmap="plasma",
        )

        # Overlay to pillow image
        buf = io.BytesIO()
        fig.savefig(
            buf, format="png", bbox_inches="tight", transparent=True, pad_inches=0
        )
        plt.close(fig)
        buf.seek(0)
        overlay = Image.open(buf).convert("RGBA").resize(maze.size)

        fig_on_maze(overlay, maze).save(f"{dataset}.png")
