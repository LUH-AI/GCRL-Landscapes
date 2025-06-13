import ogbench

import matplotlib
import numpy as np
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageOps

FIGSIZE = (10, 10)


def to_trajectories(
    dataset: dict[str, np.ndarray], batch_size: int | None = None, replace: bool = False
) -> list[dict[str, np.ndarray]]:
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


def visualize_trajs(trajs: list[dict[str, np.ndarray]]) -> Image.Image:
    """Visualize x-y trajectories in locomotion environments.

    Adapted from OGBench (Park et al. 2025).
    """

    def xy_to_ij(
        xy: np.ndarray,
        x_offset: float = 4.0,
        y_offset: float = 4.0,
        maze_unit: float = 4.0,
    ) -> np.ndarray:
        # See ogbench MazeEnv
        i = (xy[:, 0] + x_offset) / maze_unit
        j = (xy[:, 1] + y_offset) / maze_unit
        return np.column_stack((i, j))

    matplotlib.use("Agg")

    fig = figure.Figure(tight_layout=True, figsize=FIGSIZE)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot()

    max_xy = 0.0
    for traj in trajs:
        xy = xy_to_ij(traj["observations"][:, :2])
        # direction = np.array([info["direction"] for info in traj["info"]])
        # color = get_2d_colors(direction, [-1, -1], [1, 1])
        for i in range(len(xy) - 1):
            ax.plot(xy[i : i + 2, 0], xy[i : i + 2, 1], linewidth=0.7)
        max_xy = max(max_xy, np.abs(xy).max() * 1.2)

    plot_axis = [-max_xy, max_xy, -max_xy, max_xy]
    plot_axis = [0, 7, 0, 7]
    ax.axis(plot_axis)
    ax.set_aspect("equal")

    fig.tight_layout()
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())

    rgba[:, :, 3] = (255 * (rgba[:, :, :3] != 255).any(axis=2)).astype(np.uint8)

    foreground = ImageOps.flip(Image.fromarray(rgba).resize((1000, 1000)))
    foreground.save("test.png")

    return foreground


env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
    "antmaze-medium-navigate-v0",
    height=FIGSIZE[0] * 100,
    width=FIGSIZE[1] * 100,
)  # type: ignore

# move goal and ant out of picture
env.unwrapped.set_goal(goal_xy=(-10, -10))
env.unwrapped.set_xy((-10, -10))


all_trajectories = to_trajectories(train_dataset, batch_size=3)
all_xy = np.concatenate(
    [traj["observations"][:, :2] for traj in all_trajectories]
)  # .transpose(2, 0, 1).reshape(2, -1).transpose()


visualized_env = env.unwrapped.render()
visualized_trajs = visualize_trajs(all_trajectories)

background = ImageOps.invert(
    Image.fromarray(env.unwrapped.maze_map.astype(np.uint8) * 255).resize(
        (1000, 1000), resample=Image.Resampling.BOX
    )
)


background.paste(visualized_trajs, (0, 0), visualized_trajs)
background.save("traj_visualization.png")
