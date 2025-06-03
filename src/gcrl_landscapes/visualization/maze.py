import matplotlib
import numpy as np
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


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


def visualize_trajs(trajs):
    """Visualize x-y trajectories in locomotion environments.

    It reads 'xy' and 'direction' from the 'info' field of the trajectories.
    Adapted from OGBench (Park et al. 2025).
    """
    matplotlib.use("Agg")

    fig = figure.Figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot()

    max_xy = 0.0
    for traj in trajs:
        xy = np.array([info["xy"] for info in traj["info"]])
        direction = np.array([info["direction"] for info in traj["info"]])
        color = get_2d_colors(direction, [-1, -1], [1, 1])
        for i in range(len(xy) - 1):
            ax.plot(xy[i : i + 2, 0], xy[i : i + 2, 1], color=color[i], linewidth=0.7)
        max_xy = max(max_xy, np.abs(xy).max() * 1.2)

    plot_axis = [-max_xy, max_xy, -max_xy, max_xy]
    ax.axis(plot_axis)
    ax.set_aspect("equal")

    fig.tight_layout()
    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return out_image
