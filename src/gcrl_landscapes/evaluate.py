from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from ogbench.impls.utils.evaluation import evaluate
from ogbench.locomaze.ant import AntEnv
from ogbench.locomaze.humanoid import HumanoidEnv
from ogbench.locomaze.point import PointEnv
from ogbench.manipspace.envs.cube_env import CubeEnv
from ogbench.manipspace.envs.scene_env import SceneEnv
from ogbench.powderworld.powderworld_env import PowderworldEnv


class RejectionSamplingAgent:
    """Evaluation-time wrapper: sample candidate actions, execute the argmax of min-Q.

    Requires an agent whose 'critic' module returns a double Q over
    (observations, goals, actions); this holds for GCIQL, CRL, and FQL.
    Gaussian actors are sampled at temperature 1 for diversity (the
    zero-temperature mode is always included as one candidate), so the wrapper
    never does worse than greedy decoding under a perfect critic. FQL's
    one-step policy is sampled with independent input noises. The training-time
    agent is untouched; this only changes action selection at evaluation.
    """

    def __init__(self, agent, num_samples: int):
        assert num_samples > 1, "rejection sampling needs at least 2 candidates"
        self.agent = agent
        self.num_samples = int(num_samples)
        self.config = agent.config

    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        agent = self.agent
        if goals is None:
            return agent.sample_actions(
                observations, goals=goals, seed=seed, temperature=temperature
            )
        n = self.num_samples
        obs_r = jnp.repeat(jnp.asarray(observations)[None], n, axis=0)
        goals_r = jnp.repeat(jnp.asarray(goals)[None], n, axis=0)
        if agent.config["agent_name"] == "fql":
            noise = jax.random.normal(seed, (n, agent.config["action_dim"]))
            candidates = jnp.clip(
                agent.network.select("actor_onestep")(obs_r, goals_r, noise), -1, 1
            )
        else:
            dist = agent.network.select("actor")(obs_r, goals_r, temperature=1.0)
            candidates = jnp.clip(dist.sample(seed=seed), -1, 1)
            candidates = candidates.at[0].set(jnp.clip(dist.mode()[0], -1, 1))
        q1, q2 = agent.network.select("critic")(obs_r, goals_r, candidates)
        q = jnp.minimum(q1, q2)
        return candidates[jnp.argmax(q)]


def evaluate_wrapper(
    agent,
    env: gym.Env,
    eval_episodes: int,
    config: ConfigDict,
    metrics: list[str] = ["success"],
) -> tuple[list, dict[str, np.floating], list, list]:
    n_rejection = int(config.get("rejection_sampling_n", 0) or 0)
    if n_rejection > 1:
        assert config["agent_name"] in ("gciql", "crl", "fql"), (
            f"rejection sampling needs a double-Q critic; got {config['agent_name']}"
        )
        agent = RejectionSamplingAgent(agent, n_rejection)
    task_infos = (
        env.unwrapped.task_infos
        if hasattr(env.unwrapped, "task_infos")
        else env.task_infos
    )
    results = [
        evaluate(
            agent=agent,
            env=env,
            task_id=task_id,
            config=config,
            num_eval_episodes=eval_episodes,
        )
        for task_id in range(1, len(task_infos) + 1)
    ]
    eval_info, trajs, renders = zip(*results)
    eval_metrics = {
        metric: np.mean([info[metric] for info in eval_info]) for metric in metrics
    }

    length_of_trajectories = [
        [len(traj["reward"]) for traj in trajs_task] for trajs_task in trajs
    ]

    (
        goal_start_distances,
        goal_end_distances,
        real_goal_start_distances,
        real_goal_end_distances,
    ) = calc_goal_distances(trajs, env)

    for (
        eval_info_task,
        length_of_trajectories_task,
        goal_start_distances_task,
        goal_end_distances_task,
    ) in zip(
        eval_info, length_of_trajectories, goal_start_distances, goal_end_distances
    ):
        eval_info_task["length_of_trajectories"] = length_of_trajectories_task
        eval_info_task["goal_start_distances"] = goal_start_distances_task
        eval_info_task["goal_end_distances"] = goal_end_distances_task

    if real_goal_start_distances is not None and real_goal_end_distances is not None:
        for eval_info_task, real_start_task, real_end_task in zip(
            eval_info, real_goal_start_distances, real_goal_end_distances
        ):
            eval_info_task["real_goal_start_distances"] = real_start_task
            eval_info_task["real_goal_end_distances"] = real_end_task
    return eval_info, eval_metrics, trajs, renders


def compute_maze_distance(
    env: gym.Env, start_xy: np.ndarray, goal_xy: np.ndarray
) -> float:
    """Return shortest-path maze distance (continuous units) between two xy positions.

    Uses BFS on the discrete maze grid via env.unwrapped.get_oracle_subgoal().
    Returns float('inf') if start is unreachable from goal.
    Only valid for locomaze environments (AntEnv, HumanoidEnv, PointEnv).
    """
    unwrapped: Any = env.unwrapped
    _, bfs_map = unwrapped.get_oracle_subgoal(start_xy, goal_xy)
    start_ij = unwrapped.xy_to_ij(start_xy)
    dist_cells = bfs_map[start_ij[0], start_ij[1]]
    if dist_cells < 0:
        return float("inf")
    return float(dist_cells * unwrapped._maze_unit)


def calc_goal_distances(
    trajectories, env
) -> tuple[
    list[list[float]],
    list[list[float]],
    list[list[float]] | None,
    list[list[float]] | None,
]:
    real_goal_start_distances = None
    real_goal_end_distances = None
    if isinstance(env.unwrapped, (HumanoidEnv, AntEnv, PointEnv)):
        goal_start_distances = [
            [
                float(
                    np.linalg.norm(traj["info"][0]["xy_goal"] - traj["info"][0]["xy"])
                )
                for traj in trajs_task
            ]
            for trajs_task in trajectories
        ]
        goal_end_distances = [
            [
                float(
                    np.linalg.norm(traj["info"][-1]["xy_goal"] - traj["info"][-1]["xy"])
                )
                for traj in trajs_task
            ]
            for trajs_task in trajectories
        ]
        real_goal_start_distances = [
            [
                compute_maze_distance(
                    env, traj["info"][0]["xy"], traj["info"][0]["xy_goal"]
                )
                for traj in trajs_task
            ]
            for trajs_task in trajectories
        ]
        real_goal_end_distances = [
            [
                compute_maze_distance(
                    env, traj["info"][-1]["xy"], traj["info"][-1]["xy_goal"]
                )
                for traj in trajs_task
            ]
            for trajs_task in trajectories
        ]
    elif isinstance(env.unwrapped, CubeEnv):

        def info_to_distance(info: dict) -> float:
            goal_distances = [
                float(info[f"privileged/block_{i}_dist"])
                for i in range(env.unwrapped._num_cubes)
            ]
            return sum(goal_distances)

        goal_start_distances = [
            [info_to_distance(traj["info"][0]) for traj in trajs_task]
            for trajs_task in trajectories
        ]
        goal_end_distances = [
            [info_to_distance(traj["info"][-1]) for traj in trajs_task]
            for trajs_task in trajectories
        ]
    elif isinstance(env.unwrapped, PowderworldEnv):
        goal_start_distances = [
            [float(traj["info"][0]["error"]) for traj in trajs_task]
            for trajs_task in trajectories
        ]
        goal_end_distances = [
            [float(traj["info"][-1]["error"]) for traj in trajs_task]
            for trajs_task in trajectories
        ]
    elif isinstance(env.unwrapped, SceneEnv):
        n_cubes = env.unwrapped._num_cubes
        n_buttons = env.unwrapped._num_buttons
        xyz_center = np.array([0.425, 0.0, 0.0])
        robot_dims = 19
        btn_start = robot_dims + n_cubes * 9

        def decode_goal(goal_list: list) -> dict:
            g = np.array(goal_list)
            return {
                "block_pos": [
                    g[robot_dims + i * 9 : robot_dims + i * 9 + 3] / 10.0 + xyz_center
                    for i in range(n_cubes)
                ],
                "button_states": [
                    int(np.argmax(g[btn_start + j * 4 : btn_start + j * 4 + 2]))
                    for j in range(n_buttons)
                ],
                "drawer_pos": float(g[btn_start + n_buttons * 4] / 18.0),
                "window_pos": float(g[btn_start + n_buttons * 4 + 2] / 15.0),
            }

        def scene_distance(info: dict, goal: dict) -> float:
            cube_dist = sum(
                float(
                    np.linalg.norm(
                        info[f"privileged/block_{i}_pos"] - goal["block_pos"][i]
                    )
                )
                for i in range(n_cubes)
            )
            drawer_dist = float(
                abs(info["privileged/drawer_pos"][0] - goal["drawer_pos"])
            )
            window_dist = float(
                abs(info["privileged/window_pos"][0] - goal["window_pos"])
            )
            button_dist = float(
                sum(
                    int(
                        info[f"privileged/button_{i}_state"] != goal["button_states"][i]
                    )
                    for i in range(n_buttons)
                )
            )
            return cube_dist + drawer_dist + window_dist + button_dist

        goal_start_distances = [
            [
                scene_distance(traj["info"][0], decode_goal(traj["info"][0]["goal"]))
                for traj in trajs_task
            ]
            for trajs_task in trajectories
        ]
        goal_end_distances = [
            [
                scene_distance(traj["info"][-1], decode_goal(traj["info"][0]["goal"]))
                for traj in trajs_task
            ]
            for trajs_task in trajectories
        ]
    else:
        raise NotImplementedError(
            f"{type(env.unwrapped)} not supported for goal distance calculation"
        )

    return (
        goal_start_distances,
        goal_end_distances,
        real_goal_start_distances,
        real_goal_end_distances,
    )
