import gymnasium as gym
from ogbench.impls.utils.evaluation import evaluate
from ogbench.locomaze.ant import AntEnv
from ogbench.locomaze.humanoid import HumanoidEnv
from ogbench.manipspace.envs.cube_env import CubeEnv
from ogbench.powderworld.powderworld_env import PowderworldEnv
import numpy as np
from ml_collections import ConfigDict


def evaluate_wrapper(
    agent,
    env: gym.Env,
    eval_episodes: int,
    config: ConfigDict,
    metrics: list[str] = ["success"],
) -> tuple[list, dict[str, np.floating], list, list]:
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

    goal_start_distances, goal_end_distances = calc_goal_distances(trajs, env)

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
    return eval_info, eval_metrics, trajs, renders


def calc_goal_distances(
    trajectories, env
) -> tuple[list[list[float]], list[list[float]]]:
    if isinstance(env.unwrapped, (HumanoidEnv, AntEnv)):
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
    else:
        raise NotImplementedError(
            f"{type(env.unwrapped)} not supported for goal distance calculation"
        )

    return goal_start_distances, goal_end_distances
