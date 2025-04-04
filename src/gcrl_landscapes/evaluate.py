import gymnasium as gym
from ogbench.impls.utils.evaluation import evaluate
import numpy as np
from ml_collections import ConfigDict
import warnings


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
    warnings.warn("evaluation is currently only supported for two-dimensional goals")
    goal_start_distances = [
        # [:2] -> two-dimensional
        [
            float(np.linalg.norm(traj["info"][0]["goal"][:2] - traj["info"][0]["xy"]))
            for traj in trajs_task
        ]
        for trajs_task in trajs
    ]
    goal_end_distances = [
        # [:2] -> two-dimensional
        [
            float(np.linalg.norm(traj["info"][-1]["goal"][:2] - traj["info"][-1]["xy"]))
            for traj in trajs_task
        ]
        for trajs_task in trajs
    ]
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
