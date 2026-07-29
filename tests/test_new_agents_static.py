"""Static shape/finiteness tests for the FQL agent, n-step targets, and
rejection sampling. No environment or dataset download required.

Run: python -m pytest tests/test_new_agents_static.py -q  (or python tests/...)
"""

import numpy as np

import jax
import jax.numpy as jnp

from gcrl_landscapes.agents import FQLAgent, NStepGCIQLAgent, NStepGCIVLAgent
from gcrl_landscapes.agents import fql as fql_module
from gcrl_landscapes.configurations import generate_configurations
from gcrl_landscapes.evaluate import RejectionSamplingAgent
from gcrl_landscapes.training import (
    ACTOR_MODULE_PREFIXES,
    restore_frozen_params,
    snapshot_frozen_params,
)
from gcrl_landscapes.util.datasets import NStepGCDataset

import ogbench.impls.agents.gcbc as gcbc_module
import ogbench.impls.agents.gciql as gciql_module
import ogbench.impls.agents.gcivl as gcivl_module
from ogbench.impls.agents.gcbc import GCBCAgent
from ogbench.impls.utils.datasets import Dataset

OBS_DIM = 11
ACT_DIM = 3
BATCH = 16


def make_batch(rng):
    obs = rng.normal(size=(BATCH, OBS_DIM)).astype(np.float32)
    return {
        "observations": obs,
        "next_observations": rng.normal(size=(BATCH, OBS_DIM)).astype(np.float32),
        "value_goals": rng.normal(size=(BATCH, OBS_DIM)).astype(np.float32),
        "actor_goals": rng.normal(size=(BATCH, OBS_DIM)).astype(np.float32),
        "actions": rng.uniform(-1, 1, size=(BATCH, ACT_DIM)).astype(np.float32),
        "rewards": -rng.uniform(size=(BATCH,)).astype(np.float32),
        "masks": np.ones((BATCH,), dtype=np.float32),
    }


def assert_finite(info):
    for k, v in info.items():
        assert np.isfinite(np.asarray(v)).all(), f"non-finite metric {k}: {v}"


def test_fql_create_update_sample():
    rng = np.random.default_rng(0)
    config = fql_module.get_config().to_dict()
    ex_obs = rng.normal(size=(BATCH, OBS_DIM)).astype(np.float32)
    ex_act = rng.uniform(-1, 1, size=(BATCH, ACT_DIM)).astype(np.float32)
    agent = FQLAgent.create(0, ex_obs, ex_act, config)

    batch = make_batch(rng)
    agent, info = agent.update(batch)
    assert_finite(info)
    for key in (
        "actor/bc_flow_loss",
        "actor/distill_loss",
        "actor/q_loss",
        "critic/critic_loss",
        "value/value_loss",
    ):
        assert key in info, f"missing metric {key}"

    single_obs = ex_obs[0]
    action = agent.sample_actions(
        single_obs, goals=ex_obs[1], seed=jax.random.PRNGKey(0)
    )
    assert action.shape == (ACT_DIM,)
    assert jnp.all(jnp.abs(action) <= 1.0)
    print("FQL create/update/sample: OK")
    return agent


def test_nstep_agents():
    rng = np.random.default_rng(1)
    ex_obs = rng.normal(size=(BATCH, OBS_DIM)).astype(np.float32)
    ex_act = rng.uniform(-1, 1, size=(BATCH, ACT_DIM)).astype(np.float32)

    for module, agent_cls in (
        (gciql_module, NStepGCIQLAgent),
        (gcivl_module, NStepGCIVLAgent),
    ):
        config = module.get_config().to_dict()
        config["actor_loss"] = "awr"
        agent = agent_cls.create(0, ex_obs, ex_act, config)

        batch = make_batch(rng)
        agent, info = agent.update(batch)  # 1-step fallback path
        assert_finite(info)

        batch_n = make_batch(rng)
        batch_n["discounts"] = np.full((BATCH,), 0.99**3, dtype=np.float32)
        agent, info = agent.update(batch_n)  # n-step path
        assert_finite(info)
    print("n-step agents (fallback + discounts paths): OK")


def test_nstep_dataset_math():
    # Two trajectories: indices 0..7 (terminal at 7) and 8..15 (terminal at 15).
    T = 16
    gamma = 0.9
    n = 3
    obs = np.arange(T, dtype=np.float32)[:, None] * np.ones(
        (1, OBS_DIM), dtype=np.float32
    )
    terminals = np.zeros(T, dtype=np.float32)
    terminals[7] = 1
    terminals[15] = 1
    raw = dict(
        observations=obs,
        next_observations=np.concatenate([obs[1:], obs[-1:]], axis=0),
        actions=np.zeros((T, ACT_DIM), dtype=np.float32),
        terminals=terminals,
    )
    config = dict(
        discount=gamma,
        n_step=n,
        value_p_curgoal=0.0,
        value_p_trajgoal=1.0,
        value_p_randomgoal=0.0,
        value_geom_sample=False,
        actor_p_curgoal=0.0,
        actor_p_trajgoal=1.0,
        actor_p_randomgoal=0.0,
        actor_geom_sample=False,
        gc_negative=True,
        p_aug=None,
        frame_stack=None,
    )
    ds = NStepGCDataset(Dataset.create(**raw), config)

    idxs = np.array([0, 1, 5, 6])
    goal_idxs = np.array(
        [2, 1, 10, 12]
    )  # success at k=2; success at k=0; goals in the other trajectory
    ds.sample_goals = lambda *a, **k: (
        goal_idxs
    )  # fix goals for both value and actor goals

    batch = ds.sample(len(idxs), idxs=idxs)

    # Sample 0: t=0, goal at index 2 (k=2 < n): reward = -(1 + gamma), mask 0.
    np.testing.assert_allclose(batch["rewards"][0], -(1 + gamma), rtol=1e-6)
    assert batch["masks"][0] == 0.0
    # Sample 1: t=1, goal == t (k=0): reward 0, mask 0.
    np.testing.assert_allclose(batch["rewards"][1], 0.0, atol=1e-8)
    assert batch["masks"][1] == 0.0
    # Sample 2: t=5, no success, but only 2 steps to final (7): n_used=2.
    np.testing.assert_allclose(batch["rewards"][2], -(1 + gamma), rtol=1e-6)
    assert batch["masks"][2] == 1.0
    np.testing.assert_allclose(batch["discounts"][2], gamma**2, rtol=1e-6)
    np.testing.assert_allclose(batch["next_observations"][2][0], 7.0)  # obs index 5+2
    # Sample 3: t=6, no success, 1 step to final: n_used=1 (matches 1-step semantics).
    np.testing.assert_allclose(batch["rewards"][3], -1.0, rtol=1e-6)
    np.testing.assert_allclose(batch["discounts"][3], gamma, rtol=1e-6)
    np.testing.assert_allclose(batch["next_observations"][3][0], 7.0)
    print("NStepGCDataset reward/mask/discount math: OK")


def test_rejection_sampling():
    rng = np.random.default_rng(2)
    ex_obs = rng.normal(size=(BATCH, OBS_DIM)).astype(np.float32)
    ex_act = rng.uniform(-1, 1, size=(BATCH, ACT_DIM)).astype(np.float32)

    # GCIQL (Gaussian actor path).
    config = gciql_module.get_config().to_dict()
    config["actor_loss"] = "awr"
    gciql_agent = NStepGCIQLAgent.create(0, ex_obs, ex_act, config)
    wrapper = RejectionSamplingAgent(gciql_agent, num_samples=8)
    action = wrapper.sample_actions(
        ex_obs[0], goals=ex_obs[1], seed=jax.random.PRNGKey(0)
    )
    assert action.shape == (ACT_DIM,)
    assert jnp.all(jnp.abs(action) <= 1.0)

    # FQL (one-step flow path).
    fql_config = fql_module.get_config().to_dict()
    fql_agent = FQLAgent.create(0, ex_obs, ex_act, fql_config)
    wrapper = RejectionSamplingAgent(fql_agent, num_samples=8)
    action = wrapper.sample_actions(
        ex_obs[0], goals=ex_obs[1], seed=jax.random.PRNGKey(0)
    )
    assert action.shape == (ACT_DIM,)
    assert jnp.all(jnp.abs(action) <= 1.0)
    print("Rejection sampling (GCIQL + FQL): OK")


def test_rejection_sampling_with_separate_proposal():
    """SfBC-style extraction: candidates from a BC policy, scores from the critic.

    The point of the A6 arm is that no AWR-trained actor participates in action
    selection, so this asserts the executed action is the arg-max-min-Q element
    of the *proposal's* candidate set, recomputed independently.
    """
    rng = np.random.default_rng(5)
    ex_obs = rng.normal(size=(BATCH, OBS_DIM)).astype(np.float32)
    ex_act = rng.uniform(-1, 1, size=(BATCH, ACT_DIM)).astype(np.float32)

    critic_config = gciql_module.get_config().to_dict()
    critic_config["actor_loss"] = "awr"
    critic_agent = NStepGCIQLAgent.create(0, ex_obs, ex_act, critic_config)

    # A GCBC policy is the literature's pi_beta; seeded differently so its
    # actor cannot coincide with the critic agent's own actor.
    proposal_agent = GCBCAgent.create(7, ex_obs, ex_act, gcbc_module.get_config().to_dict())

    n = 8
    wrapper = RejectionSamplingAgent(critic_agent, num_samples=n, proposal_agent=proposal_agent)
    obs, goal, key = ex_obs[0], ex_obs[1], jax.random.PRNGKey(3)
    action = wrapper.sample_actions(obs, goals=goal, seed=key)

    assert action.shape == (ACT_DIM,)
    assert jnp.all(jnp.abs(action) <= 1.0)

    # Recompute the candidate set from the PROPOSAL and the scores from the
    # CRITIC, exactly as the wrapper should have.
    obs_r = jnp.repeat(jnp.asarray(obs)[None], n, axis=0)
    goals_r = jnp.repeat(jnp.asarray(goal)[None], n, axis=0)
    dist = proposal_agent.network.select("actor")(obs_r, goals_r, temperature=1.0)
    candidates = jnp.clip(dist.sample(seed=key), -1, 1)
    candidates = candidates.at[0].set(jnp.clip(dist.mode()[0], -1, 1))
    q1, q2 = critic_agent.network.select("critic")(obs_r, goals_r, candidates)
    expected = candidates[jnp.argmax(jnp.minimum(q1, q2))]
    np.testing.assert_allclose(np.asarray(action), np.asarray(expected), rtol=1e-6)

    # And it must differ from what the critic agent's own actor would have
    # proposed — otherwise the "AWR-free" claim would be untestable here.
    own = RejectionSamplingAgent(critic_agent, num_samples=n).sample_actions(
        obs, goals=goal, seed=key
    )
    assert not np.allclose(np.asarray(action), np.asarray(own)), (
        "proposal agent was ignored: candidates still came from the critic's actor"
    )
    print("Rejection sampling with separate BC proposal (SfBC): OK")


def test_restore_frozen_params():
    """A4: after an update, every non-actor subtree is bitwise unchanged and the
    actor has moved. Without this, a 'frozen signal' sweep would silently be an
    ordinary sweep."""
    rng = np.random.default_rng(11)
    ex_obs = rng.normal(size=(BATCH, OBS_DIM)).astype(np.float32)
    ex_act = rng.uniform(-1, 1, size=(BATCH, ACT_DIM)).astype(np.float32)

    config = gciql_module.get_config().to_dict()
    config["actor_loss"] = "awr"
    agent = NStepGCIQLAgent.create(0, ex_obs, ex_act, config)

    frozen = snapshot_frozen_params(agent)
    assert frozen, "no non-actor subtrees found to freeze"
    assert not any(k.startswith(ACTOR_MODULE_PREFIXES) for k in frozen), (
        "actor modules must not be frozen"
    )
    assert any("critic" in k or "value" in k for k in frozen), (
        f"expected critic/value subtrees among {sorted(frozen)}"
    )

    before = jax.tree_util.tree_map(np.asarray, agent.network.params)
    updated, _ = agent.update(make_batch(rng))
    restored = restore_frozen_params(updated, frozen)
    after = jax.tree_util.tree_map(np.asarray, restored.network.params)

    for key in before:
        leaves_before = jax.tree_util.tree_leaves(before[key])
        leaves_after = jax.tree_util.tree_leaves(after[key])
        moved = any(
            not np.array_equal(b, a) for b, a in zip(leaves_before, leaves_after)
        )
        if key.startswith(ACTOR_MODULE_PREFIXES):
            assert moved, f"actor subtree {key} did not change — nothing was trained"
        else:
            assert not moved, f"frozen subtree {key} changed despite freeze_value"
    print("restore_frozen_params (critic/value pinned, actor moves): OK")


def test_generate_configurations_threads_n_step_and_rejection_sampling():
    configs = generate_configurations(
        n=2,
        agent="GCIQL",
        hyperparameters={"lr", "alpha"},
        env="antmaze-medium-navigate-v0",
        n_step=3,
    )
    assert all(config["n_step"] == 3 for config in configs)

    configs = generate_configurations(
        n=2,
        agent="GCIQL",
        hyperparameters={"lr", "alpha"},
        env="antmaze-medium-navigate-v0",
        rejection_sampling_n=8,
    )
    assert all(config["rejection_sampling_n"] == 8 for config in configs)
    print("generate_configurations n_step/rejection_sampling_n threading: OK")


if __name__ == "__main__":
    test_fql_create_update_sample()
    test_nstep_agents()
    test_nstep_dataset_math()
    test_rejection_sampling()
    test_generate_configurations_threads_n_step_and_rejection_sampling()
    print("all static tests passed")
