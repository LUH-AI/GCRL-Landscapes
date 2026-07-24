"""Goal-conditioned Flow Q-Learning (FQL) agent.

Critic/value: GCIQL's expectile-V + double-Q backup, verbatim.
Actor: Park-style FQL — a BC flow-matching vector field plus a one-step policy
distilled from the flow rollout, with a Q maximization term on the one-step
action. Goals are handled like every other OGBench GC agent (optional GCEncoder,
otherwise concatenation), so the agent drops into the existing
create()/update()/sample_actions() pipeline.
"""

import copy
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from ogbench.impls.utils.encoders import GCEncoder, encoder_modules
from ogbench.impls.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogbench.impls.utils.networks import MLP, GCValue


class GCActorVectorField(nn.Module):
    """Goal-conditioned actor vector field for flow matching.

    (observations, goals, actions[, times]) -> velocity. Used both as the BC
    flow field (with times) and as the one-step policy (noise -> action,
    without times).
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    gc_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, goals=None, actions=None, times=None):
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        inputs.append(actions)
        if times is not None:
            inputs.append(times)
        inputs = jnp.concatenate(inputs, axis=-1)
        return MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(inputs)


class FQLAgent(flax.struct.PyTreeNode):
    """Goal-conditioned FQL agent with a GCIQL critic."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IQL value loss."""
        q1, q2 = self.network.select("target_critic")(
            batch["observations"], batch["value_goals"], batch["actions"]
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select("value")(
            batch["observations"], batch["value_goals"], params=grad_params
        )
        value_loss = self.expectile_loss(q - v, q - v, self.config["expectile"]).mean()

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """Compute the IQL critic loss."""
        next_v = self.network.select("value")(
            batch["next_observations"], batch["value_goals"]
        )
        q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v

        q1, q2 = self.network.select("critic")(
            batch["observations"],
            batch["value_goals"],
            batch["actions"],
            params=grad_params,
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def compute_flow_actions(self, observations, goals, noise, grad_params=None):
        """Roll out the BC flow field with Euler steps to produce flow actions."""
        actions = noise
        for i in range(self.config["flow_steps"]):
            times = jnp.full((*actions.shape[:-1], 1), i / self.config["flow_steps"])
            vel = self.network.select("actor_bc_flow")(
                observations, goals, actions, times, params=grad_params
            )
            actions = actions + vel / self.config["flow_steps"]
        return jnp.clip(actions, -1.0, 1.0)

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the FQL actor loss: BC flow matching + distillation + Q."""
        observations = batch["observations"]
        goals = batch["actor_goals"]
        actions = batch["actions"]
        batch_shape = actions.shape[:-1]
        action_dim = actions.shape[-1]

        rng = rng if rng is not None else self.rng
        x0_rng, t_rng, noise_rng = jax.random.split(rng, 3)

        # BC flow-matching loss.
        x0 = jax.random.normal(x0_rng, (*batch_shape, action_dim))
        t = jax.random.uniform(t_rng, (*batch_shape, 1))
        xt = (1 - t) * x0 + t * actions
        vel = actions - x0
        pred = self.network.select("actor_bc_flow")(
            observations, goals, xt, t, params=grad_params
        )
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Distill the flow rollout into the one-step policy.
        noise = jax.random.normal(noise_rng, (*batch_shape, action_dim))
        flow_actions = jax.lax.stop_gradient(
            self.compute_flow_actions(
                observations, goals, noise, grad_params=grad_params
            )
        )
        onestep_actions = self.network.select("actor_onestep")(
            observations, goals, noise, params=grad_params
        )
        distill_loss = jnp.mean((onestep_actions - flow_actions) ** 2)

        # Q maximization on the one-step action.
        q_actions = jnp.clip(onestep_actions, -1, 1)
        q1, q2 = self.network.select("critic")(observations, goals, q_actions)
        q = jnp.minimum(q1, q2)
        q_loss = -q.mean()
        if self.config["normalize_q_loss"]:
            q_loss = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-6)) * q_loss

        actor_loss = bc_flow_loss + self.config["alpha"] * distill_loss + q_loss

        return actor_loss, {
            "actor_loss": actor_loss,
            "bc_flow_loss": bc_flow_loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "q_mean": q.mean(),
            "mse": jnp.mean((q_actions - actions) ** 2),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        noise = jax.random.normal(
            seed, (*observations.shape[:-1], self.config["action_dim"])
        )
        actions = self.network.select("actor_onestep")(observations, goals, noise)
        return jnp.clip(actions, -1, 1)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        assert not config["discrete"], "FQL supports continuous action spaces only."
        ex_goals = ex_observations
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["value"] = GCEncoder(concat_encoder=encoder_module())
            encoders["critic"] = GCEncoder(concat_encoder=encoder_module())
            encoders["actor_bc_flow"] = GCEncoder(concat_encoder=encoder_module())
            encoders["actor_onestep"] = GCEncoder(concat_encoder=encoder_module())

        # Define value, critic, and actor networks.
        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=False,
            gc_encoder=encoders.get("value"),
        )
        critic_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=True,
            gc_encoder=encoders.get("critic"),
        )
        actor_bc_flow_def = GCActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["layer_norm"],
            gc_encoder=encoders.get("actor_bc_flow"),
        )
        actor_onestep_def = GCActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["layer_norm"],
            gc_encoder=encoders.get("actor_onestep"),
        )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_goals, ex_actions),
            ),
            actor_bc_flow=(
                actor_bc_flow_def,
                (ex_observations, ex_goals, ex_actions, ex_times),
            ),
            actor_onestep=(actor_onestep_def, (ex_observations, ex_goals, ex_actions)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params["modules_target_critic"] = params["modules_critic"]

        config = dict(config)
        config["action_dim"] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name="fql",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            actor_loss="fql",  # Actor loss type (fixed; FQL has its own actor objective).
            alpha=10.0,  # Distillation weight.
            flow_steps=10,  # Number of Euler steps for the BC flow rollout.
            normalize_q_loss=True,  # Whether to normalize the Q loss by its absolute mean.
            const_std=True,  # Unused; kept for pipeline compatibility.
            discrete=False,  # Whether the action space is discrete (must be False).
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name.
            # Dataset hyperparameters.
            dataset_class="GCDataset",  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(
                int
            ),  # Number of frames to stack.
        )
    )
    return config
