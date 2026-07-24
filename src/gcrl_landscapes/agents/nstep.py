"""n-step variants of GCIQL and GCIVL.

These subclasses are drop-in replacements for the base agents. When the batch
contains a per-sample 'discounts' key (produced by NStepGCDataset for
n_step > 1), the TD backup uses batch['rewards'] as the n-step discounted
reward sum and batch['discounts'] = gamma^{n_used} as the bootstrap factor.
Without that key, behavior is identical to the base agents.
"""

import jax.numpy as jnp
from ogbench.impls.agents import GCIQLAgent, GCIVLAgent


class NStepGCIQLAgent(GCIQLAgent):
    """GCIQL with optional n-step TD backups."""

    def critic_loss(self, batch, grad_params):
        """Compute the IQL critic loss with an n-step-aware backup."""
        if "discounts" not in batch:
            return super().critic_loss(batch, grad_params)

        next_v = self.network.select("value")(
            batch["next_observations"], batch["value_goals"]
        )
        q = batch["rewards"] + batch["discounts"] * batch["masks"] * next_v

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


class NStepGCIVLAgent(GCIVLAgent):
    """GCIVL with optional n-step TD backups."""

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss with an n-step-aware backup."""
        if "discounts" not in batch:
            return super().value_loss(batch, grad_params)

        discounts = batch["discounts"]
        (next_v1_t, next_v2_t) = self.network.select("target_value")(
            batch["next_observations"], batch["value_goals"]
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch["rewards"] + discounts * batch["masks"] * next_v_t

        (v1_t, v2_t) = self.network.select("target_value")(
            batch["observations"], batch["value_goals"]
        )
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch["rewards"] + discounts * batch["masks"] * next_v1_t
        q2 = batch["rewards"] + discounts * batch["masks"] * next_v2_t
        (v1, v2) = self.network.select("value")(
            batch["observations"], batch["value_goals"], params=grad_params
        )
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config["expectile"]).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config["expectile"]).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }
