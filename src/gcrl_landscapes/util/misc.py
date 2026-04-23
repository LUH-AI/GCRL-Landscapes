import jax
import optax
from typing import Callable, Any
from time import sleep
import numpy as np
import traceback


def jax_has_gpu():
    """Test if Jax is able to use a GPU

    Returns:
        True if Jax is able to use a GPU
    """
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices("gpu")[0])
        return True
    except:  # noqa: E722  # usually one should specify the error, here a catch all is enough
        return False


def retry_call(save_call: Callable) -> Any:
    """Try saving multiple times with random backoff.
    This is necessary due to a bug when running jobs on multiple partitions on slurm in parallel. It sometimes (undeterministically) leads to OSError 7: Argument list too long.

    Args:
        save_call: the function which does the saving call. Will be retried if it fails

    Returns:
        Return value of save_call or None
    """
    sleep_interval = [1, 60]
    MAX_SAVE_TRIES = 20
    save_tries = 0
    while save_tries < MAX_SAVE_TRIES:
        try:
            return save_call()
        except Exception:
            error_message = traceback.format_exc()
            print(error_message)
            save_tries += 1
            wait_time = int(
                np.random.randint(low=sleep_interval[0], high=sleep_interval[1])
            )
            print(f"Failed to save, sleeping {wait_time}s")
            sleep(wait_time)
    print("Error saving")
    return None


def get_feature_embedding(agent, batch):
    """Extract feature embeddings (phi_s, phi_g) from the agent's value network.

    Args:
        agent: The agent (QRL, CRL, or HIQL)
        batch: A batch of data containing 'observations' and 'value_goals'

    Returns:
        phi_s: State feature embeddings of shape (batch_size, latent_dim or hidden_dim)
        phi_g: Goal feature embeddings of shape (batch_size, latent_dim or hidden_dim)
    """
    # Call the value network with info=True
    result = agent.network.select('value')(
        batch['observations'],
        batch['value_goals'],
        info=True
    )

    if len(result) == 3:
        # QRL/CRL: returns (v, phi_s, phi_g)
        _, phi_s, phi_g = result
        return jax.numpy.concatenate((phi_s, phi_g), axis=-1)
    else:
        # HIQL: GCValue returns (v, features) - joint representation
        v_ensemble, features_ensemble = result
        # HIQL uses an ensemble. Select correct features
        return features_ensemble


def crl_contrastive_loss_individual(agent, batch, index, grad_params, module_name='critic'):
    """Compute the contrastive value loss for the Q or V function."""
    batch_size = batch['observations'].shape[0]

    if module_name == 'critic':
        actions = batch['actions']
    else:
        actions = None
    v, phi, psi = agent.network.select(module_name)(
        batch['observations'],
        batch['value_goals'],
        actions=actions,
        info=True,
        params=grad_params,
    )
    if len(phi.shape) == 2:  # Non-ensemble.
        phi = phi[None, ...]
        psi = psi[None, ...]
    logits = jax.numpy.einsum('eik,ejk->ije', phi, psi) / jax.numpy.sqrt(phi.shape[-1])
    # logits.shape is (B, B, e) with one term for positive pair and (B - 1) terms for negative pairs in each row.
    I = jax.numpy.eye(batch_size)
    contrastive_loss = jax.vmap(
        lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
        in_axes=-1,
        out_axes=-1,
    )(logits)
    contrastive_loss = jax.numpy.mean(contrastive_loss, axis=range(1, len(contrastive_loss.shape)))[index]

    # Compute additional statistics.
    v = jax.numpy.exp(v)
    logits = jax.numpy.mean(logits, axis=-1)
    logits_pos = jax.numpy.sum(logits * I) / jax.numpy.sum(I)
    logits_neg = jax.numpy.sum(logits * (1 - I)) / jax.numpy.sum(1 - I)

    return contrastive_loss, {
        'contrastive_loss': contrastive_loss,
        'v_mean': v.mean(),
        'v_max': v.max(),
        'v_min': v.min(),
        'binary_accuracy': jax.numpy.mean((logits > 0) == I),
        'logits_pos': logits_pos,
        'logits_neg': logits_neg,
        'logits': logits.mean(),
    }

def crl_actor_loss_individual(agent, batch, index, grad_params, rng=None):
    """Compute the actor loss (AWR or DDPG+BC)."""
    batch = jax.tree.map(lambda leaf: leaf[[index], ...], batch)
    if agent.config['actor_loss'] == 'awr':
        # AWR loss.
        v = agent.network.select('value')(batch['observations'], batch['actor_goals'])
        q1, q2 = agent.network.select('critic')(batch['observations'], batch['actor_goals'], batch['actions'])
        q = jax.numpy.minimum(q1, q2)
        adv = q - v

        exp_a = jax.numpy.exp(adv * agent.config['alpha'])
        exp_a = jax.numpy.minimum(exp_a, 100.0)

        dist = agent.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss.mean(),
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not agent.config['discrete']:
            actor_info.update(
                {
                    'mse': jax.numpy.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jax.numpy.mean(dist.scale_diag),
                }
            )

        return actor_loss, actor_info
    elif agent.config['actor_loss'] == 'ddpgbc':
        # DDPG+BC loss.
        assert not agent.config['discrete']

        dist = agent.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        if agent.config['const_std']:
            q_actions = jax.numpy.clip(dist.mode(), -1, 1)
        else:
            q_actions = jax.numpy.clip(dist.sample(seed=rng), -1, 1)
        q1, q2 = agent.network.select('critic')(batch['observations'], batch['actor_goals'], q_actions)
        q = jax.numpy.minimum(q1, q2)

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        q_loss = -q.mean() / jax.lax.stop_gradient(jax.numpy.abs(q).mean() + 1e-6)
        log_prob = dist.log_prob(batch['actions'])

        bc_loss = -(agent.config['alpha'] * log_prob).mean()

        actor_loss = q_loss + bc_loss

        return actor_loss, {
            'actor_loss': actor_loss.mean(),
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            'q_mean': q.mean(),
            'q_abs_mean': jax.numpy.abs(q).mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jax.numpy.mean((dist.mode() - batch['actions']) ** 2),
            'std': jax.numpy.mean(dist.scale_diag),
        }
    else:
        raise ValueError(f'Unsupported actor loss: {agent.config["actor_loss"]}')

@jax.jit
def crl_total_loss_individual(agent, batch, index, grad_params, rng=None):
    """Compute the total loss."""
    info = {}
    rng = rng if rng is not None else agent.rng

    critic_loss, critic_info = crl_contrastive_loss_individual(agent, batch, index, grad_params, 'critic')
    for k, v in critic_info.items():
        info[f'critic/{k}'] = v

    if agent.config['actor_loss'] == 'awr':
        value_loss, value_info = crl_contrastive_loss_individual(agent, batch, index, grad_params, 'value')
        for k, v in value_info.items():
            info[f'value/{k}'] = v
    else:
        value_loss = 0.0

    rng, actor_rng = jax.random.split(rng)
    actor_loss, actor_info = crl_actor_loss_individual(agent, batch, index, grad_params, actor_rng)
    for k, v in actor_info.items():
        info[f'actor/{k}'] = v

    loss = critic_loss + value_loss + actor_loss
    return loss, info


