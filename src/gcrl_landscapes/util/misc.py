import jax
from typing import Callable, Any
from time import sleep
import numpy as np


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
    error = None
    while save_tries < MAX_SAVE_TRIES:
        try:
            return save_call()
        except Exception as e:
            error = e
            save_tries += 1
            wait_time = int(
                np.random.randint(low=sleep_interval[0], high=sleep_interval[1])
            )
            print(f"Failed to save, sleeping {wait_time}s")
            sleep(wait_time)
    print("Error saving")
    print(error)
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
