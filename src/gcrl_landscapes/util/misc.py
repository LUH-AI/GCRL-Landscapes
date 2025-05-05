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
