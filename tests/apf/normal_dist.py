import jax
import jax.random as jrng
import numpy as np

seed = np.random.randint(0, 256)
rng = jrng.PRNGKey(seed)
map_weed = jax.random.normal(shape=(10000,), key=rng)
print(map_weed.max())
print(map_weed.min())
print(map_weed.mean())
