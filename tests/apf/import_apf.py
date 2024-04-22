import jax
import jax.numpy as jnp
from jax import lax
from envx.utils.apf_jax import apf
# import cpu_apf

import jax.random as jrng

rng = jrng.PRNGKey(0)
# a = jax.random.uniform(shape=(6, 6), key=rng) <= 0.1
a = jnp.zeros([6, 6], dtype=jnp.bool_)
print(a)
b = apf(a)
jnp.set_printoptions(precision=2)
print(b)
print(0.8 ** b)

