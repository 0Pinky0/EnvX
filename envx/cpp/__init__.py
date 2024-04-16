import gymnasium.envs.registration
from envx.cpp.lawn_mowing.lawn_mowing import LawnMowingFunctional
from envx.cpp.farmland.farmland import FarmlandFunctional
from envx.cpp.pasture.pasture import PastureFunctional
from envx.cpp.pasture_toy.pasture_toy import PastureToyFunctional

gymnasium.envs.registration.register(
    id="LawnMowing",
    entry_point="envx.cpp.lawn_mowing:LawnMowingJaxEnv",
    vector_entry_point="envx.cpp.lawn_mowing:LawnMowingJaxVectorEnv",
    max_episode_steps=LawnMowingFunctional.max_timestep,
)

gymnasium.envs.registration.register(
    id="Farmland",
    entry_point="envx.cpp.farmland:FarmlandJaxEnv",
    vector_entry_point="envx.cpp.farmland:FarmlandJaxVectorEnv",
    max_episode_steps=FarmlandFunctional.max_timestep,
)

gymnasium.envs.registration.register(
    id="Pasture",
    entry_point="envx.cpp.pasture:PastureJaxEnv",
    vector_entry_point="envx.cpp.pasture:PastureJaxVectorEnv",
    max_episode_steps=PastureFunctional.max_timestep,
)

gymnasium.envs.registration.register(
    id="PastureToy",
    entry_point="envx.cpp.pasture_toy:PastureToyJaxEnv",
    vector_entry_point="envx.cpp.pasture_toy:PastureToyJaxVectorEnv",
    max_episode_steps=PastureToyFunctional.max_timestep,
)
