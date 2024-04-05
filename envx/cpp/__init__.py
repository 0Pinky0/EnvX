import gymnasium.envs.registration

gymnasium.envs.registration.register(
    id="LawnMowing",
    entry_point="envx.cpp.lawn_mowing:LawnMowingJaxEnv",
    vector_entry_point="envx.cpp.lawn_mowing:LawnMowingJaxVectorEnv",
    max_episode_steps=1000,
)

gymnasium.envs.registration.register(
    id="Farmland-v1",
    entry_point="envx.cpp.farmland_v1:FarmlandV1JaxEnv",
    vector_entry_point="envx.cpp.farmland_v1:FarmlandV1JaxVectorEnv",
    max_episode_steps=1500,
)

gymnasium.envs.registration.register(
    id="Farmland-v2",
    entry_point="envx.cpp.farmland_v2:FarmlandV2JaxEnv",
    vector_entry_point="envx.cpp.farmland_v2:FarmlandV2JaxVectorEnv",
    max_episode_steps=1000,
)
