import gymnasium.envs.registration

gymnasium.envs.registration.register(
    id="LawnMowing",
    entry_point="envx.cpp.lawn_mowing:LawnMowingJaxEnv",
    vector_entry_point="envx.cpp.lawn_mowing:LawnMowingJaxVectorEnv",
    max_episode_steps=1000,
)
