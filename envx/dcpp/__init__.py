import gymnasium.envs.registration

gymnasium.envs.registration.register(
    id="LawnMowing-v2",
    entry_point="envx.dcpp.lawn_mowing:LawnMowingJaxEnv",
    vector_entry_point="envx.dcpp.lawn_mowing:LawnMowingJaxVectorEnv",
    max_episode_steps=1000,
)
