from gymnasium.envs.phys2d import CartPoleJaxEnv, PendulumJaxEnv
from gymnasium.envs.tabular import CliffWalkingJaxEnv
from gymnasium.wrappers import HumanRendering

if __name__ == "__main__":
    """
    Temporary environment tester function.
    """

    env = HumanRendering(PendulumJaxEnv(render_mode="rgb_array"))

    obs, info = env.reset()
    print(obs, info)

    terminal = False
    while not terminal:
        action = int(input("Please input an action\n"))
        obs, reward, terminal, truncated, info = env.step(action)
        print(obs, reward, terminal, truncated, info)

    exit()
