prevent_stiff = True
nvec = [4, 19]
v_max = 1
w_max = 1


def get_velocity(action):
    action_linear = action // nvec[1] + prevent_stiff
    linear_size = nvec[0] - 1 + prevent_stiff
    v_linear = v_max * action_linear / linear_size
    action_angular = action % nvec[1]
    angular_size = (nvec[1] - 1) // 2
    v_angular = w_max * (action_angular - angular_size) / angular_size
    return v_linear, v_angular


for i in range(nvec[0] * nvec[1]):
    print(f'{i:3d}: v{i // nvec[1]} w{i % nvec[1]:2d} -> {get_velocity(i)}')
