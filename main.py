import gymnasium as gym
import numpy as np
import torch

# import imageio

env = gym.make("CartPole-v1", render_mode="rgb_array")
obs, info = env.reset(seed=0)
frames, total_r, done = [env.render()], 0.0, False


def to_gray(frames) -> torch.Tensor:
    return torch.from_numpy(np.stack(frames)).float() @ torch.tensor(
        [0.299, 0.587, 0.114]
    )


print(to_gray(frames).shape)


# while not done:
#     action = env.action_space.sample()
#     obs, r, terminated, truncated, info = env.step(action)
#     total_r += r
#     done = terminated or truncated
#     frames.append(env.render())
# env.close()

# print("return:", total_r, "frames:", len(frames))
# imageio.mimsave("cartpole.gif", frames, fps=30)
