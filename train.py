from collections import deque

import gymnasium as gym
import imageio
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)  # extra threads cost more in dispatch than they save

OBS_DIM, HIDDEN_DIM, OUTPUT_DIM = 4, 64, 2


class Trainer:
    def __init__(self) -> None:
        self.__model = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=5e-3)

        self.__total_episodes = 10000
        self.__gamma = 0.99

    def _init_buffer(self) -> deque[torch.Tensor]:
        # 3 frames of 400 x 600 grayscale
        buf = deque(maxlen=3)
        buf.extend([torch.zeros((400, 600)) for _ in range(3)])
        return buf

    def step(
        self,
        batch: list[tuple[np.ndarray, np.ndarray, int, float, bool]],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        states, next_states, actions, rewards, dones = zip(*batch)

        states = torch.as_tensor(np.array(states))
        next_states = torch.as_tensor(np.array(next_states))
        actions = torch.as_tensor(actions)
        rewards = torch.as_tensor(rewards)
        dones = torch.as_tensor(dones, dtype=torch.float32)

        with torch.no_grad():
            target_q = (
                rewards
                + self.__gamma * (1 - dones) * self.__model(next_states).max(1).values
            )

        pred_q = self.__model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = nn.MSELoss()(pred_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())

    def _render_states(self, env: gym.Env, states: list[np.ndarray]) -> list:
        frames = []
        for state in states:
            env.unwrapped.state = np.asarray(state, dtype=np.float64)
            frames.append(env.render())
        return frames

    def train(self):
        batch = []
        env = gym.make("CartPole-v1", render_mode="rgb_array")

        best_frames = []
        best_reward = float("-inf")

        for episode in range(self.__total_episodes):
            done = False
            episode_reward = 0.0

            state, _ = env.reset(seed=0)
            states = [state]

            while not done:
                with torch.no_grad():
                    action = self.__model(torch.tensor(state)).argmax().item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                batch.append((state, next_state, action, reward, done))
                states.append(next_state)

                episode_reward += reward
                state = next_state

                if len(batch) >= 64:
                    self.step(batch, self.__optimizer)
                    batch = []

            if episode_reward >= best_reward:
                best_reward = episode_reward
                best_frames = self._render_states(env, states)
                print(f"Episode {episode} has best reward: {best_reward}")

        env.close()
        return best_frames


trainer = Trainer()
frames = trainer.train()
imageio.mimsave("cartpole.gif", frames, fps=30)
