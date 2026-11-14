from typing import TYPE_CHECKING
from typing import Self

from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
from msgspec import Struct

from errors import UnsupportedEnvironmentError


if TYPE_CHECKING:
    import gymnasium as gym


class EnvSpec(Struct, frozen=True):
    env_id: str
    obs_size: int
    action_count: int

    @classmethod
    def of(cls, env: gym.Env, env_id: str) -> Self:
        # a plain mlp dqn needs a flat box of observations and a discrete set of actions
        match env.observation_space, env.action_space:
            case Box(shape=(obs_size,)), Discrete(n=action_count):
                return cls(env_id, obs_size, int(action_count))
            case _:
                raise UnsupportedEnvironmentError(env_id)
