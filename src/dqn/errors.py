class DQNError(Exception):
    """Base for errors this package raises on purpose."""


class UnsupportedEnvironmentError(DQNError):
    """Environment spaces that a plain MLP DQN cannot handle."""

    def __init__(self, env_id: str) -> None:
        super().__init__(f"unsupported environment {env_id}: dqn needs a flat box observation space and a discrete action space")
