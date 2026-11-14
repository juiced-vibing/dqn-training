class DQNError(Exception):
    pass


class UnsupportedEnvironmentError(DQNError):
    def __init__(self, env_id: str) -> None:
        super().__init__(f"unsupported environment {env_id}: dqn needs a flat box observation space and a discrete action space")
