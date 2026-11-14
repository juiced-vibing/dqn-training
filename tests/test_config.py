from pydantic import ValidationError
import pytest

from config import TrainingConfig


@pytest.mark.parametrize(
    "fields",
    [
        {"total_steps": 0},
        {"learning_rate": 0.0},
        {"gamma": 1.5},
        {"epsilon_end": -0.1},
        {"exploration_fraction": 0.0},
        {"batch_size": 0},
    ],
)
def test_rejects_values_training_cannot_run_with(fields: dict[str, float]):
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(fields)


def test_exploration_steps_follow_the_fraction():
    config = TrainingConfig(total_steps=1_000, exploration_fraction=0.25)

    assert config.exploration_steps == config.total_steps // 4
