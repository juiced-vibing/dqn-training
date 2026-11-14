from config import TrainingConfig
from training import run_training
from utils.metrics import EvaluationMetrics
from utils.metrics import Metrics


def test_evaluates_on_final_step_even_off_interval():
    config = TrainingConfig(
        total_steps=600,
        learning_starts=100,
        buf_cap=500,
        batch_size=32,
        eval_interval=1_000,
        eval_episodes=2,
    )
    emitted: list[Metrics] = []

    run_training(config, on_metrics=emitted.append)

    evaluations = [metrics for metrics in emitted if isinstance(metrics, EvaluationMetrics)]
    assert [evaluation.step for evaluation in evaluations] == [600]


def test_a_one_step_run_still_has_an_exploration_schedule():
    # half of one step rounds down to no exploration steps at all, which used to divide by zero
    config = TrainingConfig(total_steps=1, eval_episodes=1)
    emitted: list[Metrics] = []

    run_training(config, on_metrics=emitted.append)

    assert config.exploration_steps == 1
    assert [type(metrics) for metrics in emitted] == [EvaluationMetrics]
