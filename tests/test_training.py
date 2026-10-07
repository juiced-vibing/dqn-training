from dqn.config import TrainingConfig
from dqn.training import run_training
from dqn.utils.metrics import EvaluationMetrics
from dqn.utils.metrics import Metrics


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
