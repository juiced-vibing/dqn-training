from datetime import datetime
from pathlib import Path
from typing import Annotated

import imageio
from loguru import logger
import torch
import typer

from dqn.config import TrainingConfig
from dqn.consts import GIF_FRAMES_PER_SECOND
from dqn.env.evaluation import record_episode
from dqn.training import run_training
from dqn.utils.display import TrainingDisplay
from dqn.utils.display import console
from dqn.utils.log import configure_logging
from dqn.utils.metrics import Metrics
from dqn.utils.metrics import MetricsWriter


app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Train DQN agents on Gymnasium environments."""


def _create_run_directory(root: Path, config: TrainingConfig) -> Path:
    directory = root / f"{config.env_id}-seed{config.seed}-{datetime.now().astimezone():%Y%m%d-%H%M%S}"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(config.model_dump_json(indent=2))
    return directory


@app.command()
def train(
    total_steps: Annotated[int, typer.Option(min=1)] = 500_000,
    seed: int = 1,
    runs_directory: Path = Path("runs"),
) -> None:
    """Train an agent, then save its best weights, metrics and a gif of it playing."""
    torch.set_num_threads(1)  # extra threads cost more in dispatch than they save

    config = TrainingConfig(total_steps=total_steps, seed=seed)
    run_directory = _create_run_directory(runs_directory, config)
    configure_logging(run_directory / "training.log")
    logger.debug("starting run in {}", run_directory)

    with (
        MetricsWriter(run_directory / "metrics.jsonl") as writer,
        TrainingDisplay(config.total_steps) as display,
    ):

        def record_metrics(metrics: Metrics) -> None:
            writer.write(metrics)
            display.update(metrics)

        agent = run_training(config, on_metrics=record_metrics)

    torch.save(agent.copy_weights(), run_directory / "model.pt")

    frames = record_episode(agent, config.env_id)
    imageio.mimsave(run_directory / "episode.gif", [*frames], fps=GIF_FRAMES_PER_SECOND)

    logger.debug("saved model and gif")
    console.print(f"run saved to [cyan]{run_directory}[/]")
