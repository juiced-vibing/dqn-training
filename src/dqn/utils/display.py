from typing import TYPE_CHECKING
from typing import Self

from rich.console import Console
from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import TextColumn
from rich.progress import TimeRemainingColumn
from rich.table import Table

from dqn.utils.metrics import EpisodeMetrics
from dqn.utils.metrics import EvaluationMetrics
from dqn.utils.metrics import Metrics


if TYPE_CHECKING:
    from types import TracebackType


console = Console()


class TrainingDisplay:
    """Live progress bar during training, evaluation table once it ends."""

    def __init__(self, total_steps: int) -> None:
        self.__progress = Progress(
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("return [cyan]{task.fields[episode_return]:>5.0f}[/]"),
            TextColumn("eval [green]{task.fields[evaluation_return]:>5.0f}[/]"),
            TextColumn("eps {task.fields[epsilon]:.2f}"),
            TimeRemainingColumn(),
            console=console,
        )

        self.__task = self.__progress.add_task("train", total=total_steps, episode_return=0.0, evaluation_return=0.0, epsilon=1.0)

        self.__evaluations: list[EvaluationMetrics] = []

    def _build_evaluation_table(self) -> Table:
        table = Table(title="evaluations")
        table.add_column("step", justify="right")
        table.add_column("mean return", justify="right")
        table.add_column("std", justify="right")

        best_return = max(evaluation.mean_return for evaluation in self.__evaluations)

        for evaluation in self.__evaluations:
            style = "bold green" if evaluation.mean_return == best_return else None

            table.add_row(
                f"{evaluation.step:,}",
                f"{evaluation.mean_return:.1f}",
                f"{evaluation.std_return:.1f}",
                style=style,
            )

        return table

    def update(self, metrics: Metrics) -> None:
        """Refresh the progress bar from a new metrics record."""
        match metrics:
            case EpisodeMetrics():
                self.__progress.update(
                    self.__task,
                    completed=metrics.step,
                    episode_return=metrics.episode_return,
                    epsilon=metrics.epsilon,
                )
            case EvaluationMetrics():
                self.__evaluations.append(metrics)
                self.__progress.update(self.__task, completed=metrics.step, evaluation_return=metrics.mean_return)

    def __enter__(self) -> Self:  # noqa: D105
        self.__progress.start()
        return self

    def __exit__(  # noqa: D105
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.__progress.stop()
        if exception is None:
            console.print(self._build_evaluation_table())
