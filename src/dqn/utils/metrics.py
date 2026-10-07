from typing import TYPE_CHECKING
from typing import Self

from msgspec import Struct
from msgspec.json import Encoder


if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


class EpisodeMetrics(Struct, frozen=True, tag="episode"):
    """Outcome of one finished training episode."""

    step: int
    episode_return: float
    episode_length: int
    epsilon: float


class LearnMetrics(Struct, frozen=True, tag="learn"):
    """Result of one gradient step."""

    step: int
    loss: float
    mean_q: float


class EvaluationMetrics(Struct, frozen=True, tag="evaluation"):
    """Greedy-policy returns over held-out seeds."""

    step: int
    mean_return: float
    std_return: float


type Metrics = EpisodeMetrics | LearnMetrics | EvaluationMetrics


class MetricsWriter:
    """Append metrics to a JSONL file."""

    def __init__(self, path: Path) -> None:
        self.__file = path.open("ab")
        self.__encoder = Encoder()

    def write(self, metrics: Metrics) -> None:
        """Append one record as a JSON line."""
        self.__file.write(self.__encoder.encode(metrics) + b"\n")

    def __enter__(self) -> Self:  # noqa: D105
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the file."""
        self.__file.close()
