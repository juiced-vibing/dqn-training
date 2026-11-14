from typing import TYPE_CHECKING
from typing import Self

from msgspec import Struct
from msgspec.json import Encoder


if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


class EpisodeMetrics(Struct, frozen=True, tag="episode"):
    step: int
    episode_return: float
    episode_length: int
    epsilon: float


class LearnMetrics(Struct, frozen=True, tag="learn"):
    step: int
    loss: float
    mean_q: float


class EvaluationMetrics(Struct, frozen=True, tag="evaluation"):
    step: int
    mean_return: float
    std_return: float


type Metrics = EpisodeMetrics | LearnMetrics | EvaluationMetrics


class MetricsWriter:
    def __init__(self, path: Path) -> None:
        self.__file = path.open("ab")
        self.__encoder = Encoder()

    def write(self, metrics: Metrics) -> None:
        self.__file.write(self.__encoder.encode(metrics) + b"\n")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.__file.close()
