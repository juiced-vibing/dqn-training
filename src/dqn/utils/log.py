from typing import TYPE_CHECKING

from loguru import logger


if TYPE_CHECKING:
    from pathlib import Path


def configure_logging(path: Path) -> None:
    """Send all logs to `path` only."""
    # the default stderr sink would tear the rich live display
    logger.remove()
    logger.add(path, level="DEBUG")
