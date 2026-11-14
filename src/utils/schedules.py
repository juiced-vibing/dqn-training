def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    progress = min(step / duration, 1.0)
    return start + progress * (end - start)
