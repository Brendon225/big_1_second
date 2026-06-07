import math


def finite_or_default(value: float, default: float) -> float:
    return value if math.isfinite(value) else default
