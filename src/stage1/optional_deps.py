import importlib.util
from typing import Iterable, List


class OptionalDependencyError(RuntimeError):
    """Raised when an optional training backend is requested without dependencies."""


def missing_modules(module_names: Iterable[str]) -> List[str]:
    return [name for name in module_names if importlib.util.find_spec(name) is None]


def require_modules(module_names: Iterable[str], feature: str) -> None:
    missing = missing_modules(module_names)
    if missing:
        joined = ", ".join(missing)
        raise OptionalDependencyError(
            f"Missing optional dependencies for {feature}: {joined}. "
            "Install the minimal training stack from requirements-stage1.txt before using this backend."
        )
