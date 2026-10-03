"""Dataset APIs; importing permissions/paths must not initialise graphics modules."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from epsbench.data.generate import generate_dataset as generate_dataset
    from epsbench.data.inspect import create_inspection_image as create_inspection_image
    from epsbench.data.loader import (
        DatasetLoader as DatasetLoader,
    )
    from epsbench.data.loader import (
        LoadedAnalyticOpticalTransport as LoadedAnalyticOpticalTransport,
    )
    from epsbench.data.loader import (
        LoadedCanonicalPairedOutput as LoadedCanonicalPairedOutput,
    )
    from epsbench.data.loader import (
        LoadedEcologicalVisibilityEvents as LoadedEcologicalVisibilityEvents,
    )
    from epsbench.data.loader import (
        PermissionDeniedError as PermissionDeniedError,
    )
    from epsbench.data.validate import (
        DatasetValidationError as DatasetValidationError,
    )
    from epsbench.data.validate import (
        validate_dataset as validate_dataset,
    )

__all__ = [
    "DatasetLoader",
    "DatasetValidationError",
    "LoadedAnalyticOpticalTransport",
    "LoadedCanonicalPairedOutput",
    "LoadedEcologicalVisibilityEvents",
    "PermissionDeniedError",
    "create_inspection_image",
    "generate_dataset",
    "validate_dataset",
]


def __getattr__(name: str) -> Any:
    """Resolve the same public objects on demand, preserving their identity/API."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = {
        "generate_dataset": "generate",
        "create_inspection_image": "inspect",
        "DatasetValidationError": "validate",
        "validate_dataset": "validate",
    }.get(name, "loader")
    value = getattr(import_module(f"epsbench.data.{module}"), name)
    globals()[name] = value
    return value
