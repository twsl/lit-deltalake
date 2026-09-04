from importlib import import_module
from typing import TYPE_CHECKING, Any

from lit_deltalake.readers.base import DeltaReader, ReaderCapabilities
from lit_deltalake.readers.delta_rs import DeltaRsReader

if TYPE_CHECKING:
    from lit_deltalake.readers.sail import SailDeltaReader, start_sail_spark_connect_server
    from lit_deltalake.readers.spark import SparkDeltaReader

__all__ = [
    "DeltaReader",
    "DeltaRsReader",
    "ReaderCapabilities",
    "SailDeltaReader",
    "SparkDeltaReader",
    "start_sail_spark_connect_server",
]


def __getattr__(name: str) -> Any:
    """Load optional reader backends only when they are requested."""
    if name in {"SparkDeltaReader", "SailDeltaReader", "start_sail_spark_connect_server"}:
        module_name = "sail" if name in {"SailDeltaReader", "start_sail_spark_connect_server"} else "spark"
        return getattr(import_module(f"{__name__}.{module_name}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
