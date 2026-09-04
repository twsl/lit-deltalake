from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from typing_extensions import TypeVar

SampleT = TypeVar("SampleT", default=dict[str, object])
Record = dict[str, object]
RowTransform = Callable[[object], object]
Transform = RowTransform | Sequence[RowTransform]
ValueTransform = Callable[[object], object]
type StorageOptions = Mapping[str, str]
type FilterValue = str | int | float | bool | None | Sequence[str | int | float | bool]
type FilterFunction = Callable[[Any, Any], Any]

type RowType = type[object]


@dataclass(frozen=True, slots=True)
class Filter:
    """Portable filter clause translated by a reader implementation."""

    column: str
    function: FilterFunction
    value: FilterValue

    def __post_init__(self) -> None:
        if not self.column:
            raise ValueError("Filter column must not be empty.")
        if not callable(self.function):
            raise TypeError("Filter function must be callable.")


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Describe one selected source column and optional sample conversion."""

    source: str
    name: str | None = None
    decoder: ValueTransform | None = field(default=None, compare=False, hash=False)
    transform: ValueTransform | None = field(default=None, compare=False, hash=False)

    @property
    def output_name(self) -> str:
        """Return the configured output name or source column name."""
        return self.name or self.source

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("Column source must not be empty.")
        if self.name == "":
            raise ValueError("Column output name must not be empty.")


@dataclass(frozen=True, slots=True)
class DeltaScan:
    """Engine-independent read configuration for one Delta table snapshot."""

    table_uri: str
    columns: tuple[str, ...] | None = None
    filters: tuple[Filter, ...] = ()
    version: int | None = None
    storage_options: StorageOptions | None = None
    batch_size: int = 65_536

    def __post_init__(self) -> None:
        if not self.table_uri:
            raise ValueError("Delta table URI must not be empty.")
        if self.columns is not None and not self.columns:
            raise ValueError("Delta scan columns must not be empty when provided.")
        if self.columns is not None and len(set(self.columns)) != len(self.columns):
            raise ValueError("Delta scan columns must be unique.")
        if self.version is not None and self.version < 0:
            raise ValueError("Delta table version must be non-negative.")
        if self.batch_size < 1:
            raise ValueError("Delta scan batch_size must be positive.")
