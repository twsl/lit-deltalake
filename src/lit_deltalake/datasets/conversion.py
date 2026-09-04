from collections.abc import Callable, Sequence
from dataclasses import asdict, fields, is_dataclass
from typing import Any, cast

from torch.utils.data import default_collate

from lit_deltalake.readers.types import Record, RowTransform, RowType, SampleT, Transform
from lit_deltalake.utils.errors import BackendUnavailableError, InvalidDatasetConfigurationError


def dataclass_columns(row_type: RowType) -> tuple[str, ...]:
    """Return Delta column names declared as dataclass fields."""
    if not isinstance(row_type, type) or not is_dataclass(row_type):
        raise InvalidDatasetConfigurationError("row_type must be a dataclass type.")
    dataclass_type = cast(Any, row_type)
    return tuple(field.name for field in fields(dataclass_type))


def apply_transforms(sample: object, transform: Transform | None) -> object:
    """Apply one or more ordered whole-sample transforms."""
    if transform is None:
        return sample
    transforms: Sequence[RowTransform] = (
        (cast(RowTransform, transform),) if callable(transform) else cast(Sequence[RowTransform], transform)
    )
    for callback in transforms:
        sample = callback(sample)
    return sample


def to_sample(record: Record, row_type: RowType | None, transform: Transform | None) -> SampleT:
    """Convert one selected record into a mapping or typed dataclass sample."""
    sample = row_type(**record) if row_type is not None else record
    return cast(SampleT, apply_transforms(sample, transform))


def dataclass_collate(samples: list[object]) -> object:
    """Collate matching dataclass samples while retaining their type."""
    if not samples or not is_dataclass(samples[0]):
        return default_collate(samples)
    row_type = cast(Any, type(samples[0]))
    return row_type(
        **{
            field.name: default_collate([getattr(sample, field.name) for sample in samples])
            for field in fields(row_type)
        }
    )


def tensordict_collate(samples: list[object]) -> object:
    """Collate mappings or dataclasses into one TensorDict batch."""
    try:
        from tensordict import TensorDict
    except ImportError as error:
        raise BackendUnavailableError(
            "TensorDict batches require `lit-deltalake[tensordict]`. Install the optional extra first."
        ) from error
    records = [asdict(cast(Any, sample)) if is_dataclass(sample) else sample for sample in samples]
    if any(not isinstance(record, dict) for record in records):
        raise InvalidDatasetConfigurationError("TensorDict batches require mapping or dataclass samples.")
    return TensorDict(default_collate(records), batch_size=[len(records)])
