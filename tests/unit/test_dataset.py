from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch.utils.data import DataLoader

from lit_deltalake.dataloaders import create_pytorch_dataloader
import lit_deltalake.dataloaders.pytorch as pytorch_module
from lit_deltalake.datasets import DeltaIterableDataset, DeltaKeyDataset
import lit_deltalake.datasets.iterable_dataset as dataset_module
from lit_deltalake.readers import DeltaReader, ReaderCapabilities
from lit_deltalake.readers.types import ColumnSpec, DeltaScan, Filter
from lit_deltalake.utils.errors import InvalidDatasetConfigurationError


class FakeRecordBatch:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.converted_rows: list[dict[str, object]] | None = None
        self.taken_batches: list[FakeRecordBatch] = []

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    def take(self, indices: list[int]) -> "FakeRecordBatch":
        batch = FakeRecordBatch([self.rows[index] for index in indices])
        self.taken_batches.append(batch)
        return batch

    def to_pylist(self) -> list[dict[str, object]]:
        self.converted_rows = self.rows
        return self.rows


def apply_filter_to_value(filter_function: object, value: object, filter_value: object) -> bool:
    try:
        return bool(cast(Any, filter_function)(value, filter_value))
    except AttributeError:
        assert isinstance(filter_value, tuple)
        return value in filter_value


class FakeReader(DeltaReader):
    def __init__(self, rows: list[dict[str, object]], length: int | None = None) -> None:
        self.rows = rows
        self.exact_length = length

    def scan(self, request: DeltaScan) -> Iterator[FakeRecordBatch]:
        del request
        yield FakeRecordBatch(self.rows)

    def length(self, request: DeltaScan) -> int | None:
        del request
        return self.exact_length


class KeyedFakeReader(FakeReader):
    capabilities = ReaderCapabilities(supports_keyed_lookup=True)

    def __init__(self, rows: list[dict[str, object]]) -> None:
        super().__init__(rows)
        self.scan_count = 0

    def scan(self, request: DeltaScan) -> Iterator[FakeRecordBatch]:
        self.scan_count += 1
        rows = self.rows
        for filter_clause in request.filters:
            rows = [
                row
                for row in rows
                if apply_filter_to_value(filter_clause.function, row[filter_clause.column], filter_clause.value)
            ]
        yield FakeRecordBatch(rows)


class SingleProcessFakeReader(FakeReader):
    capabilities = ReaderCapabilities(supports_worker_processes=False)


class TrackingFakeReader(FakeReader):
    def __init__(self, rows: list[dict[str, object]]) -> None:
        super().__init__(rows)
        self.batch = FakeRecordBatch(rows)

    def scan(self, request: DeltaScan) -> Iterator[FakeRecordBatch]:
        del request
        yield self.batch


class BatchedFakeReader(FakeReader):
    def __init__(self, rows: list[dict[str, object]], batch_sizes: list[int]) -> None:
        super().__init__(rows, length=len(rows))
        self.batch_sizes = batch_sizes

    def scan(self, request: DeltaScan) -> Iterator[FakeRecordBatch]:
        del request
        offset = 0
        for batch_size in self.batch_sizes:
            yield FakeRecordBatch(self.rows[offset : offset + batch_size])
            offset += batch_size


@dataclass
class TrainingRow:
    feature: float
    label: int


def test_create_pytorch_dataloader_composes_scan_dataset_and_loader() -> None:
    reader = FakeReader([{"feature": 2, "label": 1}], length=1)
    label_filter = Filter("label", lambda column, value: column == value, 1)
    loader = create_pytorch_dataloader(
        "memory://table",
        columns=("feature", "label"),
        column_specs=(ColumnSpec("feature", transform=lambda value: cast(int, value) * 3), ColumnSpec("label")),
        filters=(label_filter,),
        version=3,
        scan_batch_size=1024,
        shuffle_buffer_size=2,
        shuffle_seed=7,
        batch_size=1,
        reader=reader,
    )

    dataset = loader.dataset

    assert isinstance(dataset, DeltaIterableDataset)
    assert dataset.reader is reader
    assert dataset.scan == DeltaScan(
        "memory://table",
        columns=("feature", "label"),
        filters=(label_filter,),
        version=3,
        batch_size=1024,
    )
    assert dataset.shuffle_buffer_size == 2
    assert dataset.shuffle_seed == 7
    assert next(iter(loader)) == {"feature": 6, "label": 1}


def test_create_pytorch_dataloader_rejects_unsupported_worker_processes() -> None:
    with pytest.raises(InvalidDatasetConfigurationError, match="worker processes"):
        create_pytorch_dataloader(
            "memory://table",
            columns=("feature",),
            reader=SingleProcessFakeReader([]),
            num_workers=1,
        )


def test_create_pytorch_dataloader_derives_columns_and_batches_dataclass() -> None:
    loader = create_pytorch_dataloader(
        "memory://table",
        row_type=TrainingRow,
        batch_size=2,
        reader=FakeReader([{"feature": 1.5, "label": 1}, {"feature": 2.5, "label": 0}]),
    )

    batch = next(iter(loader))

    assert isinstance(loader.dataset, DeltaIterableDataset)
    assert loader.dataset.scan.columns == ("feature", "label")
    assert isinstance(batch, TrainingRow)
    assert torch.equal(batch.feature, torch.tensor([1.5, 2.5]))
    assert torch.equal(batch.label, torch.tensor([1, 0]))


def test_create_pytorch_dataloader_rejects_columns_and_row_type() -> None:
    with pytest.raises(InvalidDatasetConfigurationError, match="not both"):
        create_pytorch_dataloader(
            "memory://table",
            columns=("feature", "label"),
            row_type=TrainingRow,
        )


def test_create_pytorch_dataloader_creates_selected_spark_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    session = object()

    class FakeSparkReader(FakeReader):
        def __init__(self, received_session: object) -> None:
            super().__init__([])
            self.session = received_session

    monkeypatch.setattr(
        pytorch_module,
        "import_module",
        lambda name: SimpleNamespace(SparkDeltaReader=FakeSparkReader) if name.endswith(".spark") else None,
    )

    loader = create_pytorch_dataloader("memory://table", columns=("feature",), backend="spark", spark_session=session)

    assert isinstance(loader.dataset, DeltaIterableDataset)
    assert isinstance(loader.dataset.reader, FakeSparkReader)
    assert loader.dataset.reader.session is session


def test_create_pytorch_dataloader_rejects_reader_backend_conflict() -> None:
    with pytest.raises(InvalidDatasetConfigurationError, match="reader or backend"):
        create_pytorch_dataloader("memory://table", columns=("feature",), reader=FakeReader([]), backend="spark")


def test_iterable_dataset_rejects_dataclass_field_mismatch() -> None:
    with pytest.raises(InvalidDatasetConfigurationError, match="match row_type"):
        DeltaIterableDataset(
            "memory://table",
            reader=FakeReader([]),
            scan=DeltaScan("memory://table", columns=("feature", "label")),
            column_specs=(ColumnSpec("feature", name="renamed"), ColumnSpec("label")),
            row_type=TrainingRow,
        )


def test_tensordict_batch_format_stacks_dataclass_fields() -> None:
    pytest.importorskip("tensordict")
    loader = create_pytorch_dataloader(
        "memory://table",
        row_type=TrainingRow,
        batch_format="tensordict",
        batch_size=2,
        reader=FakeReader([{"feature": 1.5, "label": 1}, {"feature": 2.5, "label": 0}]),
    )

    batch = next(iter(loader))

    assert batch.batch_size == torch.Size([2])
    assert torch.equal(batch["feature"], torch.tensor([1.5, 2.5]))
    assert torch.equal(batch["label"], torch.tensor([1, 0]))


def test_iterable_dataset_transforms_and_batches_selected_columns() -> None:
    dataset = DeltaIterableDataset(
        "memory://table",
        reader=FakeReader([{"feature": 2, "label": 1, "ignored": "row"}], length=1),
        scan=DeltaScan("memory://table", columns=("feature", "label")),
        column_specs=(
            ColumnSpec("feature", name="x", transform=lambda value: cast(int, value) * 3),
            ColumnSpec("label"),
        ),
    )

    batch = next(iter(DataLoader(dataset, batch_size=1)))

    assert batch == {"x": 6, "label": 1}
    assert len(dataset) == 1


def test_iterable_dataset_rejects_unselected_column_spec() -> None:
    with pytest.raises(InvalidDatasetConfigurationError, match="must be selected"):
        DeltaIterableDataset(
            "memory://table",
            reader=FakeReader([]),
            scan=DeltaScan("memory://table", columns=("feature",)),
            column_specs=(ColumnSpec("label"),),
        )


def test_iterable_dataset_rejects_unknown_length() -> None:
    dataset = DeltaIterableDataset("memory://table", FakeReader([]), DeltaScan("memory://table", columns=("feature",)))

    with pytest.raises(TypeError, match="exact dataset length"):
        len(dataset)


def test_iterable_dataset_seeded_buffer_shuffle_is_reproducible() -> None:
    scan = DeltaScan("memory://table", columns=("id",))
    rows: list[dict[str, object]] = [{"id": index} for index in range(10)]
    first = DeltaIterableDataset("memory://table", FakeReader(rows), scan, shuffle_buffer_size=3, shuffle_seed=7)
    second = DeltaIterableDataset("memory://table", FakeReader(rows), scan, shuffle_buffer_size=3, shuffle_seed=7)

    assert list(first) == list(second)


def test_iterable_dataset_seeded_buffer_shuffle_changes_by_epoch() -> None:
    scan = DeltaScan("memory://table", columns=("id",))
    rows: list[dict[str, object]] = [{"id": index} for index in range(10)]
    dataset = DeltaIterableDataset("memory://table", FakeReader(rows), scan, shuffle_buffer_size=3, shuffle_seed=7)

    first_epoch = list(dataset)
    dataset.set_epoch(1)
    second_epoch = list(dataset)
    dataset.set_epoch(0)

    assert first_epoch != second_epoch
    assert list(dataset) == first_epoch


@pytest.mark.parametrize(
    ("shard_id", "expected_ids"),
    ((0, [0, 3, 6]), (1, [1, 4, 7]), (2, [2, 5])),
)
def test_iterable_dataset_converts_only_rows_assigned_to_shard(shard_id: int, expected_ids: list[int]) -> None:
    rows: list[dict[str, object]] = [{"id": index} for index in range(8)]
    reader = TrackingFakeReader(rows)
    dataset = DeltaIterableDataset("memory://table", reader, DeltaScan("memory://table", columns=("id",)))

    output = list(dataset._iter_rows(shard_id, shard_count=3))

    assert output == [{"id": index} for index in expected_ids]
    assert reader.batch.converted_rows is None
    assert [batch.converted_rows for batch in reader.batch.taken_batches] == [[{"id": index} for index in expected_ids]]


def test_iterable_dataset_shards_disjoint_complete_rows_across_ranks_and_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[dict[str, object]] = [{"id": index} for index in range(7)]
    dataset = DeltaIterableDataset(
        "memory://table",
        BatchedFakeReader(rows, batch_sizes=[2, 3, 2]),
        DeltaScan("memory://table", columns=("id",)),
    )
    monkeypatch.setattr(dataset_module.distributed, "is_available", lambda: True)
    monkeypatch.setattr(dataset_module.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(dataset_module.distributed, "get_world_size", lambda: 2)

    shard_rows: dict[tuple[int, int], list[dict[str, object]]] = {}
    for rank in range(2):
        monkeypatch.setattr(dataset_module.distributed, "get_rank", lambda rank=rank: rank)
        for worker_id in range(2):
            monkeypatch.setattr(
                dataset_module,
                "get_worker_info",
                lambda worker_id=worker_id: SimpleNamespace(id=worker_id, num_workers=2),
            )
            shard_rows[(rank, worker_id)] = list(dataset)

    assert shard_rows == {
        (0, 0): [{"id": 0}, {"id": 4}],
        (0, 1): [{"id": 1}, {"id": 5}],
        (1, 0): [{"id": 2}, {"id": 6}],
        (1, 1): [{"id": 3}],
    }
    flattened = [cast(int, row["id"]) for rows_for_shard in shard_rows.values() for row in rows_for_shard]
    assert sorted(flattened) == list(range(7))
    assert len(flattened) == len(set(flattened))


def test_iterable_dataset_preserves_reader_worker_capability() -> None:
    assert FakeReader([]).capabilities == ReaderCapabilities()


def test_iterable_dataset_uses_default_scan_for_table_uri() -> None:
    dataset = DeltaIterableDataset("memory://table", FakeReader([{"feature": 2}]))

    assert dataset.scan == DeltaScan("memory://table")
    assert list(dataset) == [{"feature": 2}]


def test_keyed_dataset_returns_requested_batch_order() -> None:
    dataset = DeltaKeyDataset(
        reader=KeyedFakeReader([{"id": 2, "feature": "second"}, {"id": 1, "feature": "first"}]),
        scan=DeltaScan("memory://table", columns=("id", "feature"), version=3),
        id_column="id",
        keys=(1, 2),
    )

    assert dataset.__getitems__([1, 0]) == [
        {"id": 2, "feature": "second"},
        {"id": 1, "feature": "first"},
    ]


def test_keyed_dataset_caches_repeated_single_key_lookups() -> None:
    reader = KeyedFakeReader([{"id": 1, "feature": "first"}])
    dataset = DeltaKeyDataset(
        reader=reader,
        scan=DeltaScan("memory://table", columns=("id", "feature"), version=3),
        id_column="id",
        keys=(1,),
    )

    assert dataset[0] == {"id": 1, "feature": "first"}
    assert dataset[0] == {"id": 1, "feature": "first"}
    assert reader.scan_count == 1


def test_keyed_dataset_requires_pinned_version() -> None:
    with pytest.raises(InvalidDatasetConfigurationError, match="explicit Delta table version"):
        DeltaKeyDataset(
            reader=KeyedFakeReader([]),
            scan=DeltaScan("memory://table", columns=("id",)),
            id_column="id",
            keys=(1,),
        )
