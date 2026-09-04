from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
from torch.utils.data import DataLoader

from lit_deltalake.datasets import DeltaIterableDataset
from lit_deltalake.readers import DeltaRsReader
from lit_deltalake.readers.types import DeltaScan, Filter

SAMPLE_TABLE = Path(__file__).parents[2] / "data" / "sample_events"


def equals_filter(column: object, value: object) -> object:
    return column == value


def test_delta_rs_reader_projects_filters_and_shards_workers() -> None:
    dataset = DeltaIterableDataset(
        str(SAMPLE_TABLE),
        DeltaRsReader(),
        DeltaScan(
            str(SAMPLE_TABLE),
            columns=("event_id", "current_balance", "deposit_amount"),
            filters=(Filter("is_test", equals_filter, False),),
            batch_size=2,
        ),
    )

    batches = list(DataLoader(dataset, batch_size=1, num_workers=2, multiprocessing_context="spawn"))
    identifiers = {batch["event_id"][0] for batch in batches}

    assert len(identifiers) == 95
    assert all(batch["current_balance"].item() >= 0 for batch in batches)
    assert sum(batch["deposit_amount"].item() > 0 for batch in batches) == 40


def test_delta_rs_reader_streams_record_batches() -> None:
    reader = DeltaRsReader()
    request = DeltaScan(str(SAMPLE_TABLE), columns=("event_id", "current_balance"), batch_size=32)

    batches: Iterator[pa.RecordBatch] = reader.scan(request)

    assert [batch.num_rows for batch in batches] == [32, 32, 32, 4]


def test_delta_rs_reader_reports_filtered_scan_length() -> None:
    reader = DeltaRsReader()
    request = DeltaScan(str(SAMPLE_TABLE), filters=(Filter("is_test", equals_filter, False),))

    assert reader.length(DeltaScan(str(SAMPLE_TABLE))) == 100
    assert reader.length(request) == 95
