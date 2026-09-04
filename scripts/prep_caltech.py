"""Download Hugging Face Caltech101 data and write a Delta table."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from deltalake import write_deltalake
import pyarrow as pa
import torch
from torchvision.io import decode_image

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "caltech101"
DATASET_ID = "flwrlabs/caltech101"
DATASET_REVISION = "afb5a6ba4fe9311de42497984d33266bb86e2bb9"
CHUNK_SIZE = 256


def _image_bytes(image: Any) -> bytes:
    if not isinstance(image, dict) or image.get("bytes") is None:
        raise ValueError("Expected Hugging Face image data with encoded bytes.")
    encoded = bytes(image["bytes"])
    decode_image(torch.frombuffer(bytearray(encoded), dtype=torch.uint8))
    return encoded


def _load_dataset() -> Any:
    try:
        from datasets import Image, load_dataset
    except ImportError as error:
        raise RuntimeError("Install Hugging Face Datasets with `uv add datasets`.") from error
    try:
        dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION)
        return dataset.cast_column("image", Image(decode=False))
    except Exception as error:
        raise RuntimeError(f"Unable to download {DATASET_ID} from Hugging Face.") from error


def _batches(dataset: Any) -> Iterator[pa.RecordBatch]:
    for start in range(0, len(dataset), CHUNK_SIZE):
        rows = dataset[start : start + CHUNK_SIZE]
        yield pa.RecordBatch.from_arrays(
            [
                pa.array([_image_bytes(image) for image in rows["image"]], type=pa.binary()),
                pa.array([int(label) for label in rows["label"]], type=pa.int64()),
            ],
            names=["image", "label"],
        )


def _write_table(path: Path, batches: Iterator[pa.RecordBatch]) -> int:
    first = next(batches, None)
    if first is None:
        raise ValueError(f"Cannot write empty table: {path}")
    write_deltalake(path, first, mode="overwrite")
    row_count = first.num_rows
    for batch in batches:
        write_deltalake(path, batch, mode="append")
        row_count += batch.num_rows
    return row_count


def main() -> None:
    dataset = _load_dataset()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split_name, split in dataset.items():
        counts[split_name] = _write_table(DATA_ROOT / split_name, _batches(split))
    print(f"Wrote Caltech101 Delta tables: {counts}.")


if __name__ == "__main__":
    main()
