"""Download Hugging Face ImageNet-1k data and write Delta tables."""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
from typing import Any

from deltalake import write_deltalake
import pyarrow as pa
import torch
from torchvision.io import decode_image

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "imagenet"
DATASET_ID = "ILSVRC/imagenet-1k"
DATASET_REVISION = "49e2ee26f3810fb5a7536bbf732a7b07389a47b5"
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
    token = os.environ.get("HF_TOKEN")
    try:
        dataset = load_dataset(DATASET_ID, token=token, revision=DATASET_REVISION)
        return dataset.cast_column("image", Image(decode=False))
    except Exception as error:
        raise RuntimeError(
            "Unable to download ImageNet-1k. The dataset is gated; set HF_TOKEN and accept "
            "its Hugging Face terms before running this script."
        ) from error


def _batches(dataset: Any) -> Iterator[pa.RecordBatch]:
    for start in range(0, len(dataset), CHUNK_SIZE):
        rows = dataset[start : start + CHUNK_SIZE]
        contents = [_image_bytes(image) for image in rows["image"]]
        object_ids = [int(label) for label in rows["label"]]
        yield pa.RecordBatch.from_arrays(
            [pa.array(contents, type=pa.binary()), pa.array(object_ids, type=pa.int64())],
            names=["content", "object_id"],
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
        output_name = "validation" if split_name in {"validation", "val"} else split_name
        counts[output_name] = _write_table(DATA_ROOT / output_name, _batches(split))
    print(f"Wrote ImageNet-1k Delta tables: {counts}.")


if __name__ == "__main__":
    main()
