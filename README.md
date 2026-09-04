# lit-deltalake

[![Build](https://github.com/twsl/lit-deltalake/actions/workflows/build.yaml/badge.svg)](https://github.com/twsl/lit-deltalake/actions/workflows/build.yaml)
[![Documentation](https://github.com/twsl/lit-deltalake/actions/workflows/docs.yaml/badge.svg)](https://github.com/twsl/lit-deltalake/actions/workflows/docs.yaml)

<!--- [![PyPI - Package Version](https://img.shields.io/pypi/v/lit-deltalake?logo=pypi&style=flat&color=orange)](https://pypi.org/project/lit-deltalake/) -->
<!--- [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lit-deltalake?logo=pypi&style=flat&color=blue)](https://pypi.org/project/lit-deltalake/) -->

[![Docs with MkDocs](https://img.shields.io/badge/MkDocs-docs?style=flat&logo=materialformkdocs&logoColor=white&color=%23526CFE)](https://squidfunk.github.io/mkdocs-material/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/j178/prek)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/twsl/lit-deltalake/releases)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-border.json)](https://github.com/copier-org/copier)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Bring [Delta Lake](https://github.com/delta-io/delta) to Pytorch Lightning

## Features

- Stream immutable Delta table snapshots into PyTorch `IterableDataset` instances.
- Shard Delta-rs source fragments across initialized DDP ranks and DataLoader workers.
- Use Delta-rs, an existing PySpark session, or an existing Sail Spark Connect session.
- Select, filter, transform, decode, and rename columns before training.
- Pin Delta table versions for reproducible training data.

## Installation

Install the default Delta-rs and PyArrow reader:

```bash
uv add lit-deltalake
```

Add an optional reader when needed:

```bash
uv add "lit-deltalake[spark]"  # existing PySpark session
uv add "lit-deltalake[sail]"   # existing Sail Spark Connect session
```

With `pip`:

```bash
python -m pip install lit-deltalake
```

## Usage

```python
from lit_deltalake.dataloaders import create_pytorch_dataloader
from lit_deltalake.datasets import Filter

loader = create_pytorch_dataloader(
    "/path/to/training-table",
    columns=("feature_a", "feature_b", "label"),
    filters=(Filter("split", "=", "train"),),
    version=42,
    batch_size=128,
    num_workers=4,
    multiprocessing_context="spawn",
)

for batch in loader:
    features = batch["feature_a"], batch["feature_b"]
    labels = batch["label"]
    # Train your LightningModule.
```

Use `multiprocessing_context="spawn"` or `"forkserver"` with Delta-rs workers. Delta-rs is the supported reader for
multi-rank DDP: each rank/worker scans only its assigned source fragments. Fragment sizes can differ, so this optimizes
scan locality rather than guaranteeing perfectly equal row counts.

Delta-rs streams Arrow scan batches and supports source sharding across DataLoader workers and DDP ranks. Spark and
Sail stream rows through `DataFrame.toLocalIterator()`, keeping client memory bounded by the largest Spark partition
plus one Arrow batch. They require `num_workers=0` and reject multi-rank DDP loading because each rank would otherwise
execute the full query. Repartition a table when its individual Spark partitions are too large for driver memory.
Configure cloud storage credentials through the Spark session's Hadoop settings when using those readers.

Launch the ImageNet example with local DDP:

```bash
uv run torchrun --standalone --nproc_per_node=2 \
    examples/imagenet/ddp_spark_pytorch_distributor.py TRAIN_TABLE VALIDATION_TABLE \
    --accelerator gpu --devices 1 --strategy ddp
```

## Examples

See [examples](./examples/) and [notebooks](./notebooks/) for more usage examples.

## Docs

```bash
uv run mkdocs build -f ./mkdocs.yml -d ./_build/
```

## Update template

```bash
copier update --trust -A --vcs-ref=HEAD
```

## Credits

- Thanks for [deltatorch](https://github.com/delta-incubator/deltatorch) for inspiration and code

This project was generated with [![🚀 python project template.](https://img.shields.io/badge/python--project--template-%F0%9F%9A%80-brightgreen)](https://github.com/twsl/python-project-template)
