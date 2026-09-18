# lit-deltalake

[![Build](https://github.com/twsl/lit-deltalake/actions/workflows/build.yaml/badge.svg)](https://github.com/twsl/lit-deltalake/actions/workflows/build.yaml)
[![Documentation](https://github.com/twsl/lit-deltalake/actions/workflows/docs.yaml/badge.svg)](https://github.com/twsl/lit-deltalake/actions/workflows/docs.yaml)
[![PyPI - Package Version](https://img.shields.io/pypi/v/lit-deltalake?logo=pypi&style=flat&color=orange)](https://pypi.org/project/lit-deltalake/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lit-deltalake?logo=pypi&style=flat&color=blue)](https://pypi.org/project/lit-deltalake/)

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

Bring [Delta Lake](https://github.com/delta-io/delta) to [Pytorch Lightning](https://github.com/lightning-ai/pytorch-lightning)

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
from lit_deltalake.readers.types import Filter
import lightning
import torch
from torch import nn
from torch.utils.data import DataLoader


loader = create_pytorch_dataloader(
    "/path/to/training-table",
    columns=("feature_a", "feature_b", "label"),
    filters=(Filter("split", lambda column, value: column == value, "train"),),
    batch_size=128,
    num_workers=4,
    multiprocessing_context="spawn",
)


class EventsDataModule(lightning.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self.loader = loader

    def train_dataloader(self) -> DataLoader:
        return self.loader


class Classifier(lightning.LightningModule):
    def __init__(self, learning_rate: float = 1e-3) -> None:
        super().__init__()
        self.model = nn.Linear(2, 2)
        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        features = torch.stack((batch["feature_a"], batch["feature_b"]), dim=1).float()
        labels = batch["label"]
        loss = self.loss_function(self(features), labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


trainer = lightning.Trainer(max_epochs=10)
trainer.fit(Classifier(), datamodule=EventsDataModule())
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
