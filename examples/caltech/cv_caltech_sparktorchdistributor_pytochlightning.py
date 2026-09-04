"""Train Caltech101 from local or remote Delta tables with PyTorch Lightning.

Tables require binary ``image`` data and integer ``label`` values. Prepare them
with ``uv run python scripts/prep_caltech.py`` and install torchvision.
"""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, cast

import lightning as lightning
import torch
from torch import nn
from torchvision.io import decode_image

from lit_deltalake.datamodule import DeltaDataModule
from lit_deltalake.datasets import DeltaIterableDataset
from lit_deltalake.readers import DeltaRsReader
from lit_deltalake.readers.types import ColumnSpec, DeltaScan


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    train_uri: str
    test_uri: str
    batch_size: int
    epochs: int
    learning_rate: float
    num_classes: int
    num_workers: int
    version: int | None


def _decode_image(value: bytes) -> Any:
    return decode_image(torch.frombuffer(bytearray(value), dtype=torch.uint8))


def _image_transform() -> Any:
    from torchvision import transforms

    return transforms.Compose(
        (
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        )
    )


def _dataset_factory(table_uri: str, config: TrainingConfig, shuffle: bool) -> DeltaIterableDataset[Any]:
    return DeltaIterableDataset(
        table_uri,
        DeltaRsReader(),
        DeltaScan(table_uri, columns=("image", "label"), version=config.version),
        column_specs=(
            ColumnSpec("image", decoder=lambda value: _decode_image(cast(bytes, value)), transform=_image_transform()),
            ColumnSpec("label", transform=lambda value: int(cast(int, value))),
        ),
        shuffle_buffer_size=config.batch_size * 16 if shuffle else 0,
    )


class CaltechDataModule(DeltaDataModule[Any]):
    """Create Delta-backed train, validation, and test DataLoaders."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__(
            train_factory=lambda: _dataset_factory(config.train_uri, config, shuffle=True),
            validation_factory=lambda: _dataset_factory(config.test_uri, config, shuffle=False),
            test_factory=lambda: _dataset_factory(config.test_uri, config, shuffle=False),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            multiprocessing_context="spawn" if config.num_workers else None,
        )
        self.num_classes = config.num_classes


class CaltechClassifier(lightning.LightningModule):
    """Fine-tune MobileNetV3 for Caltech101 classification."""

    def __init__(self, num_classes: int, learning_rate: float) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        self.save_hyperparameters()
        self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)
        self.learning_rate = learning_rate

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> torch.Tensor:
        del batch_index
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> None:
        del batch_index
        self._step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> None:
        del batch_index
        self._step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        logits = self(batch["image"])
        labels = batch["label"].to(dtype=torch.long)
        loss = nn.functional.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        self.log(f"{stage}_loss", loss, on_step=stage == "train", on_epoch=True, prog_bar=stage != "train")
        self.log(f"{stage}_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss


def _arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Train Caltech101 from Delta tables with local PyTorch Lightning.")
    parser.add_argument("train_uri")
    parser.add_argument("test_uri")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-classes", type=int, default=101)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--version", type=int)
    return parser


def main() -> None:
    arguments: Namespace = _arguments().parse_args()
    if arguments.epochs < 1 or arguments.num_classes < 2 or arguments.num_workers < 0:
        raise ValueError("epochs must be positive, num_classes at least 2, and num_workers non-negative.")
    config = TrainingConfig(
        train_uri=arguments.train_uri,
        test_uri=arguments.test_uri,
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        num_classes=arguments.num_classes,
        num_workers=arguments.num_workers,
        version=arguments.version,
    )
    torch.set_float32_matmul_precision("medium")
    data_module = CaltechDataModule(config)
    trainer = lightning.Trainer(accelerator="auto", devices=1, max_epochs=config.epochs)
    model = CaltechClassifier(data_module.num_classes, config.learning_rate)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
