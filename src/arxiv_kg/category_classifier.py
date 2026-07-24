"""Train, validate, evaluate, and inspect an arXiv category classifier."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class PaperExample:
    """One supervised example: paper text plus its correct primary category."""

    arxiv_id: str
    title: str
    text: str
    label: str


@dataclass(frozen=True)
class ClassificationMetrics:
    """Final metrics calculated once from untouched test papers."""

    test_loss: float
    accuracy: float
    macro_f1: float
    weighted_f1: float
    top_3_accuracy: float
    correct_papers: int
    test_papers: int
    validation_papers: int
    training_papers: int
    categories: int
    excluded_papers: int
    best_epoch: int
    untrained_validation_accuracy: float
    majority_test_accuracy: float
    uniform_random_expected_accuracy: float
    expected_calibration_error: float


@dataclass
class ClassificationRun:
    """A fitted model plus serializable evidence from training and evaluation."""

    model: "CategoryNetwork"
    vectorizer: FeatureUnion
    encoder: LabelEncoder
    metrics: ClassificationMetrics
    history: list[dict[str, float | int]]
    per_category: dict[str, Any]
    confusion_pairs: list[dict[str, Any]]
    high_confidence_mistakes: list[dict[str, Any]]
    selective_accuracy: list[dict[str, float | int]]
    configuration: dict[str, Any]


class CategoryNetwork(nn.Module):
    """One-hidden-layer classifier that returns raw category logits."""

    def __init__(
        self,
        input_size: int,
        category_count: int,
        *,
        hidden_neurons: int = 384,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_neurons),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_neurons, category_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def _required_text(record: dict[str, Any], key: str, line_number: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {key} must be a non-empty string")
    return value.strip()


def load_paper_examples(
    dataset_path: Path,
    minimum_class_count: int,
) -> tuple[list[PaperExample], int]:
    """Load valid JSONL records and remove classes too small for three splits."""
    if minimum_class_count < 3:
        raise ValueError(
            "minimum_class_count must be at least 3 for train, validation, and test splits"
        )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"paper dataset not found: {dataset_path}")

    records: list[PaperExample] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: paper record must be an object")
            title = _required_text(record, "title", line_number)
            abstract = _required_text(record, "abstract", line_number)
            label = _required_text(record, "primary_category", line_number)
            arxiv_id = str(record.get("arxiv_id") or f"line-{line_number}")
            records.append(
                PaperExample(
                    arxiv_id=arxiv_id,
                    title=title,
                    text=f"{title}\n\n{abstract}",
                    label=label,
                )
            )

    counts = Counter(record.label for record in records)
    included = [
        record
        for record in records
        if counts[record.label] >= minimum_class_count
    ]
    if not included:
        raise ValueError("no papers remain after applying minimum_class_count")
    return included, len(records) - len(included)


def load_labeled_papers(
    dataset_path: Path,
    minimum_class_count: int,
) -> tuple[list[str], list[str], int]:
    """Compatibility wrapper returning texts, labels, and excluded row count."""
    examples, excluded = load_paper_examples(dataset_path, minimum_class_count)
    return (
        [example.text for example in examples],
        [example.label for example in examples],
        excluded,
    )


def stratified_three_way_split(
    targets: np.ndarray,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split every class independently so rare retained classes reach each split."""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation_fraction and test_fraction must be between 0 and 1")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction plus test_fraction must be less than 1")

    generator = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []
    for target in np.unique(targets):
        indices = np.flatnonzero(targets == target)
        generator.shuffle(indices)
        if len(indices) < 3:
            raise ValueError(f"class {target} has fewer than 3 papers")

        test_count = max(1, int(round(len(indices) * test_fraction)))
        validation_count = max(1, int(round(len(indices) * validation_fraction)))
        while test_count + validation_count >= len(indices):
            if test_count >= validation_count and test_count > 1:
                test_count -= 1
            elif validation_count > 1:
                validation_count -= 1
            else:
                raise ValueError(f"class {target} cannot be split three ways")

        test_indices.extend(indices[:test_count])
        validation_indices.extend(indices[test_count : test_count + validation_count])
        train_indices.extend(indices[test_count + validation_count :])

    for indices in (train_indices, validation_indices, test_indices):
        generator.shuffle(indices)
    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(validation_indices, dtype=np.int64),
        np.asarray(test_indices, dtype=np.int64),
    )


def build_text_vectorizer(max_features: int) -> FeatureUnion:
    """Combine word semantics with character patterns common in technical terms."""
    if max_features < 100:
        raise ValueError("max_features must be at least 100")
    word_features = int(max_features * 0.75)
    character_features = max_features - word_features
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=word_features,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.995,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    dtype=np.float32,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    max_features=character_features,
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit device and fail clearly when unavailable."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is false")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS requested but torch.backends.mps.is_available() is false")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: cpu, cuda, mps, auto")
    return torch.device(requested)


def _as_tensor(features: Any) -> torch.Tensor:
    return torch.from_numpy(features.toarray().astype(np.float32, copy=False))


def _class_weights(targets: np.ndarray, category_count: int) -> torch.Tensor:
    counts = np.bincount(targets, minlength=category_count).astype(np.float32)
    weights = np.sqrt(counts.max() / counts)
    return torch.from_numpy(weights / weights.mean())


def _loader(
    features: torch.Tensor,
    targets: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(features, torch.from_numpy(targets.astype(np.int64))),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def _train_one_epoch(
    model: CategoryNetwork,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size
    return loss_sum / total, correct / total


@torch.inference_mode()
def _evaluate(
    model: CategoryNetwork,
    dataloader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    total = 0
    target_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        batch_size = labels.size(0)
        loss_sum += loss_function(logits, labels).item() * batch_size
        total += batch_size
        target_parts.append(labels.cpu().numpy())
        probability_parts.append(torch.softmax(logits, dim=1).cpu().numpy())
    return (
        loss_sum / total,
        np.concatenate(target_parts),
        np.concatenate(probability_parts),
    )


def _expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    error = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            bin_accuracy = (predictions[mask] == targets[mask]).mean()
            error += float(mask.mean() * abs(bin_accuracy - confidence[mask].mean()))
    return error


def _selective_accuracy(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, float | int]]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    results: list[dict[str, float | int]] = []
    for threshold in (0.50, 0.70, 0.90):
        mask = confidence >= threshold
        results.append(
            {
                "confidence_threshold": threshold,
                "papers": int(mask.sum()),
                "coverage": float(mask.mean()),
                "accuracy": float((predictions[mask] == targets[mask]).mean())
                if mask.any()
                else 0.0,
            }
        )
    return results


def _error_analysis(
    examples: list[PaperExample],
    targets: np.ndarray,
    probabilities: np.ndarray,
    encoder: LabelEncoder,
    maximum_mistakes: int = 25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    mistakes = np.flatnonzero(predictions != targets)
    ranked_mistakes = mistakes[np.argsort(-confidence[mistakes])][:maximum_mistakes]
    mistake_records: list[dict[str, Any]] = []
    for index in ranked_mistakes:
        top_indices = probabilities[index].argsort()[-3:][::-1]
        mistake_records.append(
            {
                "arxiv_id": examples[int(index)].arxiv_id,
                "title": examples[int(index)].title,
                "true_category": str(encoder.inverse_transform([targets[index]])[0]),
                "predicted_category": str(
                    encoder.inverse_transform([predictions[index]])[0]
                ),
                "confidence": float(confidence[index]),
                "top_predictions": [
                    {
                        "category": str(encoder.inverse_transform([class_index])[0]),
                        "probability": float(probabilities[index, class_index]),
                    }
                    for class_index in top_indices
                ],
            }
        )

    pairs = Counter(
        (
            str(encoder.inverse_transform([targets[index]])[0]),
            str(encoder.inverse_transform([predictions[index]])[0]),
        )
        for index in mistakes
    )
    confusion_pairs = [
        {"true_category": true, "predicted_category": predicted, "papers": count}
        for (true, predicted), count in pairs.most_common(20)
    ]
    return mistake_records, confusion_pairs


def train_and_evaluate(
    dataset_path: Path,
    *,
    minimum_class_count: int = 5,
    epochs: int = 20,
    batch_size: int = 128,
    seed: int = 42,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    max_features: int = 8_000,
    hidden_neurons: int = 384,
    dropout: float = 0.25,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 4,
    device_name: str = "cpu",
) -> ClassificationRun:
    """Tune epoch count on validation data, then evaluate untouched test data."""
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must all be positive")
    if hidden_neurons < 1:
        raise ValueError("hidden_neurons must be positive")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be at least 0 and less than 1")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device_name)

    examples, excluded_papers = load_paper_examples(
        dataset_path, minimum_class_count
    )
    encoder = LabelEncoder()
    targets = encoder.fit_transform([example.label for example in examples])
    if len(encoder.classes_) < 2:
        raise ValueError("classifier requires at least two eligible categories")
    train_indices, validation_indices, test_indices = stratified_three_way_split(
        targets,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    train_examples = [examples[index] for index in train_indices]
    validation_examples = [examples[index] for index in validation_indices]
    test_examples = [examples[index] for index in test_indices]
    vectorizer = build_text_vectorizer(max_features)
    train_features = _as_tensor(
        vectorizer.fit_transform([example.text for example in train_examples])
    )
    validation_features = _as_tensor(
        vectorizer.transform([example.text for example in validation_examples])
    )
    test_features = _as_tensor(
        vectorizer.transform([example.text for example in test_examples])
    )
    train_targets = targets[train_indices]
    validation_targets = targets[validation_indices]
    test_targets = targets[test_indices]

    train_loader = _loader(
        train_features,
        train_targets,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _loader(
        validation_features,
        validation_targets,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    test_loader = _loader(
        test_features,
        test_targets,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    model = CategoryNetwork(
        train_features.shape[1],
        len(encoder.classes_),
        hidden_neurons=hidden_neurons,
        dropout=dropout,
    ).to(device)
    evaluation_loss = nn.CrossEntropyLoss()
    _, untrained_targets, untrained_probabilities = _evaluate(
        model, validation_loader, evaluation_loss, device
    )
    untrained_validation_accuracy = float(
        accuracy_score(
            untrained_targets,
            untrained_probabilities.argmax(axis=1),
        )
    )

    training_loss = nn.CrossEntropyLoss(
        weight=_class_weights(train_targets, len(encoder.classes_)).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_macro_f1 = -1.0
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        epoch_train_loss, epoch_train_accuracy = _train_one_epoch(
            model, train_loader, optimizer, training_loss, device
        )
        validation_loss, epoch_validation_targets, validation_probabilities = (
            _evaluate(model, validation_loader, evaluation_loss, device)
        )
        validation_predictions = validation_probabilities.argmax(axis=1)
        validation_accuracy = float(
            accuracy_score(epoch_validation_targets, validation_predictions)
        )
        validation_macro_f1 = float(
            f1_score(
                epoch_validation_targets,
                validation_predictions,
                average="macro",
                zero_division=0,
            )
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "train_accuracy": epoch_train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "validation_macro_f1": validation_macro_f1,
            }
        )
        if validation_macro_f1 > best_validation_macro_f1 + 1e-4:
            best_validation_macro_f1 = validation_macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("training completed without producing a model checkpoint")
    model.load_state_dict(best_state)
    test_loss, evaluated_targets, test_probabilities = _evaluate(
        model, test_loader, evaluation_loss, device
    )
    test_predictions = test_probabilities.argmax(axis=1)
    category_count = len(encoder.classes_)
    majority_class = int(np.bincount(train_targets).argmax())
    majority_accuracy = float((evaluated_targets == majority_class).mean())
    top_k = min(3, category_count)
    metrics = ClassificationMetrics(
        test_loss=test_loss,
        accuracy=float(accuracy_score(evaluated_targets, test_predictions)),
        macro_f1=float(
            f1_score(
                evaluated_targets,
                test_predictions,
                average="macro",
                zero_division=0,
            )
        ),
        weighted_f1=float(
            f1_score(
                evaluated_targets,
                test_predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        top_3_accuracy=float(
            np.equal(
                test_probabilities.argsort(axis=1)[:, -top_k:],
                evaluated_targets[:, None],
            )
            .any(axis=1)
            .mean()
        ),
        correct_papers=int((evaluated_targets == test_predictions).sum()),
        test_papers=len(test_indices),
        validation_papers=len(validation_indices),
        training_papers=len(train_indices),
        categories=category_count,
        excluded_papers=excluded_papers,
        best_epoch=best_epoch,
        untrained_validation_accuracy=untrained_validation_accuracy,
        majority_test_accuracy=majority_accuracy,
        uniform_random_expected_accuracy=1.0 / category_count,
        expected_calibration_error=_expected_calibration_error(
            evaluated_targets, test_probabilities
        ),
    )
    per_category = classification_report(
        evaluated_targets,
        test_predictions,
        labels=np.arange(category_count),
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    high_confidence_mistakes, confusion_pairs = _error_analysis(
        test_examples,
        evaluated_targets,
        test_probabilities,
        encoder,
    )
    configuration = {
        "seed": seed,
        "device": str(device),
        "minimum_class_count": minimum_class_count,
        "maximum_epochs": epochs,
        "batch_size": batch_size,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "max_features": max_features,
        "hidden_neurons": hidden_neurons,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "early_stopping_patience": patience,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "features": "title and abstract word plus character TF-IDF",
        "label": "primary_category",
        "class_weighting": "square-root inverse training frequency",
    }
    return ClassificationRun(
        model=model.to("cpu"),
        vectorizer=vectorizer,
        encoder=encoder,
        metrics=metrics,
        history=history,
        per_category=per_category,
        confusion_pairs=confusion_pairs,
        high_confidence_mistakes=high_confidence_mistakes,
        selective_accuracy=_selective_accuracy(
            evaluated_targets, test_probabilities
        ),
        configuration=configuration,
    )


def _plot_history(history: list[dict[str, float | int]], destination: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to save learning curves; install '.[ml]'"
        ) from exc

    epochs = [int(item["epoch"]) for item in history]
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot(epochs, [item["train_loss"] for item in history], label="train")
    loss_axis.plot(
        epochs,
        [item["validation_loss"] for item in history],
        label="validation",
    )
    loss_axis.set(title="Loss by epoch", xlabel="Epoch", ylabel="Cross-entropy")
    loss_axis.legend()
    loss_axis.grid(alpha=0.3)

    accuracy_axis.plot(
        epochs,
        [item["train_accuracy"] for item in history],
        label="train accuracy",
    )
    accuracy_axis.plot(
        epochs,
        [item["validation_accuracy"] for item in history],
        label="validation accuracy",
    )
    accuracy_axis.plot(
        epochs,
        [item["validation_macro_f1"] for item in history],
        label="validation macro-F1",
    )
    accuracy_axis.set(
        title="Generalization by epoch",
        xlabel="Epoch",
        ylabel="Score",
        ylim=(0, 1),
    )
    accuracy_axis.legend()
    accuracy_axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def save_run(run: ClassificationRun, output_dir: Path) -> None:
    """Persist model, preprocessing, labels, metrics, curves, and error analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": run.model.state_dict(),
            "input_size": run.model.layers[0].in_features,
            "hidden_neurons": run.model.layers[0].out_features,
            "category_count": len(run.encoder.classes_),
            "configuration": run.configuration,
        },
        output_dir / "model.pt",
    )
    with (output_dir / "vectorizer.pkl").open("wb") as handle:
        pickle.dump(run.vectorizer, handle)
    (output_dir / "labels.json").write_text(
        json.dumps(run.encoder.classes_.tolist(), indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "metrics": asdict(run.metrics),
        "configuration": run.configuration,
        "history": run.history,
        "selective_accuracy": run.selective_accuracy,
        "confusion_pairs": run.confusion_pairs,
        "high_confidence_mistakes": run.high_confidence_mistakes,
        "per_category": run.per_category,
        "limitations": [
            "Confidence is model output, not certainty.",
            "Metrics estimate performance only for papers similar to this dataset.",
            "Categories excluded for low sample count cannot be predicted.",
            "Primary-category labels may omit valid secondary categories.",
        ],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    _plot_history(run.history, output_dir / "learning_curves.png")


def load_classifier(
    model_dir: Path,
) -> tuple[CategoryNetwork, FeatureUnion, np.ndarray]:
    """Load trusted artifacts produced by :func:`save_run` for CPU inference."""
    required = ("model.pt", "vectorizer.pkl", "labels.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"classifier artifact directory {model_dir} is missing: "
            + ", ".join(missing)
        )
    checkpoint = torch.load(
        model_dir / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    configuration = checkpoint["configuration"]
    model = CategoryNetwork(
        int(checkpoint["input_size"]),
        int(checkpoint["category_count"]),
        hidden_neurons=int(checkpoint["hidden_neurons"]),
        dropout=float(configuration["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with (model_dir / "vectorizer.pkl").open("rb") as handle:
        vectorizer = pickle.load(handle)
    labels = np.asarray(
        json.loads((model_dir / "labels.json").read_text(encoding="utf-8")),
        dtype=object,
    )
    if len(labels) != int(checkpoint["category_count"]):
        raise ValueError("labels.json category count does not match model.pt")
    return model, vectorizer, labels


@torch.inference_mode()
def predict_paper(
    model_dir: Path,
    *,
    title: str,
    abstract: str,
    top_k: int = 3,
) -> list[dict[str, float | str]]:
    """Return ranked category probabilities for one paper."""
    if not title.strip() or not abstract.strip():
        raise ValueError("title and abstract must both be non-empty")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    model, vectorizer, labels = load_classifier(model_dir)
    features = _as_tensor(vectorizer.transform([f"{title.strip()}\n\n{abstract.strip()}"]))
    probabilities = torch.softmax(model(features), dim=1).squeeze(0).numpy()
    ranked = probabilities.argsort()[-min(top_k, len(labels)) :][::-1]
    return [
        {"category": str(labels[index]), "probability": float(probabilities[index])}
        for index in ranked
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an arXiv primary-category classifier."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/papers.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/category_classifier"),
    )
    parser.add_argument("--min-class-count", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=8_000)
    parser.add_argument("--hidden-neurons", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps", "auto"),
        default="cpu",
    )
    args = parser.parse_args()

    run = train_and_evaluate(
        args.dataset,
        minimum_class_count=args.min_class_count,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        max_features=args.max_features,
        hidden_neurons=args.hidden_neurons,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device_name=args.device,
    )
    save_run(run, args.output_dir)
    metrics = run.metrics
    print(f"Untouched-test accuracy: {metrics.accuracy:.1%}")
    print(f"Untouched-test macro-F1: {metrics.macro_f1:.1%}")
    print(f"Untouched-test top-3 accuracy: {metrics.top_3_accuracy:.1%}")
    print(
        f"Correct: {metrics.correct_papers}/{metrics.test_papers}; "
        f"best epoch: {metrics.best_epoch}"
    )
    print(
        f"Papers: {metrics.training_papers} train, "
        f"{metrics.validation_papers} validation, {metrics.test_papers} test; "
        f"{metrics.categories} categories, {metrics.excluded_papers} excluded"
    )
    print(f"Artifacts: {args.output_dir}")


def predict_main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict an arXiv primary category with a trained classifier."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/category_classifier"),
    )
    parser.add_argument("--title", required=True)
    parser.add_argument("--abstract", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    predictions = predict_paper(
        args.model_dir,
        title=args.title,
        abstract=args.abstract,
        top_k=args.top_k,
    )
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
