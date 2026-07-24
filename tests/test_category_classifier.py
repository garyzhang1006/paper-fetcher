import json

import numpy as np
import pytest

pytest.importorskip("torch")

from arxiv_kg.category_classifier import (
    CategoryNetwork,
    load_labeled_papers,
    load_paper_examples,
    predict_paper,
    save_run,
    stratified_three_way_split,
    train_and_evaluate,
)


def write_records(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def paper(index, category, terms):
    return {
        "arxiv_id": f"2607.{index:05d}",
        "title": f"{terms} study {index}",
        "abstract": f"{terms} methods results evidence {terms}",
        "primary_category": category,
    }


def test_load_labeled_papers_excludes_rare_categories(tmp_path):
    dataset = tmp_path / "papers.jsonl"
    records = [
        paper(1, "cs.LG", "learning model"),
        paper(2, "cs.LG", "learning data"),
        paper(3, "cs.LG", "learning algorithm"),
        paper(4, "q-bio.QM", "rare topic"),
    ]
    write_records(dataset, records)

    texts, labels, excluded = load_labeled_papers(
        dataset, minimum_class_count=3
    )

    assert len(texts) == 3
    assert labels == ["cs.LG", "cs.LG", "cs.LG"]
    assert excluded == 1


def test_load_paper_examples_reports_bad_json_line(tmp_path):
    dataset = tmp_path / "papers.jsonl"
    dataset.write_text('{"title": "valid"}\n{bad json}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 1: abstract"):
        load_paper_examples(dataset, minimum_class_count=3)


def test_load_paper_examples_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="paper dataset not found"):
        load_paper_examples(tmp_path / "missing.jsonl", minimum_class_count=3)


def test_stratified_three_way_split_is_disjoint_and_covers_each_class():
    targets = np.repeat(np.arange(3), 10)
    train, validation, test = stratified_three_way_split(
        targets,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=42,
    )

    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert set(np.concatenate([train, validation, test])) == set(range(30))
    for split in (train, validation, test):
        assert set(targets[split]) == {0, 1, 2}


def test_category_network_returns_one_logit_per_category():
    import torch

    model = CategoryNetwork(100, 4, hidden_neurons=16)
    logits = model(torch.zeros((3, 100)))

    assert tuple(logits.shape) == (3, 4)


def test_train_evaluate_and_save_artifacts(tmp_path):
    dataset = tmp_path / "papers.jsonl"
    records = [
        paper(index, "cs.LG", "neural learning gradient model")
        for index in range(12)
    ]
    records += [
        paper(index + 100, "math.CO", "graph combinatorics theorem proof")
        for index in range(12)
    ]
    write_records(dataset, records)

    run = train_and_evaluate(
        dataset,
        minimum_class_count=5,
        epochs=3,
        patience=2,
        batch_size=8,
        max_features=100,
        hidden_neurons=16,
        seed=7,
    )
    output_dir = tmp_path / "classifier"
    save_run(run, output_dir)

    assert run.metrics.categories == 2
    assert run.metrics.training_papers == 16
    assert run.metrics.validation_papers == 4
    assert run.metrics.test_papers == 4
    assert 0 <= run.metrics.accuracy <= 1
    assert 0 <= run.metrics.macro_f1 <= 1
    assert len(run.history) <= 3
    assert (output_dir / "model.pt").is_file()
    assert (output_dir / "vectorizer.pkl").is_file()
    assert (output_dir / "labels.json").is_file()
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "learning_curves.png").is_file()
    predictions = predict_paper(
        output_dir,
        title="Neural gradient learning",
        abstract="A learning model trained with gradients",
        top_k=2,
    )
    assert len(predictions) == 2
    assert {item["category"] for item in predictions} == {"cs.LG", "math.CO"}
    assert all(0 <= item["probability"] <= 1 for item in predictions)


def test_train_and_evaluate_rejects_invalid_class_threshold(tmp_path):
    with pytest.raises(ValueError, match="at least 3"):
        train_and_evaluate(
            tmp_path / "missing.jsonl",
            minimum_class_count=2,
        )
