"""Fictional records so Notebook 2 runs without internet."""

from __future__ import annotations

from datetime import UTC, datetime

from .db import Database
from .models import PaperRecord


def seed_demo_papers(db: Database) -> dict[str, int]:
    papers = [
        PaperRecord(
            arxiv_id="9999.00001",
            versioned_id="9999.00001v1",
            version=1,
            title="Graph-Guided Diffusion Policies for Household Robot Manipulation",
            abstract=(
                "We study robot manipulation with multimodal demonstrations. "
                "Our method combines a graph neural network with a diffusion model "
                "and is evaluated by task success rate on a small household benchmark."
            ),
            authors=["Ada Student", "Grace Mentor"],
            categories=["cs.RO", "cs.LG"],
            primary_category="cs.RO",
            published_at=datetime(2026, 6, 1, tzinfo=UTC),
            updated_at=datetime(2026, 6, 1, tzinfo=UTC),
            abs_url="https://arxiv.org/abs/9999.00001v1",
        ),
        PaperRecord(
            arxiv_id="9999.00002",
            versioned_id="9999.00002v1",
            version=1,
            title="Contrastive Transformers for Data-Efficient Image Classification",
            abstract=(
                "This paper proposes contrastive learning for transformer-based image "
                "classification. Experiments on CIFAR-10 and ImageNet report accuracy "
                "and F1 score improvements over a supervised baseline."
            ),
            authors=["Katherine Learner", "Ada Student"],
            categories=["cs.CV", "cs.LG"],
            primary_category="cs.CV",
            published_at=datetime(2026, 6, 2, tzinfo=UTC),
            updated_at=datetime(2026, 6, 2, tzinfo=UTC),
            abs_url="https://arxiv.org/abs/9999.00002v1",
        ),
    ]
    result = {"inserted": 0, "updated": 0, "unchanged": 0}
    for paper in papers:
        result[db.upsert_paper(paper)] += 1
    return result
