from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_dataset_fetcher():
    path = Path(__file__).parents[1] / "dataset/fetch_arxiv_dataset.py"
    spec = importlib.util.spec_from_file_location("dataset_fetcher", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def paper(identifier: str, category: str, abstract_words: int):
    return {
        "versioned_id": identifier,
        "primary_category": category,
        "title": "A clear and useful research paper title",
        "abstract": "word " * abstract_words,
        "categories": [category],
        "submitted_date": "2026-07-03",
    }


def test_curation_hits_target_and_preserves_each_category():
    fetcher = load_dataset_fetcher()
    papers = [paper(f"large-{i}", "cs.LG", i + 20) for i in range(10)]
    papers += [paper(f"small-{i}", "math.CO", i + 20) for i in range(2)]

    retained = fetcher.curate_papers(papers, target=6)

    assert len(retained) == 6
    assert {item["primary_category"] for item in retained} == {"cs.LG", "math.CO"}
    assert fetcher.curate_papers(papers, target=6) == retained


def test_withdrawn_paper_ranks_below_otherwise_weaker_paper():
    fetcher = load_dataset_fetcher()
    withdrawn = paper("withdrawn", "cs.LG", 250)
    withdrawn["title"] = "Withdrawn research result"
    active = paper("active", "cs.LG", 20)

    assert fetcher.curate_papers([withdrawn, active], target=1) == [active]


def test_astrophysics_categories_are_excluded_after_curation():
    fetcher = load_dataset_fetcher()
    papers = [
        paper("astro", "astro-ph.GA", 30),
        paper("condensed-matter", "cond-mat.mtrl-sci", 30),
        paper("economics", "econ.EM", 30),
        paper("machine-learning", "cs.LG", 30),
    ]

    retained = fetcher.exclude_primary_categories(papers)

    assert retained == [papers[3]]
