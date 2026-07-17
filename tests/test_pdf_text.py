import pymupdf
import pytest

from arxiv_kg.pdf_text import extract_pdf_to_text, select_text_for_llm


def test_pdf_text_extraction_adds_page_markers(tmp_path):
    pdf = tmp_path / "tiny.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(
        page.rect + (72, 72, -72, -72),
        "A tiny machine learning paper for an offline unit test. " * 30,
    )
    document.save(pdf)
    document.close()

    output = extract_pdf_to_text(pdf, tmp_path / "tiny.txt")
    text = output.read_text(encoding="utf-8")
    assert "=== PAGE 1 ===" in text
    assert "machine learning" in text


def test_pdf_text_extraction_rejects_nearly_empty_pdf(tmp_path):
    pdf = tmp_path / "empty.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(pdf)
    document.close()

    with pytest.raises(ValueError, match="Very little text"):
        extract_pdf_to_text(pdf, tmp_path / "empty.txt")


def test_section_aware_selector_prioritizes_useful_sections():
    full_text = (
        "INTRODUCTION\n" + "background " * 200 + "\n"
        "METHODS\nWe train a diffusion model.\n"
        "EXPERIMENTS\nWe evaluate on CIFAR-10.\n"
        "CONCLUSION\nThe method improves accuracy.\n"
    )

    selected = select_text_for_llm(
        title="Paper",
        abstract="Abstract.",
        full_text=full_text,
        max_characters=350,
    )

    assert len(selected) <= 350
    assert "PRIORITIZED PAPER SECTIONS" in selected
    assert "diffusion model" in selected
    assert "background background" not in selected


def test_section_selector_preserves_each_priority_section_under_tight_budget():
    full_text = (
        "METHODS\n" + "method detail " * 100 + "\n"
        "EXPERIMENTS\n" + "experiment detail " * 100 + "\n"
        "LIMITATIONS\nSmall fictional sample.\n"
        "CONCLUSION\nThe method needs more evaluation.\n"
    )

    selected = select_text_for_llm(
        title="Paper",
        abstract="Abstract.",
        full_text=full_text,
        max_characters=400,
    )

    assert len(selected) <= 400
    assert "METHODS" in selected
    assert "EXPERIMENTS" in selected
    assert "LIMITATIONS" in selected
    assert "CONCLUSION" in selected


def test_selector_falls_back_to_head_and_tail_without_sections():
    full_text = "BEGIN " + "middle " * 200 + " END"

    selected = select_text_for_llm(
        title="Paper",
        abstract="Abstract.",
        full_text=full_text,
        max_characters=250,
    )

    assert len(selected) <= 250
    assert "SELECTED FULL PAPER TEXT" in selected
    assert "BEGIN" in selected
    assert "END" in selected
    assert "middle omitted for prompt budget" in selected


def test_selector_rejects_nonpositive_budget():
    with pytest.raises(ValueError, match="must be positive"):
        select_text_for_llm(
            title="Paper", abstract="Abstract.", full_text=None, max_characters=0
        )


def test_selector_rejects_budget_that_cannot_hold_title_and_abstract():
    with pytest.raises(ValueError, match="title and abstract"):
        select_text_for_llm(
            title="Long title",
            abstract="Long abstract",
            full_text=None,
            max_characters=10,
        )


def test_selector_never_exceeds_budget_when_only_one_character_remains():
    header = "TITLE\nT\n\nABSTRACT\nA\n"
    max_characters = len(header) + 1

    selected = select_text_for_llm(
        title="T",
        abstract="A",
        full_text="METHODS\n" + "x" * 100,
        max_characters=max_characters,
    )

    assert selected == header
    assert len(selected) <= max_characters
