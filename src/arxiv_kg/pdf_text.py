"""PDF-to-text and prompt-selection utilities for Tutorial 2."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

SECTION_NAMES = ("method", "methodology", "experiments", "results", "limitations", "conclusion")
SECTION_RE = re.compile(
    rf"(?im)^\s*(?:\d+(?:\.\d+)*\s+)?({'|'.join(SECTION_NAMES)})s?\s*$"
)


def extract_pdf_to_text(pdf_path: str | Path, text_path: str | Path) -> Path:
    pdf_path = Path(pdf_path)
    text_path = Path(text_path)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = text_path.with_suffix(text_path.suffix + ".part")

    pages: list[str] = []
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text", sort=True)
            pages.append(f"\n=== PAGE {page_number} ===\n{page_text}")

    joined = "".join(pages)
    if len(re.sub(r"\s+", "", joined)) < 200:
        raise ValueError("Very little text was extracted; PDF may be scanned or unusual")
    temporary_path.write_text(joined, encoding="utf-8")
    temporary_path.replace(text_path)
    return text_path


def _head_tail(full_text: str, budget: int) -> str:
    omission = "\n\n[... middle omitted for prompt budget ...]\n\n"
    if budget <= len(omission):
        return full_text[:budget]
    content_budget = budget - len(omission)
    first = int(content_budget * 0.75)
    last = content_budget - first
    return (
        full_text[:first]
        + omission
        + full_text[-last:]
    )


def _section_aware_selection(full_text: str, budget: int) -> str | None:
    matches = list(SECTION_RE.finditer(full_text))
    if not matches:
        return None
    sections: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(full_text)
        sections.append(full_text[match.start():end].strip())
    separator = "\n\n"
    selected = separator.join(sections)
    if len(selected) <= budget:
        return selected

    separator_budget = len(separator) * (len(sections) - 1)
    if separator_budget >= budget:
        headings = separator.join(section.splitlines()[0] for section in sections)
        return headings[:budget]

    content_budget = budget - separator_budget
    per_section, extra = divmod(content_budget, len(sections))
    selected_sections = [
        section[: per_section + (index < extra)]
        for index, section in enumerate(sections)
    ]
    return separator.join(selected_sections)


def select_text_for_llm(
    *, title: str, abstract: str, full_text: str | None, max_characters: int = 70_000
) -> str:
    """Bound model input; prefer useful sections and fall back to head/tail."""

    if max_characters < 1:
        raise ValueError("max_characters must be positive")
    header = f"TITLE\n{title}\n\nABSTRACT\n{abstract}\n"
    if len(header) > max_characters:
        raise ValueError("max_characters must fit the complete title and abstract")
    if not full_text:
        return header
    remaining = max_characters - len(header)
    full_prefix = "\nFULL PAPER TEXT\n"
    if len(full_prefix) + len(full_text) <= remaining:
        return header + full_prefix + full_text

    section_prefix = "\nPRIORITIZED PAPER SECTIONS\n"
    fallback_prefix = "\nSELECTED FULL PAPER TEXT\n"
    has_priority_sections = SECTION_RE.search(full_text) is not None
    if has_priority_sections:
        section_budget = remaining - len(section_prefix)
        if section_budget <= 0:
            return header
        selected = _section_aware_selection(full_text, section_budget)
        prefix = section_prefix
    else:
        fallback_budget = remaining - len(fallback_prefix)
        if fallback_budget <= 0:
            return header
        selected = _head_tail(full_text, fallback_budget)
        prefix = fallback_prefix
    return header + prefix + selected
