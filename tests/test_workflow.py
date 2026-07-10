from pathlib import Path


def test_daily_workflow_tests_before_bounded_fetch_and_persists_state():
    workflow = (
        Path(__file__).parents[1] / ".github/workflows/daily-arxiv-fetch.yml"
    ).read_text()

    assert 'timezone: "America/Chicago"' in workflow
    assert "paper-fetcher-daily" in workflow
    assert "--max-results 200" in workflow
    assert "--revision-max-results 200" in workflow
    assert "persist-credentials: false" in workflow
    assert "git add data/arxiv_kg.sqlite3" in workflow
    assert workflow.index("python -m pytest") < workflow.index("paper-fetcher-daily")
    assert workflow.index("paper-fetcher-daily") < workflow.index("GITHUB_TOKEN:")
