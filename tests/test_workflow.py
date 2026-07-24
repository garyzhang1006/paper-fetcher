from pathlib import Path


def test_manual_workflow_tests_before_bounded_fetch_and_persists_state():
    workflow = (
        Path(__file__).parents[1] / ".github/workflows/daily-arxiv-fetch.yml"
    ).read_text()

    assert "schedule:" not in workflow
    assert "timezone:" not in workflow
    assert "workflow_dispatch:" in workflow
    assert 'python-version: "3.14"' in workflow
    assert "python -m pip install '.[test,notebooks]'" in workflow
    assert "paper-fetcher-daily" in workflow
    assert "--max-results 200" in workflow
    assert "--revision-max-results 200" in workflow
    assert "persist-credentials: false" in workflow
    assert "git add data/arxiv_kg.sqlite3" in workflow
    test_index = workflow.index("python -m pytest")
    import_index = workflow.index('python -c "import arxiv_kg"')
    notebook_index = workflow.index("python -m jupyter nbconvert")
    fetch_index = workflow.index("paper-fetcher-daily")

    assert 'RUN_LLM: "false"' in workflow
    assert "--execute notebooks/02_feature_extractor.ipynb" in workflow
    assert '--output-dir "${RUNNER_TEMP}"' in workflow
    assert import_index < test_index < notebook_index < fetch_index
    assert workflow.index("paper-fetcher-daily") < workflow.index("GITHUB_TOKEN:")


def test_push_ci_installs_ml_dependencies_and_runs_full_suite():
    workflow = (
        Path(__file__).parents[1] / ".github/workflows/tests.yml"
    ).read_text()

    assert "push:" in workflow
    assert "pull_request:" in workflow
    assert 'python-version: "3.13"' in workflow
    assert "python -m pip install '.[test,ml]'" in workflow
    assert "python -m pytest" in workflow
