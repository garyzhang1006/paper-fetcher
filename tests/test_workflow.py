from pathlib import Path


def test_daily_workflow_tests_before_bounded_agent_and_persists_state():
    workflow = (
        Path(__file__).parents[1] / ".github/workflows/daily-arxiv-fetch.yml"
    ).read_text()

    assert "schedule:" in workflow
    assert 'cron: "17 13 * * *"' in workflow
    assert "workflow_dispatch:" in workflow
    assert 'python-version: "3.14"' in workflow
    assert "python -m pip install '.[test,notebooks]'" in workflow
    assert "paper-fetcher-agent" in workflow
    assert "--dataset dataset/papers.jsonl" in workflow
    assert "--sources config/sources.json" in workflow
    assert "--output-dir output/knowledge_graph" in workflow
    assert "--max-results 200" in workflow
    assert "--revision-max-results 200" in workflow
    assert "persist-credentials: false" in workflow
    assert "permissions:\n      contents: read" in workflow
    assert "permissions:\n      contents: write" in workflow
    assert "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803" in workflow
    assert "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1" in workflow
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in workflow
    assert "actions/download-artifact@634f93cb2916e3fdff6788551b99b062d0335ce0" in workflow
    assert "continue-on-error: true" in workflow
    assert "needs.collect.outputs.agent_outcome == 'failure'" in workflow
    assert "git add data/arxiv_kg.sqlite3" in workflow
    test_index = workflow.index("python -m pytest")
    import_index = workflow.index('python -c "import arxiv_kg"')
    notebook_index = workflow.index("python -m jupyter nbconvert")
    fetch_index = workflow.index("paper-fetcher-agent")

    assert 'RUN_LLM: "false"' in workflow
    assert "--execute notebooks/02_feature_extractor.ipynb" in workflow
    assert '--output-dir "${RUNNER_TEMP}"' in workflow
    assert import_index < test_index < notebook_index < fetch_index
    assert workflow.index("paper-fetcher-agent") < workflow.index("GITHUB_TOKEN:")
    assert "git add data/arxiv_kg.sqlite3 output/knowledge_graph" in workflow


def test_push_ci_installs_ml_dependencies_and_runs_full_suite():
    workflow = (
        Path(__file__).parents[1] / ".github/workflows/tests.yml"
    ).read_text()

    assert "push:" in workflow
    assert "pull_request:" in workflow
    assert 'python-version: "3.13"' in workflow
    assert "python -m pip install '.[test,ml]'" in workflow
    assert "python -m pytest" in workflow
