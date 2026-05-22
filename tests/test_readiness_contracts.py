from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ci_workflow_targets_v2_release_branches():
    workflow = (REPO_ROOT / ".github/workflows/cicd-workflow.yml").read_text()

    assert 'pull_request:\n    branches: [ "v2", "v2_sam" ]' in workflow
    assert 'push:\n    branches: [ "v2", "v2_sam" ]' in workflow
    assert "github.ref == 'refs/heads/v2'" in workflow
    assert "dawidd6/action-download-artifact@v6" in workflow
    assert 'branches: [ "main" ]' not in workflow


def test_readme_orients_users_to_site_config_and_tutorials():
    readme = (REPO_ROOT / "README.md").read_text()

    assert "~/.frequensolve/site.toml" in readme
    assert "fs.Site()" in readme
    assert "docs/source/tutorials/index.rst" in readme
    assert "examples/tutorials" in readme
    assert "AWSSite" in readme
