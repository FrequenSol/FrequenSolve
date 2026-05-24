from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ci_workflow_avoids_duplicate_pr_and_push_runs():
    workflow = (REPO_ROOT / ".github/workflows/cicd-workflow.yml").read_text()

    assert 'pull_request:\n    branches: [ "v2" ]' in workflow
    assert 'push:\n    branches: [ "v2" ]' in workflow
    assert 'branches: [ "v2", "v2_sam" ]' not in workflow
    assert "dawidd6/action-download-artifact@v6" in workflow
    assert 'branches: [ "main" ]' not in workflow


def test_ci_workflow_does_not_deploy_legacy_docs_host():
    workflow = (REPO_ROOT / ".github/workflows/cicd-workflow.yml").read_text()

    assert "deploy-docs:" not in workflow
    assert "docs/host" not in workflow
    assert "terraform init" not in workflow
    assert "make deploy-all" not in workflow


def test_readme_orients_users_to_site_config_and_tutorials():
    readme = (REPO_ROOT / "README.md").read_text()

    assert "~/.frequensolve/site.toml" in readme
    assert "fs.Site()" in readme
    assert "docs/source/tutorials/index.rst" in readme
    assert "examples/tutorials" in readme
    assert "AWSSite" in readme


def test_readme_points_published_docs_to_cloud_amplify_docs_app():
    readme = (REPO_ROOT / "README.md").read_text()

    assert "FrequenSol/cloud-amplify" in readme
    assert "Publish Python Docs" in readme
    assert "/python/<version>/" in readme
    assert "former `docs/host` Terraform stack" in readme
    assert "destroyed and removed" in readme
    assert not (REPO_ROOT / "docs/host").exists()
