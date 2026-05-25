import importlib.util
import runpy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_versioneer_release():
    version_file = REPO_ROOT / "src/frequensolve/_version.py"
    spec = importlib.util.spec_from_file_location("frequensolve_version", version_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_versions()["version"]


def test_ci_workflow_avoids_duplicate_pr_and_push_runs():
    workflow = (REPO_ROOT / ".github/workflows/cicd-workflow.yml").read_text()

    assert 'pull_request:\n    branches: [ "v2" ]' in workflow
    assert 'push:\n    branches: [ "v2" ]' in workflow
    assert 'branches: [ "v2", "v2_sam" ]' not in workflow
    assert 'branches: [ "main" ]' not in workflow


def test_ci_workflow_runs_supported_python_matrix_on_node24_actions():
    workflow = (REPO_ROOT / ".github/workflows/cicd-workflow.yml").read_text()
    publish_workflow = (REPO_ROOT / ".github/workflows/publish-pypi.yml").read_text()

    assert 'python-version: ["3.10", "3.11", "3.12", "3.13", "3.14"]' in workflow
    assert "actions/checkout@v6" in workflow
    assert "actions/setup-python@v6" in workflow
    assert "codecov/codecov-action@v6" in workflow
    assert "actions/upload-artifact@v7" in workflow
    assert "actions/create-github-app-token@v3" in workflow
    assert "pre-commit/action@" not in workflow
    assert "tibdex/github-app-token" not in workflow
    assert "the-actions-org/workflow-dispatch" not in workflow
    assert "dawidd6/action-download-artifact" not in workflow
    assert "concurrency:" in workflow
    assert "docker-image-integration-" in workflow
    assert "cancel-in-progress: false" in workflow
    assert 'gh workflow run "$DOWNSTREAM_WORKFLOW"' in workflow
    assert (
        'dispatch_actor="${{ steps.generate-token.outputs.app-slug }}[bot]"' in workflow
    )
    assert "dispatch_started_at=" in workflow
    assert (
        'gh api "repos/$DOWNSTREAM_REPO/actions/workflows/$DOWNSTREAM_WORKFLOW/runs"'
        in workflow
    )
    assert '-f actor="$dispatch_actor"' in workflow
    assert '-f created=">=$dispatch_started_at"' in workflow
    assert "sort_by(.created_at)[]" in workflow
    assert "Expected exactly one downstream workflow run" not in workflow
    assert "gh run watch" in workflow
    assert "gh run download" in workflow
    assert "actions/checkout@v6" in publish_workflow
    assert "actions/setup-python@v6" in publish_workflow


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


def test_sphinx_docs_release_matches_package_version():
    conf = runpy.run_path(str(REPO_ROOT / "docs/source/conf.py"))
    package_release = _load_versioneer_release()

    assert conf["release"] == package_release
    assert conf["version"] == conf["_short_version"](package_release)


def test_sphinx_docs_include_published_version_selector_assets():
    conf = runpy.run_path(str(REPO_ROOT / "docs/source/conf.py"))
    custom_js = (REPO_ROOT / "docs/source/_static/custom.js").read_text()
    custom_css = (REPO_ROOT / "docs/source/_static/custom.css").read_text()

    assert "custom.js" in conf["html_js_files"]
    assert "custom.css" in conf["html_css_files"]
    assert "/python/docs-manifest.json" in custom_js
    assert "latestPath" in custom_js
    assert "fs-docs-version-selector" in custom_js
    assert "window.location.assign" in custom_js
    assert ".fs-docs-version-selector" in custom_css
