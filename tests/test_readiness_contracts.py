import importlib.util
import json
import runpy
import tempfile
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

    assert "python -m pip install frequensolve" in readme
    assert "python -m pip install -e ." in readme
    assert "~/.frequensolve/site.toml" in readme
    assert "fs.Site()" in readme
    assert "docs/source/tutorials/index.rst" in readme
    assert "examples/tutorials" in readme
    assert "AWSSite" in readme


def test_example_notebooks_use_configured_site_factory():
    constructor_patterns = (
        "fs.LocalSite(",
        "fs.AWSSite(",
        "fs.Stampede3Site(",
        "fs.SlurmSite(",
        "fs.SlurmRunConfig(",
        "fs.SlurmSiteConfig(",
    )
    output_patterns = (
        "LocalSite:",
        "AWSSite:",
        "Stampede3Site:",
        "SlurmSite:",
    )
    offenders = []
    notebook_paths = [
        REPO_ROOT / "examples/ex01_simple.ipynb",
        *sorted((REPO_ROOT / "examples/tutorials").rglob("*.ipynb")),
    ]

    for notebook_path in notebook_paths:
        notebook = json.loads(notebook_path.read_text())
        for cell_index, cell in enumerate(notebook.get("cells", []), start=1):
            source = "".join(cell.get("source", []))
            for pattern in constructor_patterns:
                if pattern in source:
                    offenders.append(
                        f"{notebook_path.relative_to(REPO_ROOT)} cell {cell_index}: "
                        f"{pattern}"
                    )
            outputs = json.dumps(cell.get("outputs", []))
            for pattern in output_patterns:
                if pattern in outputs:
                    offenders.append(
                        f"{notebook_path.relative_to(REPO_ROOT)} cell {cell_index} "
                        f"output: {pattern}"
                    )

    assert not offenders


def test_readme_quickstart_uses_current_project_owned_simulation_api():
    import frequensolve as fs

    readme = (REPO_ROOT / "README.md").read_text()
    assert "project.new_simulation(" in readme
    assert 'SeismicSimulation(name="simple_acoustic")' not in readme

    u = fs.ureg
    with tempfile.TemporaryDirectory() as tmp:
        project = fs.Project(name="quickstart", path=Path(tmp) / "quickstart")
        sim = project.new_simulation(
            name="simple_acoustic",
            physics="acoustic",
            dimension=2,
            units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
        )
        model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
        model.add_surface(name="top", depth=0.0 * u.km)
        model.add_layer(
            name="layer",
            properties={"Vp": 1.5 * u.km / u.s, "Rho": 2.2 * u.g / u.cm**3},
        )
        model.add_surface(name="bottom", depth=0.5 * u.km)

        sim += model
        sim += model.hex_mesh_generator([8, 4])

        assert project.save().exists()


def test_readme_points_published_docs_to_cloud_amplify_docs_app():
    readme = (REPO_ROOT / "README.md").read_text()

    assert "FrequenSol/cloud-amplify" in readme
    assert "Publish Python Docs" in readme
    assert "/python/<version>/" in readme
    assert "former `docs/host` Terraform stack" in readme
    assert "destroyed and removed" in readme
    assert not (REPO_ROOT / "docs/host").exists()


def test_docs_and_examples_reference_canonical_frequensol_domains():
    checked_paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "docs/source",
        REPO_ROOT / "src/frequensolve/orchestrator/sites/aws",
        REPO_ROOT / "tests/test_auth.py",
        REPO_ROOT / "tests/test_s3_upload.py",
        REPO_ROOT / "tests/test_awssite_domain.py",
    ]
    legacy_domains = ("frequensolve.app", "frequensol.app")
    text_paths = []

    for path in checked_paths:
        if path.is_dir():
            text_paths.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.suffix in {".py", ".rst", ".md", ".toml", ".txt"}
            )
        else:
            text_paths.append(path)

    offenders = []
    for path in sorted(text_paths):
        text = path.read_text()
        for legacy_domain in legacy_domains:
            if legacy_domain in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {legacy_domain}")

    assert not offenders
    assert 'domain = "app.frequensol.com"' in (REPO_ROOT / "README.md").read_text()
    assert (
        'Documentation = "https://docs.frequensol.com"'
        in (REPO_ROOT / "pyproject.toml").read_text()
    )


def test_contributing_and_changelog_match_current_repo_workflows():
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text()
    releasing = (REPO_ROOT / "RELEASING.md").read_text()
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text()
    installation = (REPO_ROOT / "docs/source/installation.rst").read_text()

    assert not (REPO_ROOT / "docs/source/contributing.rst").exists()
    assert not (REPO_ROOT / "docs/source/changelog.rst").exists()
    assert "Python 3.10 through 3.14" in contributing
    assert "make test" in contributing
    assert "make generate_reference_images" in contributing
    assert "Publish Python Docs" in contributing
    assert "docs/host" in contributing
    assert "python -m pip install frequensolve" in installation
    assert "python -m pip install -e ." in installation
    assert "Publish PyPI" in releasing
    assert "trusted publishing" in releasing
    assert "Do not add PyPI passwords or API tokens" in releasing
    assert "0.1.1" not in changelog
    assert "0.1.0" not in changelog
    assert "0.0.1" in changelog


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
    assert "Looking for other docs?" in custom_js
    assert "fs-docs-home-link" in custom_js
    assert 'homeLink.href = "/"' in custom_js
    assert ".fs-docs-version-selector" in custom_css
    assert ".fs-docs-home-link" in custom_css


def test_sphinx_docs_use_frequensol_docs_site_palette():
    conf = runpy.run_path(str(REPO_ROOT / "docs/source/conf.py"))
    custom_css = (REPO_ROOT / "docs/source/_static/custom.css").read_text()

    assert conf["html_theme_options"]["style_nav_header_background"] == "#0a090c"
    assert "--fs-near-black: #0a090c" in custom_css
    assert "--fs-surface: #121212" in custom_css
    assert "--fs-blue: #45588c" in custom_css
    assert "--fs-gold: #e1b07e" in custom_css
    assert "--fs-gold-soft" in custom_css
    assert ".wy-nav-side" in custom_css
    assert ".wy-nav-content" in custom_css
    assert ".wy-menu-vertical li.current ul a" in custom_css
    assert ".rst-content h1" in custom_css
    assert ".rst-content p a" in custom_css
    assert "color: var(--fs-blue-light);" in custom_css
    assert ".rst-content table.docutils th" in custom_css
    assert "@media screen and (min-width: 769px)" in custom_css
    assert "overflow: visible !important;" in custom_css
    assert "table-layout: fixed;" in custom_css
    assert "@media screen and (max-width: 768px)" in custom_css
    assert "-webkit-overflow-scrolling: touch;" in custom_css
    assert ".rst-content dl dt.sig" in custom_css
    assert "overflow-wrap: anywhere;" in custom_css
    assert "html.writer-html5 .rst-content dl.field-list" in custom_css
    assert "box-shadow: inset 3px 0 0 var(--fs-gold);" in custom_css
    assert "border-bottom: 1px solid var(--fs-gold-soft);" in custom_css
    assert "border-top-color: var(--fs-gold)" in custom_css


def test_sphinx_docs_use_transparent_frequensol_logo():
    conf = runpy.run_path(str(REPO_ROOT / "docs/source/conf.py"))
    logo_relpath = conf["html_logo"]
    logo_path = REPO_ROOT / "docs/source" / logo_relpath
    logo_bytes = logo_path.read_bytes()

    assert logo_relpath == "_static/logo-transparent.png"
    assert logo_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert logo_bytes[25] in {4, 6}
