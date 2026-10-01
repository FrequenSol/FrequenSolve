"""Installation rendering follows the publication version, not package imports."""

from types import SimpleNamespace

import pytest

from docs.starter_install import (
    render_starter_installation,
    starter_installation_rst,
)


@pytest.mark.parametrize("version", ["0.6.4rc2", "1.2.0a1", "1.2.0b3"])
def test_prerelease_uses_matching_public_wheel_and_extras(version):
    rendered = starter_installation_rst(version)
    expected = (
        'python -m pip install "frequensolve[cloud,visual] @ '
        "https://github.com/FrequenSol/FrequenSolve/releases/download/"
        f'v{version}/frequensolve-{version}-py3-none-any.whl"'
    )
    assert expected in rendered
    assert "--index-url" not in rendered
    assert "not available yet" in rendered


def test_stable_uses_exact_package_index_release():
    rendered = starter_installation_rst("0.6.4")
    assert 'python -m pip install "frequensolve[cloud,visual]==0.6.4"' in rendered
    assert "github.com" not in rendered


@pytest.mark.parametrize(
    "version",
    [
        "0.6.4+3.g1234",
        "0.6.4.dev1",
        "0+unknown",
        "0.6.4rc2.dirty",
        '0.6.4";exit',
        "../0.6.4",
    ],
)
def test_unreleased_or_noncanonical_version_never_invents_install_target(version):
    rendered = starter_installation_rst(version)
    assert "approved project wheel" in rendered
    assert "pip install" not in rendered
    assert "https://" not in rendered


def test_source_hook_uses_resolved_sphinx_release_and_only_starter_document():
    app = SimpleNamespace(config=SimpleNamespace(release="0.6.4rc2"))
    source = ["Before\n\n.. frequensolve-starter-installation\n\nAfter"]
    render_starter_installation(app, "acoustic_starter", source)
    assert "v0.6.4rc2/frequensolve-0.6.4rc2-py3-none-any.whl" in source[0]
    assert source[0].startswith("Before\n") and source[0].endswith("\nAfter")
    other = [".. frequensolve-starter-installation"]
    render_starter_installation(app, "installation", other)
    assert other == [".. frequensolve-starter-installation"]
