"""Version-matched installation instructions for the native starter guide."""

import re

_CANONICAL_RELEASE = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?P<prerelease>(?:a|b|rc)(?:0|[1-9][0-9]*))?"
)


def starter_installation_rst(release: str) -> str:
    """Generate commands only for canonical final or prerelease versions."""
    match = _CANONICAL_RELEASE.fullmatch(release)
    if match is None:
        return (
            "This guide describes a local or unreleased package version. No public "
            "wheel or package-index release is assumed. Obtain the exact matching "
            "approved project wheel from your project administrator and install "
            "that wheel with the ``cloud,visual`` extras in this environment. "
            "Alternatively, use an available published package and its matching "
            "versioned guide."
        )
    if match.group("prerelease"):
        wheel = (
            "https://github.com/FrequenSol/FrequenSolve/releases/download/"
            f"v{release}/frequensolve-{release}-py3-none-any.whl"
        )
        requirement = f"frequensolve[cloud,visual] @ {wheel}"
        explanation = (
            "This is a prerelease guide. Install its matching wheel from the "
            "official public GitHub release; dependencies are resolved from "
            "your normal Python package index. No TestPyPI index configuration "
            "is needed."
        )
        fallback = (
            "If the matching GitHub asset is not available yet, publication "
            "is incomplete: wait for that release or obtain its exact approved "
            "wheel from your project administrator. Do not substitute an older "
            "wheel or change the version in the URL."
        )
    else:
        requirement = f"frequensolve[cloud,visual]=={release}"
        explanation = "Install the exact stable package version for this guide:"
        fallback = (
            "If the exact version is unavailable from your configured package "
            "index, obtain the matching approved project wheel from your "
            "project administrator, or use an available package and its "
            "matching versioned guide."
        )
    return (
        f"{explanation}\n\n.. code-block:: console\n\n"
        f'   python -m pip install "{requirement}"\n\n{fallback}'
    )


def render_starter_installation(app, docname, source):
    """Use the final Sphinx config, including publication release overrides."""
    if docname == "acoustic_starter":
        source[0] = source[0].replace(
            ".. frequensolve-starter-installation",
            starter_installation_rst(app.config.release),
        )
