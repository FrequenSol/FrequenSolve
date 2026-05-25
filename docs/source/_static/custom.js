(function () {
  const MANIFEST_PATH = "/python/docs-manifest.json";
  const CONTAINER_CLASS = "fs-docs-version-selector";

  function isRecord(value) {
    return value !== null && typeof value === "object";
  }

  function absoluteHref(path) {
    try {
      return new URL(path, window.location.origin).href;
    } catch {
      return path;
    }
  }

  function currentPythonVersion() {
    const segments = window.location.pathname.split("/").filter(Boolean);
    const pythonIndex = segments.indexOf("python");

    if (pythonIndex === -1 || pythonIndex + 1 >= segments.length) {
      return "";
    }

    try {
      return decodeURIComponent(segments[pythonIndex + 1]);
    } catch {
      return segments[pythonIndex + 1];
    }
  }

  function createOption({ label, href, version }) {
    const option = document.createElement("option");
    option.dataset.version = version;
    option.textContent = label;
    option.value = absoluteHref(href);
    return option;
  }

  function renderVersionSelector(manifest) {
    const python = isRecord(manifest) && isRecord(manifest.python) ? manifest.python : null;
    const versions = Array.isArray(python?.versions)
      ? python.versions.filter((entry) => isRecord(entry) && typeof entry.version === "string" && entry.version)
      : [];

    if (!python || versions.length === 0 || document.querySelector(`.${CONTAINER_CLASS}`)) {
      return;
    }

    const mount = document.querySelector(".wy-side-nav-search") || document.querySelector(".wy-nav-side");

    if (!mount) {
      return;
    }

    const latestVersion = typeof python.latest === "string" && python.latest ? python.latest : versions[0].version;
    const label = document.createElement("label");
    const labelText = document.createElement("span");
    const select = document.createElement("select");
    const currentVersion = currentPythonVersion();

    label.className = CONTAINER_CLASS;
    labelText.textContent = "Python docs version";
    select.setAttribute("aria-label", "Python docs version");

    select.appendChild(
      createOption({
        label: `Latest (${latestVersion})`,
        href: typeof python.latestPath === "string" && python.latestPath ? python.latestPath : "/python/latest/",
        version: "latest",
      }),
    );

    versions.forEach((entry) => {
      const label = typeof entry.label === "string" && entry.label ? entry.label : entry.version;
      const href = typeof entry.path === "string" && entry.path ? entry.path : `/python/${encodeURIComponent(entry.version)}/`;
      select.appendChild(createOption({ label, href, version: entry.version }));
    });

    Array.from(select.options).forEach((option) => {
      if (option.dataset.version === currentVersion) {
        select.value = option.value;
      }
    });

    select.addEventListener("change", () => {
      if (select.value) {
        window.location.assign(select.value);
      }
    });

    label.append(labelText, select);
    mount.appendChild(label);
  }

  function loadVersionSelector() {
    if (!window.fetch) {
      return;
    }

    fetch(MANIFEST_PATH, { cache: "no-cache" })
      .then((response) => (response.ok ? response.json() : null))
      .then((manifest) => {
        if (manifest) {
          renderVersionSelector(manifest);
        }
      })
      .catch(() => {});
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", loadVersionSelector, { once: true });
  } else {
    loadVersionSelector();
  }
})();
