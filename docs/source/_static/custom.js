(function () {
  const MANIFEST_PATH = "/python/docs-manifest.json";
  const CONTAINER_CLASS = "fs-docs-version-selector";
  const HOME_LINK_CLASS = "fs-docs-home-link";

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

  function renderDocsHomeLink() {
    if (document.querySelector(`.${HOME_LINK_CLASS}`)) {
      return;
    }

    const mount = document.querySelector(".wy-side-nav-search") || document.querySelector(".wy-nav-side");

    if (!mount) {
      return;
    }

    const versionSelector = document.querySelector(`.${CONTAINER_CLASS}`);
    const homeLink = document.createElement("a");

    homeLink.className = HOME_LINK_CLASS;
    homeLink.href = "/";
    homeLink.textContent = "Looking for other docs?";

    if (versionSelector && versionSelector.parentElement === mount) {
      versionSelector.insertAdjacentElement("afterend", homeLink);
      return;
    }

    mount.appendChild(homeLink);
  }

  function loadVersionSelector() {
    if (!window.fetch) {
      renderDocsHomeLink();
      return;
    }

    fetch(MANIFEST_PATH, { cache: "no-cache" })
      .then((response) => (response.ok ? response.json() : null))
      .then((manifest) => {
        if (manifest) {
          renderVersionSelector(manifest);
        }
        renderDocsHomeLink();
      })
      .catch(() => {
        renderDocsHomeLink();
      });
  }

  function nextToken(nodes, index) {
    for (let i = index + 1; i < nodes.length; i += 1) {
      const node = nodes[i];

      if (node.nodeType === Node.TEXT_NODE) {
        if (node.textContent.trim() !== "") {
          return null;
        }
        continue;
      }

      if (node.nodeType === Node.ELEMENT_NODE) {
        return node;
      }
    }

    return null;
  }

  function previousToken(nodes, index) {
    for (let i = index - 1; i >= 0; i -= 1) {
      const node = nodes[i];

      if (node.nodeType === Node.TEXT_NODE) {
        if (node.textContent.trim() !== "") {
          return null;
        }
        continue;
      }

      if (node.nodeType === Node.ELEMENT_NODE) {
        return node;
      }
    }

    return null;
  }

  function updateLinePrefix(prefix, text) {
    const lastNewline = text.lastIndexOf("\n");

    if (lastNewline !== -1) {
      return text.slice(lastNewline + 1);
    }

    return prefix + text;
  }

  function updateParenDepth(depth, text) {
    let nextDepth = depth;

    for (const char of text) {
      if (char === "(" || char === "[" || char === "{") {
        nextDepth += 1;
      } else if (char === ")" || char === "]" || char === "}") {
        nextDepth = Math.max(0, nextDepth - 1);
      }
    }

    return nextDepth;
  }

  function enhancePythonCodeBlocks() {
    document.querySelectorAll("div.highlight-python pre").forEach((pre) => {
      const nodes = Array.from(pre.childNodes);
      let linePrefix = "";
      let parenDepth = 0;

      nodes.forEach((node, index) => {
        if (node.nodeType === Node.TEXT_NODE) {
          linePrefix = updateLinePrefix(linePrefix, node.textContent);
          parenDepth = updateParenDepth(parenDepth, node.textContent);
          return;
        }

        if (node.nodeType !== Node.ELEMENT_NODE) {
          return;
        }

        const text = node.textContent || "";

        if (node.classList.contains("n")) {
          const previous = previousToken(nodes, index);
          const next = nextToken(nodes, index);

          if (previous?.classList.contains("o") && previous.textContent === ".") {
            node.classList.add(/^[A-Z]/.test(text) ? "fs-code-type-name" : "fs-code-member-name");
          } else if (
            next?.classList.contains("o") &&
            next.textContent === "=" &&
            parenDepth > 0 &&
            /^\s+$/.test(linePrefix)
          ) {
            node.classList.add("fs-code-param-name");
          } else {
            node.classList.add("fs-code-var-name");
          }
        }

        linePrefix = updateLinePrefix(linePrefix, text);
        parenDepth = updateParenDepth(parenDepth, text);
      });
    });
  }

  function glossaryTermLinks() {
    return Array.from(document.querySelectorAll('a.reference.internal[href*="glossary.html#term-"]')).filter((link) =>
      link.querySelector(".std-term"),
    );
  }

  function normalizeTermId(hash) {
    if (!hash || !hash.startsWith("#term-")) {
      return "";
    }

    try {
      return decodeURIComponent(hash.slice(1));
    } catch {
      return hash.slice(1);
    }
  }

  function createGlossaryTooltip() {
    const tooltip = document.createElement("aside");

    tooltip.className = "fs-glossary-tooltip";
    tooltip.setAttribute("role", "tooltip");
    tooltip.hidden = true;
    tooltip.innerHTML = `
      <div class="fs-glossary-tooltip__term"></div>
      <div class="fs-glossary-tooltip__definition"></div>
      <a class="fs-glossary-tooltip__link" href="#">Open glossary entry</a>
    `;
    document.body.appendChild(tooltip);

    return tooltip;
  }

  function extractGlossaryEntries(documentRoot) {
    const entries = new Map();

    documentRoot.querySelectorAll("dl.glossary > dt[id]").forEach((termNode) => {
      const definitionNode = termNode.nextElementSibling;

      if (!definitionNode || definitionNode.tagName.toLowerCase() !== "dd") {
        return;
      }

      const term = Array.from(termNode.childNodes)
        .filter((node) => !(node.nodeType === Node.ELEMENT_NODE && node.classList.contains("headerlink")))
        .map((node) => node.textContent)
        .join("")
        .trim();
      const definition = definitionNode.textContent.replace(/\s+/g, " ").trim();

      if (term && definition) {
        entries.set(termNode.id, { definition, term });
      }
    });

    return entries;
  }

  function positionGlossaryTooltip(tooltip, target) {
    const gap = 10;
    const margin = 12;
    const rect = target.getBoundingClientRect();

    tooltip.style.left = "0px";
    tooltip.style.top = "0px";
    tooltip.hidden = false;

    const tooltipRect = tooltip.getBoundingClientRect();
    let left = rect.left;
    let top = rect.bottom + gap;

    if (left + tooltipRect.width > window.innerWidth - margin) {
      left = window.innerWidth - tooltipRect.width - margin;
    }

    if (top + tooltipRect.height > window.innerHeight - margin) {
      top = rect.top - tooltipRect.height - gap;
    }

    tooltip.style.left = `${Math.max(margin, left)}px`;
    tooltip.style.top = `${Math.max(margin, top)}px`;
  }

  function enhanceGlossaryTermLinks() {
    const links = glossaryTermLinks();

    if (links.length === 0 || !window.fetch || !window.DOMParser) {
      return;
    }

    const tooltip = createGlossaryTooltip();
    const tooltipTerm = tooltip.querySelector(".fs-glossary-tooltip__term");
    const tooltipDefinition = tooltip.querySelector(".fs-glossary-tooltip__definition");
    const tooltipLink = tooltip.querySelector(".fs-glossary-tooltip__link");
    const glossaryEntriesByUrl = new Map();
    let activeLink = null;
    let hideTimer = 0;

    function hideTooltip() {
      tooltip.hidden = true;
      activeLink = null;
    }

    function queueHideTooltip() {
      window.clearTimeout(hideTimer);
      hideTimer = window.setTimeout(() => {
        if (!tooltip.matches(":hover") && activeLink !== document.activeElement) {
          hideTooltip();
        }
      }, 180);
    }

    function showTooltip(link, entry) {
      window.clearTimeout(hideTimer);
      activeLink = link;
      tooltipTerm.textContent = entry.term;
      tooltipDefinition.textContent = entry.definition;
      tooltipLink.href = link.href;
      tooltipLink.setAttribute("aria-label", `Open ${entry.term} in the glossary`);
      positionGlossaryTooltip(tooltip, link);
    }

    function loadGlossaryEntries(url) {
      const glossaryUrl = new URL(url.href);

      glossaryUrl.hash = "";

      const cacheKey = glossaryUrl.href;

      if (!glossaryEntriesByUrl.has(cacheKey)) {
        glossaryEntriesByUrl.set(
          cacheKey,
          fetch(cacheKey)
            .then((response) => (response.ok ? response.text() : ""))
            .then((html) => {
              if (!html) {
                return new Map();
              }

              const parser = new DOMParser();
              const glossaryDocument = parser.parseFromString(html, "text/html");

              return extractGlossaryEntries(glossaryDocument);
            })
            .catch(() => new Map()),
        );
      }

      return glossaryEntriesByUrl.get(cacheKey);
    }

    function showLinkTooltip(link) {
      const url = new URL(link.href);
      const termId = normalizeTermId(url.hash);

      if (!termId) {
        return Promise.resolve(false);
      }

      return loadGlossaryEntries(url).then((entries) => {
        const entry = entries.get(termId);

        if (!entry) {
          return false;
        }

        showTooltip(link, entry);
        return true;
      });
    }

    links.forEach((link) => {
      link.classList.add("fs-glossary-term");
      link.setAttribute("data-fs-glossary-term", "true");

      link.addEventListener("mouseenter", () => {
        showLinkTooltip(link);
      });
      link.addEventListener("mouseleave", queueHideTooltip);
      link.addEventListener("focus", () => {
        showLinkTooltip(link);
      });
      link.addEventListener("blur", queueHideTooltip);
      link.addEventListener("click", (event) => {
        if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
          return;
        }

        if (activeLink === link && !tooltip.hidden) {
          return;
        }

        event.preventDefault();
        showLinkTooltip(link);
      });
    });

    tooltip.addEventListener("mouseenter", () => {
      window.clearTimeout(hideTimer);
    });
    tooltip.addEventListener("mouseleave", queueHideTooltip);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !tooltip.hidden) {
        hideTooltip();
      }
    });
    window.addEventListener("scroll", () => {
      if (activeLink && !tooltip.hidden) {
        positionGlossaryTooltip(tooltip, activeLink);
      }
    }, { passive: true });
    window.addEventListener("resize", () => {
      if (activeLink && !tooltip.hidden) {
        positionGlossaryTooltip(tooltip, activeLink);
      }
    });
  }

  function initialize() {
    enhancePythonCodeBlocks();
    enhanceGlossaryTermLinks();
    loadVersionSelector();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
