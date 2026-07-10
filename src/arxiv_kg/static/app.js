const state = {
  papers: [],
  searchTimer: null,
};

const elements = {
  paperCount: document.querySelector("#paper-count"),
  featureCount: document.querySelector("#feature-count"),
  graphCount: document.querySelector("#graph-count"),
  paperList: document.querySelector("#paper-list"),
  resultsSummary: document.querySelector("#results-summary"),
  search: document.querySelector("#paper-search"),
  categoryFilter: document.querySelector("#category-filter"),
  form: document.querySelector("#fetch-form"),
  submit: document.querySelector(".fetch-submit"),
  fetchStatus: document.querySelector("#fetch-status"),
  fetchPanel: document.querySelector("#fetch-panel"),
  paperTemplate: document.querySelector("#paper-template"),
  mobileMenu: document.querySelector(".mobile-menu"),
  sidebar: document.querySelector(".sidebar"),
};

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({ error: "Invalid server response" }));
  if (!response.ok) {
    const message = payload.detail ? `${payload.error}: ${payload.detail}` : payload.error;
    throw new Error(message || `Request failed with status ${response.status}`);
  }
  return payload;
}

function formatNumber(value) {
  return new Intl.NumberFormat().format(value || 0);
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown date";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(date);
}

async function loadStats() {
  const counts = await fetchJson("/api/stats");
  elements.paperCount.textContent = formatNumber(counts.papers);
  elements.featureCount.textContent = formatNumber(counts.features);
  elements.graphCount.textContent = formatNumber(counts.edges);
}

function showLoadingState() {
  elements.paperList.setAttribute("aria-busy", "true");
  elements.paperList.replaceChildren();
  const wrapper = document.createElement("div");
  wrapper.className = "loading-state";
  wrapper.setAttribute("aria-label", "Loading saved papers");
  for (let index = 0; index < 3; index += 1) {
    const skeleton = document.createElement("div");
    skeleton.className = "skeleton";
    wrapper.append(skeleton);
  }
  elements.paperList.append(wrapper);
}

function showMessageState(kind, title, message) {
  elements.paperList.replaceChildren();
  const wrapper = document.createElement("div");
  wrapper.className = `${kind}-state`;
  const heading = document.createElement("strong");
  heading.textContent = title;
  const copy = document.createElement("p");
  copy.textContent = message;
  wrapper.append(heading, copy);
  elements.paperList.append(wrapper);
  elements.paperList.setAttribute("aria-busy", "false");
}

function renderPapers(papers) {
  elements.paperList.replaceChildren();
  elements.paperList.setAttribute("aria-busy", "false");

  const hasFilters = Boolean(elements.search.value.trim() || elements.categoryFilter.value);
  if (!papers.length) {
    if (hasFilters) {
      showMessageState(
        "empty",
        "No matching papers",
        "Try a broader search or clear the category filter."
      );
    } else {
      showMessageState(
        "empty",
        "Your library is ready",
        "Choose research categories and fetch from arXiv to build your first collection."
      );
    }
    elements.resultsSummary.textContent = "0 papers";
    return;
  }

  const fragment = document.createDocumentFragment();
  papers.forEach((paper) => {
    const card = elements.paperTemplate.content.cloneNode(true);
    const title = card.querySelector(".paper-title");
    title.textContent = paper.title;
    title.href = paper.abs_url;
    card.querySelector(".category-pill").textContent = paper.primary_category;
    card.querySelector(".paper-date").textContent = `Published ${formatDate(paper.published_at)}`;
    card.querySelector(".version-pill").textContent = `v${paper.version}`;
    card.querySelector(".paper-authors").textContent = paper.authors.join(", ");
    card.querySelector(".paper-abstract").textContent = paper.abstract;
    card.querySelector(".paper-id").textContent = `arXiv:${paper.versioned_id}`;
    const abstractLink = card.querySelector(".abstract-link");
    abstractLink.href = paper.abs_url;
    abstractLink.setAttribute("aria-label", `Open abstract for ${paper.title}`);
    const pdfLink = card.querySelector(".pdf-link");
    if (paper.pdf_url) {
      pdfLink.href = paper.pdf_url;
      pdfLink.setAttribute("aria-label", `Open PDF for ${paper.title}`);
    } else {
      pdfLink.remove();
    }
    fragment.append(card);
  });
  elements.paperList.append(fragment);
  elements.resultsSummary.textContent = `${formatNumber(papers.length)} saved ${papers.length === 1 ? "paper" : "papers"}`;
}

function updateCategoryOptions(papers) {
  const selected = elements.categoryFilter.value;
  const categories = [...new Set(papers.flatMap((paper) => paper.categories))].sort();
  const options = [new Option("All categories", "")];
  categories.forEach((category) => options.push(new Option(category, category)));
  elements.categoryFilter.replaceChildren(...options);
  if (categories.includes(selected)) elements.categoryFilter.value = selected;
}

async function loadPapers({ updateCategories = false } = {}) {
  showLoadingState();
  const params = new URLSearchParams({ limit: "100" });
  const search = elements.search.value.trim();
  const category = elements.categoryFilter.value;
  if (search) params.set("search", search);
  if (category) params.set("category", category);

  try {
    const payload = await fetchJson(`/api/papers?${params.toString()}`);
    state.papers = payload.papers;
    renderPapers(payload.papers);
    if (updateCategories) {
      const allPapers = await fetchJson("/api/papers?limit=500");
      updateCategoryOptions(allPapers.papers);
    }
  } catch (error) {
    elements.resultsSummary.textContent = "Library unavailable";
    showMessageState("error", "Could not load papers", error.message);
  }
}

function setFetchStatus(kind, message) {
  elements.fetchStatus.hidden = false;
  elements.fetchStatus.className = `fetch-status ${kind}`;
  elements.fetchStatus.textContent = message;
}

function setFetching(isFetching) {
  elements.submit.disabled = isFetching;
  elements.submit.classList.toggle("loading", isFetching);
  elements.submit.setAttribute("aria-busy", String(isFetching));
}

async function handleFetch(event) {
  event.preventDefault();
  const categories = [...elements.form.querySelectorAll('input[name="category"]:checked')]
    .map((input) => input.value);
  if (!categories.length) {
    setFetchStatus("error", "Choose at least one research category.");
    elements.form.querySelector('input[name="category"]').focus();
    return;
  }

  setFetching(true);
  elements.fetchStatus.hidden = true;
  try {
    const payload = await fetchJson("/api/fetch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        categories,
        lookback_hours: Number(document.querySelector("#lookback").value),
        max_results: Number(document.querySelector("#max-results").value),
        scan_revisions: document.querySelector("#scan-revisions").checked,
      }),
    });
    const report = payload.report;
    setFetchStatus(
      "success",
      `Fetch complete. ${report.inserted} new, ${report.updated} revised, ${report.unchanged} already current.`
    );
    await Promise.all([loadStats(), loadPapers({ updateCategories: true })]);
  } catch (error) {
    setFetchStatus("error", error.message);
  } finally {
    setFetching(false);
  }
}

function focusFetchPanel() {
  elements.fetchPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  window.setTimeout(() => elements.form.querySelector('input[name="category"]').focus(), 250);
  closeMobileMenu();
}

function closeMobileMenu() {
  elements.sidebar.classList.remove("open");
  elements.mobileMenu.setAttribute("aria-expanded", "false");
}

function bindEvents() {
  elements.form.addEventListener("submit", handleFetch);
  elements.search.addEventListener("input", () => {
    window.clearTimeout(state.searchTimer);
    state.searchTimer = window.setTimeout(() => loadPapers(), 250);
  });
  elements.categoryFilter.addEventListener("change", () => loadPapers());
  document.querySelectorAll("[data-focus-fetch]").forEach((button) => {
    button.addEventListener("click", focusFetchPanel);
  });
  elements.mobileMenu.addEventListener("click", () => {
    const isOpen = elements.sidebar.classList.toggle("open");
    elements.mobileMenu.setAttribute("aria-expanded", String(isOpen));
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeMobileMenu();
  });
}

async function initialize() {
  bindEvents();
  const results = await Promise.allSettled([loadStats(), loadPapers({ updateCategories: true })]);
  if (results[0].status === "rejected") {
    setFetchStatus("error", `Could not load database counts: ${results[0].reason.message}`);
  }
}

initialize();
