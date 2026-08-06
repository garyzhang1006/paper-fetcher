const state = {
  papers: [],
  searchTimer: null,
};

const elements = {
  paperCount: document.querySelector("#paper-count"),
  featureCount: document.querySelector("#feature-count"),
  graphCount: document.querySelector("#graph-count"),
  sourceCount: document.querySelector("#source-count"),
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
  graphBuildStatus: document.querySelector("#graph-build-status"),
  placementForm: document.querySelector("#placement-form"),
  placementId: document.querySelector("#placement-id"),
  placementResult: document.querySelector("#placement-result"),
  hotTopics: document.querySelector("#hot-topic-list"),
  emergingTopics: document.querySelector("#emerging-topic-list"),
  hubs: document.querySelector("#hub-list"),
  clusters: document.querySelector("#cluster-list"),
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
  elements.sourceCount.textContent = formatNumber(counts.source_items);
}

function itemName(item) {
  return item.name || item.topic || item.concept_name || item.concept || item.label || item.node_id || "Unnamed";
}

function renderAnalysisList(element, items, detailFormatter, emptyMessage) {
  element.replaceChildren();
  if (!items.length) {
    const item = document.createElement("li");
    item.textContent = emptyMessage;
    element.append(item);
    return;
  }
  items.slice(0, 8).forEach((entry) => {
    const item = document.createElement("li");
    const name = document.createElement("strong");
    name.textContent = itemName(entry);
    item.append(name);
    const detail = detailFormatter(entry);
    if (detail) {
      const small = document.createElement("small");
      small.textContent = detail;
      item.append(small);
    }
    element.append(item);
  });
}

async function loadGraphInsights() {
  try {
    const [trendPayload, hubPayload, clusterPayload] = await Promise.all([
      fetchJson("/api/trends"),
      fetchJson("/api/hubs"),
      fetchJson("/api/clusters"),
    ]);
    if (trendPayload.status !== "ready") {
      elements.graphBuildStatus.textContent = "Graph has not been built";
      elements.graphBuildStatus.classList.remove("ready");
      return;
    }
    elements.graphBuildStatus.textContent = `Build ${trendPayload.build_id.slice(0, 12)} · ${formatDate(trendPayload.built_at)}`;
    elements.graphBuildStatus.classList.add("ready");

    const trendReport = trendPayload.trends || {};
    const hot = Array.isArray(trendReport.hot) ? trendReport.hot : [];
    const emerging = Array.isArray(trendReport.emerging) ? trendReport.emerging : [];
    const historyMessage = trendReport.status === "insufficient_history"
      ? `Insufficient history: ${formatNumber(trendReport.coverage_days)} days available.`
      : "No topic signal available.";
    renderAnalysisList(
      elements.hotTopics,
      hot,
      (item) => `${formatNumber(item.recent_count || item.current_count)} recent documents`,
      historyMessage
    );
    renderAnalysisList(
      elements.emergingTopics,
      emerging,
      (item) => `growth ${Number(item.growth || item.growth_score || 0).toFixed(2)} · ${formatNumber(item.recent_count || item.current_count)} recent`,
      "No topic met emerging thresholds."
    );

    const hubs = Array.isArray(hubPayload.hubs) ? hubPayload.hubs : [];
    renderAnalysisList(
      elements.hubs,
      hubs,
      (item) => `PageRank ${Number(item.pagerank || 0).toFixed(4)} · degree ${formatNumber(item.degree)}`,
      "No hubs available."
    );
    const clusters = Array.isArray(clusterPayload.clusters) ? clusterPayload.clusters : [];
    renderAnalysisList(
      elements.clusters,
      clusters,
      (item) => `${formatNumber(item.member_ids?.length || 0)} documents`,
      "No related-document clusters met threshold."
    );
  } catch (error) {
    elements.graphBuildStatus.textContent = `Graph unavailable: ${error.message}`;
    elements.graphBuildStatus.classList.remove("ready");
  }
}

function renderPlacement(payload) {
  elements.placementResult.replaceChildren();
  const heading = document.createElement("strong");
  heading.textContent = payload.document.name;
  const summary = document.createElement("p");
  summary.className = "placement-summary";
  summary.textContent = payload.document.properties.summary || "No grounded summary available.";
  const chips = document.createElement("div");
  chips.className = "relationship-chips";
  payload.relationships.forEach((relationship) => {
    const neighborId = relationship.source_id === payload.document.node_id
      ? relationship.target_id
      : relationship.source_id;
    const neighbor = payload.neighbors.find((item) => item.node_id === neighborId);
    const chip = document.createElement("span");
    chip.className = "relationship-chip";
    chip.textContent = `${relationship.relation.replaceAll("_", " ")} · ${neighbor?.name || neighborId}`;
    chips.append(chip);
  });
  elements.placementResult.append(heading, summary, chips);
}

async function handlePlacement(event) {
  event.preventDefault();
  const identifier = elements.placementId.value.trim();
  if (!identifier) {
    elements.placementResult.textContent = "Enter a paper or feed-item ID.";
    elements.placementId.focus();
    return;
  }
  elements.placementResult.textContent = "Loading graph placement…";
  try {
    const payload = await fetchJson(`/api/placement?id=${encodeURIComponent(identifier)}`);
    renderPlacement(payload);
  } catch (error) {
    elements.placementResult.textContent = error.message;
  }
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

function renderPapers(papers, total) {
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
  elements.resultsSummary.textContent = total > papers.length
    ? `Showing ${formatNumber(papers.length)} of ${formatNumber(total)} saved papers`
    : `${formatNumber(total)} saved ${total === 1 ? "paper" : "papers"}`;
}

function updateCategoryOptions(categories) {
  const selected = elements.categoryFilter.value;
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
    renderPapers(payload.papers, payload.total);
    if (updateCategories) {
      updateCategoryOptions(payload.categories);
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
  elements.placementForm.addEventListener("submit", handlePlacement);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeMobileMenu();
  });
}

async function initialize() {
  bindEvents();
  const results = await Promise.allSettled([
    loadStats(),
    loadPapers({ updateCategories: true }),
    loadGraphInsights(),
  ]);
  if (results[0].status === "rejected") {
    setFetchStatus("error", `Could not load database counts: ${results[0].reason.message}`);
  }
}

initialize();
