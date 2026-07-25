"use strict";

/* Backward Factor Trace — circuit explorer.
 * Pure static app: reads models/index.json + per-model manifest.json and crops
 * factor tiles out of per-node sprite sheets on <canvas>. No dependencies. */

const state = {
  index: null,     // { models: [...] }
  model: null,     // current manifest
  nodeById: null,  // id -> node
  pathMap: null,   // "0,1,.." -> node
  nodeIdx: null,   // current node id
  sprites: new Map(), // url -> Promise<Image>
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, txt) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  return e;
};
const pathKey = (p) => (p || []).join(",");

/* ── sprite loading + tile drawing ─────────────────────────────────────── */

function spriteURL(node) { return `models/${state.model.id}/${node.sprite}`; }

function loadSprite(node) {
  const url = spriteURL(node);
  if (!state.sprites.has(url)) {
    state.sprites.set(url, new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => res(im);
      im.onerror = () => rej(new Error("sprite " + url));
      im.src = url;
    }));
  }
  return state.sprites.get(url);
}

// col 0 = weighted-average image; cols 1.. = example stimuli.
function makeTile(img, node, row, col, size, smooth) {
  const dpr = window.devicePixelRatio || 1;
  const cv = document.createElement("canvas");
  cv.width = Math.round(size * dpr);
  cv.height = Math.round(size * dpr);
  cv.style.width = size + "px";
  cv.style.height = size + "px";
  if (!smooth) cv.classList.add("crisp");
  const ctx = cv.getContext("2d");
  ctx.imageSmoothingEnabled = smooth;
  ctx.scale(dpr, dpr);
  const t = node.tile;
  ctx.drawImage(img, col * t, row * t, t, t, 0, 0, size, size);
  return cv;
}

function isSmooth() { return state.model.image_kind === "rgb"; }

/* ── tree helpers ──────────────────────────────────────────────────────── */

function chainTo(node) {           // [root, ..., node]
  const out = [];
  for (let d = 0; d <= node.path.length; d++) {
    const n = state.pathMap.get(pathKey(node.path.slice(0, d)));
    if (n) out.push(n);
  }
  return out;
}

function layerOfNode(node) {
  return state.model.layers.find((l) => l.idx === node.layer_idx);
}

// The input-most traced layer (manifest layers are sorted ascending by idx).
function isInputLayer(node) {
  return node.layer_idx === state.model.layers[0].idx;
}

/* ── model loading / selection ─────────────────────────────────────────── */

async function loadIndex() {
  state.index = await (await fetch("models/index.json")).json();
  const nav = $("#modelNav");
  nav.innerHTML = "";
  state.index.models.forEach((m) => {
    const b = el("button", null, m.title);
    b.dataset.id = m.id;
    b.title = m.subtitle;
    b.onclick = () => selectModel(m.id);
    nav.appendChild(b);
  });
}

async function selectModel(id) {
  const entry = state.index.models.find((m) => m.id === id);
  const manifest = await (await fetch(entry.file)).json();
  state.model = manifest;
  state.nodeById = new Map(manifest.nodes.map((n) => [n.id, n]));
  state.pathMap = new Map(manifest.nodes.map((n) => [pathKey(n.path), n]));
  document.querySelectorAll("#modelNav button").forEach((b) =>
    b.classList.toggle("active", b.dataset.id === id));
  $("#footMeta").textContent =
    `${manifest.meta.n_layers} layers · ${manifest.meta.n_nodes} traced nodes · ` +
    `depth ${manifest.depth} · ${manifest.meta.n_stimuli} stimuli` +
    (manifest.meta.test_acc != null ? ` · acc ${(manifest.meta.test_acc * 100).toFixed(1)}%` : "");
  $("#layout").hidden = false;
  goTo(manifest.root);
}

function goTo(nodeId) {
  state.nodeIdx = nodeId;
  render();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ── render: network schematic ─────────────────────────────────────────── */

async function renderNetwork() {
  const stack = $("#netStack");
  stack.innerHTML = "";
  const cur = state.nodeById.get(state.nodeIdx);
  const chain = chainTo(cur);
  const visited = new Map(chain.map((n) => [n.layer_idx, n]));
  await Promise.all(chain.map(loadSprite));

  const layers = state.model.layers.slice().sort((a, b) => b.idx - a.idx); // output→input
  for (let i = 0; i < layers.length; i++) {
    const L = layers[i];
    const node = visited.get(L.idx);
    const box = el("div", "layer");
    if (node) box.classList.add("visited");
    else box.classList.add("dim");
    if (node && node.id === cur.id) box.classList.add("current");
    box.appendChild(el("span", "layer-name", L.name));
    box.appendChild(el("span", "layer-meta", `${L.type} · ${L.size} units`));
    if (node) box.onclick = () => goTo(node.id);
    if (node) box.style.cursor = "pointer";
    stack.appendChild(box);

    // connector to the layer below, carrying the chosen factor's avg image
    if (i < layers.length - 1 && node) {
      const belowLayer = layers[i + 1];
      const belowNode = visited.get(belowLayer.idx);
      const conn = el("div", "connector");
      if (belowNode) {
        conn.classList.add("active");
        const chosenK = belowNode.path[belowNode.path.length - 1];
        const img = await loadSprite(node);
        conn.appendChild(makeTile(img, node, chosenK, 0, 30, isSmooth()));
        const f = node.factors[chosenK];
        conn.appendChild(el("span", "edge-label", f ? f.label : `factor ${chosenK}`));
        const arr = el("span", "arrow", "↓"); conn.appendChild(arr);
      }
      stack.appendChild(conn);
    }
  }

  // input image tail
  const inp = el("div", "layer input dim");
  inp.appendChild(el("span", "layer-name", state.model.input.name));
  inp.appendChild(el("span", "layer-meta", state.model.image_kind === "rgb" ? "3 channels" : "1 channel"));
  stack.appendChild(inp);
}

/* ── render: breadcrumb + node header ──────────────────────────────────── */

function renderCrumbs() {
  const box = $("#crumbs");
  box.innerHTML = "";
  const cur = state.nodeById.get(state.nodeIdx);
  const chain = chainTo(cur);
  chain.forEach((n, i) => {
    if (i > 0) box.appendChild(el("span", "sep", "›"));
    let label;
    if (i === 0) label = "output";
    else {
      const parent = chain[i - 1];
      const k = n.path[n.path.length - 1];
      label = parent.factors[k] ? parent.factors[k].label : `factor ${k}`;
    }
    const c = el("span", "crumb" + (n.id === cur.id ? " here" : ""), label);
    if (n.id !== cur.id) c.onclick = () => goTo(n.id);
    box.appendChild(c);
  });
}

function renderNodeHead() {
  const head = $("#nodeHead");
  head.innerHTML = "";
  const cur = state.nodeById.get(state.nodeIdx);
  const L = layerOfNode(cur);
  const title = el("div");
  title.appendChild(el("div", "nh-title", `${cur.n_factors} factor${cur.n_factors > 1 ? "s" : ""} at ${L.name}`));
  const depthTxt = cur.depth === 0 ? "output layer — pick a class circuit to unpack"
    : `depth ${cur.depth} of ${state.model.depth}`;
  title.appendChild(el("div", "nh-sub", `${L.type} layer · ${depthTxt}`));
  head.appendChild(title);
  head.appendChild(el("span", "tag", L.type));
  if (cur.depth === 0) head.appendChild(el("span", "tag gray", "root"));
}

/* ── render: factor cards ──────────────────────────────────────────────── */

function profileEl(node, factor) {
  const wrap = el("div", "profile");
  const labels = state.model.profile_labels;
  if (!factor.profile || !labels) return wrap;
  const idx = factor.profile.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 4);
  const total = factor.profile.reduce((a, b) => a + b, 0) || 1;
  idx.forEach(([v, i]) => {
    if (v <= 0) return;
    const row = el("div", "row");
    row.appendChild(el("span", "plabel", labels[i]));
    const track = el("div", "ptrack");
    const fill = el("i"); fill.style.width = Math.round((v / total) * 100) + "%";
    track.appendChild(fill); row.appendChild(track);
    row.appendChild(el("span", "pval", Math.round((v / total) * 100) + "%"));
    wrap.appendChild(row);
  });
  return wrap;
}

async function renderFactors() {
  const grid = $("#factorGrid");
  grid.innerHTML = "";
  const cur = state.nodeById.get(state.nodeIdx);
  const img = await loadSprite(cur);
  const smooth = isSmooth();

  cur.factors.forEach((f) => {
    const expandable = f.child != null;
    const card = el("div", "card " + (expandable ? "expandable" : "terminal"));

    const top = el("div", "card-top");
    const wavgWrap = el("div", "wavg-wrap");
    wavgWrap.appendChild(makeTile(img, cur, f.k, 0, 46, smooth));
    wavgWrap.appendChild(el("div", "avg-cap", "weighted avg"));
    top.appendChild(wavgWrap);

    const info = el("div", "card-info");
    info.appendChild(el("div", "flabel", f.label));
    const imp = el("div", "imp");
    imp.appendChild(el("span", null, "importance"));
    const bar = el("div", "bar"); const bi = el("i");
    bi.style.width = Math.round(f.lam * 100) + "%"; bar.appendChild(bi);
    imp.appendChild(bar);
    imp.appendChild(el("span", null, Math.round(f.lam * 100) + "%"));
    info.appendChild(imp);
    info.appendChild(profileEl(cur, f));
    top.appendChild(info);
    card.appendChild(top);

    // hero: the real example stimuli that drive this factor
    const nEx = Math.min(f.n_ex, 6);
    const exWrap = el("div", "examples");
    const exHead = el("div", "ex-head");
    exHead.appendChild(el("div", "ex-label", "top example stimuli"));
    const inspect = el("button", "ex-inspect", "⤢ inspect");
    inspect.title = "Inspect this factor";
    inspect.onclick = (ev) => { ev.stopPropagation(); openModal(cur, f); };
    exHead.appendChild(inspect);
    exWrap.appendChild(exHead);
    const exRow = el("div", "ex-row");
    for (let t = 0; t < nEx; t++) exRow.appendChild(makeTile(img, cur, f.k, 1 + t, 62, smooth));
    exWrap.appendChild(exRow);
    card.appendChild(exWrap);

    const foot = el("div", "card-foot");
    if (expandable) {
      const child = state.nodeById.get(f.child);
      const cl = layerOfNode(child);
      foot.appendChild(el("span", null, `${child.n_factors} sub-factors`));
      foot.appendChild(el("span", "go", `explore ${cl.name} →`));
      card.onclick = () => goTo(f.child);
    } else {
      foot.appendChild(el("span", "leaf",
        isInputLayer(cur) ? "input-layer factor" : "not expanded further"));
      foot.appendChild(el("span", "leaf", "◾ leaf"));
      card.onclick = () => openModal(cur, f);
      card.style.cursor = "pointer";
    }
    card.appendChild(foot);
    grid.appendChild(card);
  });
}

async function render() {
  await Promise.all([renderNetwork(), (async () => { renderCrumbs(); renderNodeHead(); await renderFactors(); })()]);
}

/* ── modal ─────────────────────────────────────────────────────────────── */

async function openModal(node, factor) {
  const card = $("#modalCard");
  card.innerHTML = "";
  const img = await loadSprite(node);
  const smooth = isSmooth();

  const close = el("button", "m-close", "×");
  close.onclick = closeModal; card.appendChild(close);

  card.appendChild(el("h3", null, factor.label));
  const L = layerOfNode(node);
  card.appendChild(el("div", "m-sub",
    `${L.name} (${L.type}) · importance ${Math.round(factor.lam * 100)}%` +
    (factor.child != null ? ""
      : (isInputLayer(node) ? " · input-layer leaf" : " · trace leaf (not expanded)"))));

  // hero: the real example stimuli
  card.appendChild(el("div", "m-section-label", "top example stimuli"));
  const exs = el("div", "m-examples");
  for (let t = 0; t < factor.n_ex; t++) exs.appendChild(makeTile(img, node, factor.k, 1 + t, 104, smooth));
  card.appendChild(exs);

  // secondary: weighted-average image + class profile
  const body = el("div", "m-body"); body.style.marginTop = "18px";
  const left = el("div");
  left.appendChild(el("div", "m-section-label", "weighted average"));
  left.appendChild(makeTile(img, node, factor.k, 0, 116, smooth));
  body.appendChild(left);

  const right = el("div"); right.style.flex = "1"; right.style.minWidth = "200px";
  right.appendChild(el("div", "m-section-label", "class profile"));
  right.appendChild(profileEl(node, factor));
  body.appendChild(right);
  card.appendChild(body);

  if (factor.child != null) {
    const child = state.nodeById.get(factor.child);
    const cl = layerOfNode(child);
    const go = el("button", "m-go");
    go.innerHTML = `Explore ${child.n_factors} sub-factors at ${cl.name} →`;
    go.onclick = () => { closeModal(); goTo(factor.child); };
    card.appendChild(go);
  }
  $("#modal").classList.remove("hidden");
}
function closeModal() { $("#modal").classList.add("hidden"); }

/* ── wiring ────────────────────────────────────────────────────────────── */

function goUp() {
  const cur = state.nodeById.get(state.nodeIdx);
  if (cur.path.length === 0) return;
  const parent = state.pathMap.get(pathKey(cur.path.slice(0, -1)));
  if (parent) goTo(parent.id);
}

function wire() {
  $("#helpBtn").onclick = () => $("#help").classList.remove("hidden");
  $("#helpClose").onclick = () => $("#help").classList.add("hidden");
  $("#help").onclick = (e) => { if (e.target.id === "help") $("#help").classList.add("hidden"); };
  $("#modalBackdrop").onclick = closeModal;
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { closeModal(); $("#help").classList.add("hidden"); }
    if (e.key === "Backspace" && $("#modal").classList.contains("hidden")) {
      e.preventDefault(); goUp();
    }
  });
}

async function main() {
  wire();
  await loadIndex();
  $("#splash").classList.add("hidden");
  await selectModel(state.index.models[0].id);
}
main();
