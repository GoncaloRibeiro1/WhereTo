// Turn Destination map app. Talks to server.py (see its docstring for the API).
//
// Flow: click a road -> the GPS arrow is placed on it (/api/snap) -> pulsing
// circles mark where the car can go next (/api/next) -> clicking one drives
// the arrow there, leaving a trail, and the heatmap + places update
// (/api/predict). Undo steps back one move, Reset starts again.
"use strict";

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = j.detail;
    throw new Error(typeof d === "string" ? d : d ? JSON.stringify(d) : r.statusText);
  }
  return j;
}

let toastTimer;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 5000);
}

const pct = (p) => (p >= 0.1 ? (p * 100).toFixed(0) : p >= 0.01 ? (p * 100).toFixed(1) : (p * 100).toFixed(2)) + "%";
const GLYPH = { left: "↰", straight: "↑", right: "↱" };
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// Distance (m) and compass bearing between [lat, lon] points (flat earth: fine in a city)
function metres(a, b) {
  const kx = 111320 * Math.cos((a[0] * Math.PI) / 180);
  return Math.hypot((b[1] - a[1]) * kx, (b[0] - a[0]) * 110540);
}
function bearing(a, b) {
  const kx = Math.cos((a[0] * Math.PI) / 180);
  return ((Math.atan2((b[1] - a[1]) * kx, b[0] - a[0]) * 180) / Math.PI + 360) % 360;
}

const dark = () => {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
};

// ---------------------------------------------------------------------------
// Map and layers
// ---------------------------------------------------------------------------
const map = L.map("map", { zoomControl: false }).setView([41.16, -8.62], 13);
L.control.zoom({ position: "topright" }).addTo(map);

// Panes stack the layers: heatmap < route trail < choices < markers
map.createPane("heat").style.zIndex = 350;
map.createPane("trail").style.zIndex = 420;
map.createPane("choices").style.zIndex = 430;

// Standard OpenStreetMap tiles (no API key). In dark mode style.css inverts
// them with a filter, since OSM has no free dark style.
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (lastPrediction) drawHeat(lastPrediction);
  drawTrail();
});

const heatLayer = L.layerGroup().addTo(map);
const heatRenderer = L.canvas({ pane: "heat", padding: 0.3 });
const choiceLayer = L.layerGroup().addTo(map);
const placeLayer = L.layerGroup().addTo(map);
const trailLine = L.polyline([], { pane: "trail", weight: 6, opacity: 0.9, lineCap: "round", lineJoin: "round", interactive: false }).addTo(map);
const trailCasing = L.polyline([], { pane: "trail", weight: 10, opacity: 0.9, color: "#fff", lineCap: "round", lineJoin: "round", interactive: false });
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function drawTrail() {
  trailLine.setStyle({ color: css("--nav") });
  trailCasing.setStyle({ color: dark() ? "#1a1a19" : "#ffffff" });
}
trailCasing.addTo(map);
trailLine.bringToFront();

// The GPS arrow: a navigation chevron, rotated to the heading
const ARROW_SVG = `<svg viewBox="0 0 40 40" aria-hidden="true">
  <circle cx="20" cy="20" r="18" style="fill: var(--nav); fill-opacity: .18"/>
  <path d="M20 5 L32 33 L20 26 L8 33 Z" style="fill: var(--nav)" stroke="#fff" stroke-width="2.5" stroke-linejoin="round"/>
</svg>`;
const arrow = L.marker([0, 0], {
  icon: L.divIcon({ className: "gps-arrow", html: `<div class="rot">${ARROW_SVG}</div>`, iconSize: [40, 40], iconAnchor: [20, 20] }),
  interactive: false,
  keyboard: false,
  zIndexOffset: 1000,
});
let shownHeading = 0;   // unwrapped, so 350° -> 10° turns 20°, not 340°
function setArrow(pos, heading) {
  arrow.setLatLng(pos);
  if (!map.hasLayer(arrow)) arrow.addTo(map);
  shownHeading += ((((heading - shownHeading) % 360) + 540) % 360) - 180;
  const rot = arrow.getElement()?.querySelector(".rot");
  if (rot) rot.style.transform = `rotate(${shownHeading}deg)`;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
// snap:    the /api/snap options for the placed road (two if it is two-way)
// dir:     which of them the arrow uses
// route:   edges [u, v, key] driven so far (first = the road it was placed on)
// trail:   [lat, lon] points drawn behind the arrow
// ahead:   points from the arrow to the junction it is heading for
// pos/heading: where the arrow is
// moves:   {turn, name} per junction, for the route list
// history: snapshots of the above, for Undo
let S = null;
let history = [];
let animating = false;
// Each kind of request has its own counter; a reply is used only if no
// newer request of that kind was sent meanwhile.
let choiceSeq = 0;
let predSeq = 0;
let lastPrediction = null;
let info = null;

const snapshot = () => JSON.parse(JSON.stringify(S));

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------
function settings() {
  const time = $("time").value || null;
  const trip = parseFloat($("trip").value);
  return {
    beta: parseFloat($("beta").value) || 0.03,
    trip_scale: trip > 0 ? trip * 60 : null,
    landmarks: $("landmarks").checked,
    time,
    day: $("day").value || null,
    cell: parseFloat($("cell").value),
  };
}

for (const id of ["time", "day", "landmarks", "beta", "trip", "cell"]) {
  $(id).addEventListener("change", () => S && S.route.length > 1 && refreshPrediction());
}
$("now").addEventListener("click", () => {
  const d = new Date();
  $("time").value = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  $("day").value = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"][d.getDay()];
  if (S && S.route.length > 1) refreshPrediction();
});
$("anytime").addEventListener("click", () => {
  $("time").value = "";
  $("day").value = "";
  if (S && S.route.length > 1) refreshPrediction();
});

// ---------------------------------------------------------------------------
// UI text and buttons
// ---------------------------------------------------------------------------
function updateUI(choices) {
  const placed = !!S;
  $("undo").disabled = !placed || animating;
  $("reset").disabled = !placed || animating;
  $("flip").hidden = !(placed && history.length === 0 && S.snap.length > 1);

  let text;
  if (!placed) text = "Click a road to place the car.";
  else if (animating) text = "Driving…";
  else if (choices && choices.dead_end) text = "Dead end: no legal way on. Undo or reset.";
  else if (history.length === 0)
    text = `On ${S.snap[S.dir].name}. Click a pulsing circle to drive to the next junction` +
      (S.snap.length > 1 ? ", or flip the direction." : ".");
  else text = "Click the next junction the car drives to.";
  $("instruction").textContent = text;

  // Route list
  $("route-section").hidden = !placed;
  if (!placed) $("route").innerHTML = "";
  else {
    const items = [`<li>Start on <b>${esc(S.snap[S.dir].name)}</b></li>`];
    for (const m of S.moves)
      items.push(`<li><span class="turn">${GLYPH[m.turn]} ${m.turn}</span> onto ${esc(m.name)}</li>`);
    $("route").innerHTML = items.join("");
  }
}

// ---------------------------------------------------------------------------
// Placing the car
// ---------------------------------------------------------------------------
map.on("click", async (e) => {
  if (animating || (S && history.length > 0)) {
    if (S && history.length > 0) toast("Reset to place the car somewhere else.");
    return;
  }
  if (info && !L.latLngBounds(info.bounds).contains(e.latlng)) {
    toast("That's outside the loaded map (Porto, Matosinhos, Maia).");
    return;
  }
  try {
    const snap = await api("/api/snap", { lat: e.latlng.lat, lon: e.latlng.lng });
    S = { snap: snap.options, dir: 0 };
    history = [];
    startOnRoad();
  } catch (err) {
    toast(err.message);
  }
});

function startOnRoad() {
  const o = S.snap[S.dir];
  Object.assign(S, {
    route: [o.edge], trail: [o.position], ahead: o.ahead,
    pos: o.position, heading: o.bearing, moves: [],
  });
  trailLine.setLatLngs(S.trail);
  trailCasing.setLatLngs(S.trail);
  setArrow(S.pos, S.heading);
  clearPrediction();
  loadChoices();
}

$("flip").addEventListener("click", () => {
  if (!S || S.snap.length < 2 || history.length) return;
  S.dir = 1 - S.dir;
  startOnRoad();
});

// ---------------------------------------------------------------------------
// Choices: pulsing circles + a dashed preview of the road leading there
// ---------------------------------------------------------------------------
async function loadChoices() {
  const id = ++choiceSeq;
  choiceLayer.clearLayers();
  updateUI();
  try {
    const res = await api("/api/next", { route: S.route });
    if (id !== choiceSeq) return;
    drawChoices(res);
    updateUI(res);
  } catch (err) {
    toast(err.message);
  }
}

// The point a fraction `frac` of the way along a path of [lat, lon]
function pointAlong(path, frac) {
  let total = 0;
  for (let i = 1; i < path.length; i++) total += metres(path[i - 1], path[i]);
  let left = total * frac;
  for (let i = 1; i < path.length; i++) {
    const d = metres(path[i - 1], path[i]);
    if (d >= left && d > 0) {
      const f = left / d, a = path[i - 1], b = path[i];
      return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
    }
    left -= d;
  }
  return path[path.length - 1];
}

function drawChoices(res) {
  choiceLayer.clearLayers();
  const nav = css("--nav");
  // Two choices can lead to the same junction by different roads; their
  // circles would overlap, so those go partway along their own road instead.
  const sameTarget = {};
  for (const o of res.options) sameTarget[o.target] = (sameTarget[o.target] || 0) + 1;
  const spots = [S.pos];
  for (const o of res.options) {
    const path = S.ahead.concat(o.coords.slice(1));
    const where = sameTarget[o.target] > 1 ? pointAlong(o.coords, 0.6) : o.target;
    const preview = L.polyline(path, {
      pane: "choices", color: nav, weight: 5, opacity: 0.45, dashArray: "2 9", lineCap: "round",
    });
    const hit = L.polyline(path, { pane: "choices", weight: 18, opacity: 0 }); // easier to hover/click
    const dot = L.marker(where, {
      icon: L.divIcon({ className: "choice", html: '<div class="ring"></div><div class="dot"></div>', iconSize: [36, 36] }),
      keyboard: true,
      title: `${o.turn} onto ${o.name}`,
    });
    const label = `${GLYPH[o.turn]} ${o.turn} onto ${esc(o.name)}`;
    dot.bindTooltip(label, { direction: "top", offset: [0, -14], className: "choice-tip" });
    hit.bindTooltip(label, { sticky: true, className: "choice-tip" });

    const hot = (on) => {
      preview.setStyle({ opacity: on ? 1 : 0.45, dashArray: on ? null : "2 9" });
      dot.getElement()?.classList.toggle("hot", on);
    };
    for (const layer of [dot, hit]) {
      layer.on("mouseover", () => hot(true));
      layer.on("mouseout", () => hot(false));
      layer.on("click", (ev) => { L.DomEvent.stop(ev); drive(o, path); });
    }
    choiceLayer.addLayer(preview).addLayer(hit).addLayer(dot);
    spots.push(where);
  }
  // Keep the arrow and every circle in view (zooming out only if needed)
  const view = L.latLngBounds(spots);
  if (!map.getBounds().pad(-0.08).contains(view))
    map.fitBounds(view.pad(0.25), { maxZoom: map.getZoom() });
}

// ---------------------------------------------------------------------------
// Driving to a choice
// ---------------------------------------------------------------------------
async function drive(option, path) {
  if (animating) return;
  history.push(snapshot());
  S.route = S.route.concat(option.edges);
  S.moves.push({ turn: option.turn, name: option.name });
  choiceLayer.clearLayers();

  // Ask for the next choices and the prediction while the arrow moves
  const id = ++choiceSeq;
  const pid = ++predSeq;
  const nextP = api("/api/next", { route: S.route });
  api("/api/predict", { route: S.route, settings: settings() })
    .then((res) => pid === predSeq && showPrediction(res))
    .catch((err) => pid === predSeq && toast(err.message));

  animating = true;
  updateUI();
  await animateAlong(path);
  animating = false;
  S.pos = option.target;
  S.heading = option.arrival_bearing;
  S.ahead = [option.target];
  setArrow(S.pos, S.heading);

  try {
    const res = await nextP;
    if (id !== choiceSeq) return;
    drawChoices(res);
    updateUI(res);
  } catch (err) {
    updateUI();
    toast(err.message);
  }
}

function animateAlong(path) {
  // Cumulative distance along the path
  const cum = [0];
  for (let i = 1; i < path.length; i++) cum.push(cum[i - 1] + metres(path[i - 1], path[i]));
  const total = cum[cum.length - 1];
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const duration = reduce ? 0 : Math.min(2200, Math.max(450, 250 + total * 1.6));
  const base = S.trail.slice();
  map.panInside(path[path.length - 1], { padding: [80, 80] });

  return new Promise((resolve) => {
    const t0 = performance.now();
    function frame(now) {
      const t = duration ? Math.min(1, (now - t0) / duration) : 1;
      const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      const d = eased * total;
      let i = 1;
      while (i < path.length - 1 && cum[i] < d) i++;
      const seg = cum[i] - cum[i - 1] || 1;
      const f = Math.min(1, Math.max(0, (d - cum[i - 1]) / seg));
      const a = path[i - 1], b = path[i];
      const p = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
      const done = path.slice(1, i).concat([p]);
      S.trail = base.concat(done);
      trailLine.setLatLngs(S.trail);
      trailCasing.setLatLngs(S.trail);
      if (metres(a, b) > 0.5) setArrow(p, bearing(a, b));
      if (t < 1) requestAnimationFrame(frame);
      else {
        S.trail = base.concat(path.slice(1));
        trailLine.setLatLngs(S.trail);
        trailCasing.setLatLngs(S.trail);
        resolve();
      }
    }
    requestAnimationFrame(frame);
  });
}

// ---------------------------------------------------------------------------
// Undo / reset
// ---------------------------------------------------------------------------
function undo() {
  if (!S || animating) return;
  if (!history.length) return reset();
  S = history.pop();
  trailLine.setLatLngs(S.trail);
  trailCasing.setLatLngs(S.trail);
  setArrow(S.pos, S.heading);
  if (S.route.length > 1) refreshPrediction();
  else clearPrediction();
  loadChoices();
}

function reset() {
  if (animating) return;
  choiceSeq++;
  predSeq++;
  S = null;
  history = [];
  choiceLayer.clearLayers();
  trailLine.setLatLngs([]);
  trailCasing.setLatLngs([]);
  arrow.remove();
  clearPrediction();
  updateUI();
}

$("undo").addEventListener("click", undo);
$("reset").addEventListener("click", reset);
document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, select")) return;
  if (e.key === "Backspace" || (e.key === "z" && (e.ctrlKey || e.metaKey))) { e.preventDefault(); undo(); }
  if (e.key === "Escape") reset();
});

// ---------------------------------------------------------------------------
// Prediction: heatmap squares, place pins, lists
// ---------------------------------------------------------------------------
async function refreshPrediction() {
  const id = ++predSeq;
  try {
    const res = await api("/api/predict", { route: S.route, settings: settings() });
    if (id === predSeq) showPrediction(res);
  } catch (err) {
    if (id === predSeq) toast(err.message);
  }
}

function clearPrediction() {
  lastPrediction = null;
  heatLayer.clearLayers();
  placeLayer.clearLayers();
  $("places").innerHTML = "";
  $("places-section").hidden = true;
  $("legend-section").hidden = true;
}

// One-hue sequential ramp (orange), light -> dark on the light map and
// dark -> bright on the dark map, so "unlikely" always recedes into the map.
const RAMP = {
  light: ["#fde7d9", "#fbc7a6", "#f6a074", "#eb6834", "#c94f1d", "#9a3a12", "#6b2709"],
  dark: ["#4a2414", "#7a3516", "#a8461c", "#d95926", "#ef7d47", "#f8a77a", "#fdd2b5"],
};
function rampColor(t) {
  const r = RAMP[dark() ? "dark" : "light"];
  const x = Math.min(1, Math.max(0, t)) * (r.length - 1);
  const i = Math.min(r.length - 2, Math.floor(x));
  const f = x - i;
  const a = parseInt(r[i].slice(1), 16), b = parseInt(r[i + 1].slice(1), 16);
  const ch = (s) => Math.round(((a >> s) & 255) * (1 - f) + ((b >> s) & 255) * f);
  return `rgb(${ch(16)},${ch(8)},${ch(0)})`;
}

function showPrediction(res) {
  lastPrediction = res;
  drawHeat(res);
  drawPlaces(res);
}

function drawHeat(res) {
  heatLayer.clearLayers();
  const cells = res.cells;
  if (!cells.length) return;
  const max = cells[0].p;
  const cellM = settings().cell;
  // Shade by sqrt(p / max): probability is spread thin, and a linear scale
  // would leave everything but the top few squares invisible.
  for (const c of cells) {
    const t = Math.sqrt(c.p / max);
    if (t < 0.06) break; // sorted by p: the rest are too faint to see
    const poly = L.polygon(c.corners, {
      renderer: heatRenderer, pane: "heat", stroke: false,
      fillColor: rampColor(t), fillOpacity: 0.06 + 0.64 * t,
    });
    poly.bindTooltip(`${pct(c.p)} chance the destination is in this ${cellM} m square`, { sticky: true });
    heatLayer.addLayer(poly);
  }
  // Legend
  const stops = [0, 0.25, 0.5, 0.75, 1].map((t) => `${rampColor(t)} ${t * 100}%`).join(", ");
  $("legend-bar").style.background = `linear-gradient(to right, ${stops})`;
  $("legend-mid").textContent = pct(max / 4);
  $("legend-max").textContent = pct(max);
  $("legend-note").textContent = `Chance the destination is in each ${cellM} m square (colour scale is square-root, so faint squares still show).`;
  $("legend-section").hidden = false;
}

function drawPlaces(res) {
  placeLayer.clearLayers();
  const list = $("places");
  list.innerHTML = "";
  if (!res.places.length) {
    $("places-section").hidden = true;
    return;
  }
  const top = res.places.slice(0, 10);
  const max = top[0].p;
  top.forEach((pl, i) => {
    const name = pl.name || pl.category;
    const cat = pl.category.replace(/^[a-z]+=/, "").replace(/_/g, " ");
    const pin = L.marker(pl.position, {
      icon: L.divIcon({ className: "", html: `<div class="place-pin">${i + 1}</div>`, iconSize: [24, 24], iconAnchor: [12, 12] }),
      zIndexOffset: 500 - i,
    }).bindPopup(`<b>${esc(name)}</b><br>${esc(cat)} · ${pct(pl.p)}`);
    if (i < 5) placeLayer.addLayer(pin);

    const li = document.createElement("li");
    li.innerHTML = `<span class="name" title="${esc(name)}">${esc(name)}</span><span class="pct">${pct(pl.p)}</span>
      <span class="cat">${esc(cat)}</span><span></span>
      <div class="bar"><span style="width:${(100 * pl.p) / max}%"></span></div>`;
    li.addEventListener("click", () => {
      map.flyTo(pl.position, Math.max(map.getZoom(), 16));
      if (!placeLayer.hasLayer(pin)) placeLayer.addLayer(pin);
      pin.openPopup();
    });
    list.appendChild(li);
  });
  $("elsewhere").textContent = res.elsewhere != null
    ? `${pct(res.elsewhere)} somewhere without a landmark (homes, small shops, …).` : "";
  $("places-section").hidden = false;
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
(async () => {
  drawTrail();
  try {
    info = await api("/api/info");
    map.fitBounds(info.bounds, { padding: [20, 20] });
    for (const d of info.days.filter((d) => d !== "weekday")) {
      const o = document.createElement("option");
      o.value = d;
      o.textContent = d[0].toUpperCase() + d.slice(1);
      $("day").appendChild(o);
    }
    if (!info.landmarks) {
      $("landmarks").checked = false;
      $("landmarks").disabled = true;
      $("landmarks").parentElement.title = "The server was started without landmarks";
    }
    updateUI();
  } catch (err) {
    $("instruction").textContent = "Cannot reach the server: " + err.message;
  }
})();
