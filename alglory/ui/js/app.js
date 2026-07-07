// App shell: tab router, API client, WebSocket, status polling, FX mode.
import { initDeck, deckHandleEvent } from "/js/deck.js";
import { initControl, controlHandleEvent } from "/js/control.js";
import { initVault, refreshVault } from "/js/vault.js";
import { initMap, refreshMap } from "/js/map.js";
import { initInsights, refreshInsights } from "/js/insights.js";

// ---- FX quality mode -------------------------------------------------
const FX_MODES = ["FULL", "LITE", "OFF"];
export function fxMode() {
  return localStorage.getItem("alglory_fx") || "FULL";
}
function cycleFx() {
  const next = FX_MODES[(FX_MODES.indexOf(fxMode()) + 1) % FX_MODES.length];
  localStorage.setItem("alglory_fx", next);
  document.getElementById("fx-toggle").textContent = `FX: ${next}`;
}

// ---- API helper ------------------------------------------------------
export async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* keep statusText */ }
    throw new Error(detail);
  }
  return res.json();
}

export function toast(msg, ms = 5000) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => { el.hidden = true; }, ms);
}

// ---- Tab router ------------------------------------------------------
const REFRESHERS = { vault: refreshVault, map: refreshMap, insights: refreshInsights };
function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((p) =>
    p.classList.toggle("active", p.id === `tab-${name}`));
  if (REFRESHERS[name]) REFRESHERS[name]().catch((e) => toast(e.message));
}
function routeFromHash() {
  showTab((location.hash || "#deck").slice(1));
}

// ---- WebSocket with reconnect ---------------------------------------
let backoff = 500;
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/events`);
  ws.onopen = () => { backoff = 500; };
  ws.onmessage = (msg) => {
    const event = JSON.parse(msg.data);
    deckHandleEvent(event);
    controlHandleEvent(event);
    if (event.type === "campaign_finished") {
      refreshVault().catch(() => {});
      toast(`Campaign ${event.status}${event.error ? ": " + event.error : ""} — ${event.vaulted_total} strategies vaulted`);
    }
  };
  ws.onclose = () => {
    setTimeout(connectWS, backoff);
    backoff = Math.min(backoff * 2, 10000);
  };
}

// ---- MT5 status pill -------------------------------------------------
async function pollStatus() {
  try {
    const s = await api("/api/status");
    const pill = document.getElementById("mt5-pill");
    if (s.mt5.connected) {
      pill.textContent = "MT5: LINKED";
      pill.className = "pill pill-ok";
    } else {
      pill.textContent = s.mt5.available ? "MT5: OFFLINE" : "MT5: NOT INSTALLED";
      pill.className = "pill pill-warn";
    }
    pill.title = s.mt5.message;
    document.dispatchEvent(new CustomEvent("alglory:status", { detail: s }));
  } catch (_) { /* server briefly unavailable */ }
}

// ---- boot ------------------------------------------------------------
document.getElementById("fx-toggle").textContent = `FX: ${fxMode()}`;
document.getElementById("fx-toggle").addEventListener("click", cycleFx);
window.addEventListener("hashchange", routeFromHash);

initDeck();
initControl();
initVault();
initMap();
initInsights();
routeFromHash();
connectWS();
pollStatus();
setInterval(pollStatus, 5000);
