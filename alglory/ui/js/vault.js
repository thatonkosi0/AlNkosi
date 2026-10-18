// VAULT tab: strategy table, detail drawer, deploy flow.
import { api, toast } from "/js/app.js";

const PCT = (x) => `${(x * 100).toFixed(1)}%`;
const NUM = (x) => (x === null || x === undefined ? "-" : x.toFixed(2));

// Escape text before it goes into innerHTML (names/notes come from the backend).
function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function cls(x) { return x >= 0 ? "pos" : "neg"; }

const STAGE_LABEL = {
  legacy: "LEGACY", holdout_passed: "HOLDOUT PASSED", on_demo: "ON DEMO",
  demo_passed: "DEMO PASSED", demo_failed: "DEMO FAILED",
};
const STAGE_CLS = { holdout_passed: "pill-ok", on_demo: "pill-warn", demo_passed: "pill-ok", demo_failed: "neg" };

function stageBadge(stage) {
  return `<span class="pill ${STAGE_CLS[stage] || ""}">${esc(STAGE_LABEL[stage] || stage)}</span>`;
}

function robustCell(r) {
  const g = r.robustness_grade;
  if (!g) return `<span title="not scored yet — click RESCORE">-</span>`;
  const color = g === "A" || g === "B" ? "pos" : g === "D" || g === "F" ? "neg" : "";
  const score = r.robustness_score == null ? "" : ` ${r.robustness_score.toFixed(2)}`;
  return `<span class="${color}" title="walk-forward robustness score">${g}${score}</span>`;
}

const REUSED_TEXT = "holdout reused — results are weaker evidence";

function reusedMark(r) {
  return r.holdout_reused
    ? ` <span class="pill pill-warn" title="${esc(REUSED_TEXT)}">REUSED</span>`
    : "";
}

function holdoutCells(r) {
  if (r.stage === "legacy" || r.ho_trades == null || r.ho_win_rate == null) {
    return `<td colspan="3" class="neg" title="never tested on an untouched holdout">legacy — no holdout</td>`;
  }
  return `<td>${PCT(r.ho_win_rate)}${reusedMark(r)}</td><td>${NUM(r.ho_profit_factor)}</td><td>${esc(r.ho_trades)}</td>`;
}

function isLegacy(row) {
  return row.stage === "legacy" || !row.holdout_json;
}

function holdoutDetail(row) {
  if (isLegacy(row)) {
    return `<div><h4>HOLDOUT (untouched data)</h4><p class="neg">legacy — no holdout</p></div>`;
  }
  let m = {};
  try { m = JSON.parse(row.holdout_json) || {}; } catch { m = {}; }
  const reused = row.holdout_reused
    ? `<p class="neg">${reusedMark(row)} ${esc(REUSED_TEXT)}</p>` : "";
  return `<div><h4>HOLDOUT (untouched data)</h4>${reused}<table>
    <tr><td>WIN RATE</td><td>${m.win_rate == null ? "-" : PCT(m.win_rate)}</td></tr>
    <tr><td>PROFIT FACTOR</td><td>${NUM(m.profit_factor)}</td></tr>
    <tr><td>TRADES</td><td>${esc(m.trades ?? "-")}</td></tr>
    <tr><td>NET PROFIT</td><td class="${cls(m.net_profit ?? 0)}">${m.net_profit == null ? "-" : PCT(m.net_profit)}</td></tr>
  </table></div>`;
}

function demoLine(row) {
  const d = row.demo || {};
  let text = "";
  if (d.trades != null) {
    text += ` demo: ${esc(d.trades)} trades, win ${PCT(d.win_rate || 0)}, net ${(d.net || 0).toFixed(2)} ${esc(d.note)}`;
  }
  if (isLegacy(row)) text += ` <span class="neg">legacy — no holdout</span>`;
  return text;
}

async function refreshDemo() {
  const out = document.getElementById("demo-panel");
  out.hidden = false;
  out.textContent = "Refreshing demo results from MT5…";
  try {
    const res = await api("/api/demo/refresh", { method: "POST" });
    const a = res.account;
    let text = `ACCOUNT  worst day ${PCT(a.worst_day_loss)}  max drawdown ${PCT(a.max_drawdown)}  ` +
      (a.breaches.length ? `BREACHES: ${a.breaches.join("; ")}` : "no breaches");
    if (!res.strategies.length) text += "\n(no strategies are on demo)";
    for (const s of res.strategies) {
      text += `\n${s.name}  ${STAGE_LABEL[s.stage] || s.stage}  trades ${s.trades}  ` +
        `win ${PCT(s.win_rate)}  net ${s.net.toFixed(2)}  ${s.note || ""}${s.legacy ? "  (legacy)" : ""}`;
    }
    out.textContent = text;
    refreshVault().catch((e) => toast(e.message));
  } catch (err) {
    out.textContent = `DEMO REFRESH FAILED: ${err.message}`;
  }
}

export async function refreshVault() {
  const params = new URLSearchParams();
  const symbol = document.getElementById("vf-symbol").value.trim();
  const tribe = document.getElementById("vf-tribe").value;
  const sort = document.getElementById("vf-sort").value;
  if (symbol) params.set("symbol", symbol.toUpperCase());
  if (tribe) params.set("tribe", tribe);
  params.set("sort", sort);
  const stage = document.getElementById("vf-stage").value;
  if (stage) params.set("stage", stage);
  // Picking the LEGACY stage explicitly must not be hidden by the toggle.
  const showLegacy = document.getElementById("vf-legacy").checked || stage === "legacy";
  params.set("include_legacy", showLegacy);
  const minProfit = document.getElementById("vf-minprofit").value;
  const minPf = document.getElementById("vf-minpf").value;
  const maxDd = document.getElementById("vf-maxdd").value;
  if (minProfit !== "") params.set("min_oos_net_profit", Number(minProfit) / 100);
  if (minPf !== "") params.set("min_oos_profit_factor", minPf);
  if (maxDd !== "") params.set("max_oos_max_drawdown", Number(maxDd) / 100);
  const rows = await api(`/api/vault?${params}`);
  const tbody = document.querySelector("#vault-table tbody");
  document.getElementById("vault-empty").hidden = rows.length > 0;
  tbody.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${esc(r.name)}</td><td>${stageBadge(r.stage)}</td>${holdoutCells(r)}
      <td>${esc(r.tribe)}</td><td>${esc(r.symbol)}</td><td>${esc(r.timeframe)}</td>
      <td class="${cls(r.oos_net_profit)}" title="select slice — used for selection">${PCT(r.oos_net_profit)}</td>
      <td>${NUM(r.oos_profit_factor)}</td>
      <td>${PCT(r.oos_max_drawdown)}</td>
      <td>${NUM(r.oos_sharpe)}</td>
      <td>${r.oos_trades}</td>
      <td>${robustCell(r)}</td>`;
    tr.addEventListener("click", () => showDetail(r.id));
    tbody.appendChild(tr);
  }
}

function geneTags(genome) {
  const s = genome.signal;
  const f = genome.filters;
  const m = genome.management;
  const sigParams = Object.entries(s.params).map(([k, v]) => `${k}=${v}`).join(" ");
  let html = `<div class="gene-group"><span class="gene-tag gene-signal">SIGNAL: ${s.kind} ${sigParams}</span>
    <span class="gene-tag gene-signal">DIR: ${genome.direction}</span></div><div class="gene-group">`;
  if (f.trend_ma) html += `<span class="gene-tag gene-filter">TREND MA ${f.trend_ma}</span>`;
  if (f.session) html += `<span class="gene-tag gene-filter">SESSION ${f.session[0]}-${f.session[1]}h</span>`;
  if (f.atr_regime) html += `<span class="gene-tag gene-filter">ATR ${f.atr_regime}</span>`;
  if (!f.trend_ma && !f.session && !f.atr_regime) html += `<span class="gene-tag gene-filter">NO FILTERS</span>`;
  html += `</div><div class="gene-group">
    <span class="gene-tag gene-mgmt">SL ${m.sl_atr.toFixed(2)} ATR</span>
    <span class="gene-tag gene-mgmt">TP ${m.tp_atr.toFixed(2)} ATR</span>`;
  if (m.trail_atr) html += `<span class="gene-tag gene-mgmt">TRAIL ${m.trail_atr.toFixed(2)} ATR</span>`;
  if (m.breakeven_atr) html += `<span class="gene-tag gene-mgmt">BE ${m.breakeven_atr.toFixed(2)} ATR</span>`;
  if (m.max_bars) html += `<span class="gene-tag gene-mgmt">MAX ${m.max_bars} BARS</span>`;
  html += `</div><div class="gene-group">
    <span class="gene-tag gene-risk">RISK ${(genome.risk.risk_pct * 100).toFixed(2)}%/trade</span></div>`;
  return html;
}

function metricsTable(m) {
  return `<table>
    <tr><td>NET PROFIT</td><td class="${cls(m.net_profit)}">${PCT(m.net_profit)}</td></tr>
    <tr><td>MAX DD</td><td>${PCT(m.max_drawdown)}</td></tr>
    <tr><td>PROFIT FACTOR</td><td>${NUM(m.profit_factor)}</td></tr>
    <tr><td>SHARPE</td><td>${NUM(m.sharpe)}</td></tr>
    <tr><td>WIN RATE</td><td>${PCT(m.win_rate)}</td></tr>
    <tr><td>TRADES</td><td>${m.trades}</td></tr>
    <tr><td>AVG YEARLY</td><td class="${cls(m.avg_yearly_return)}">${PCT(m.avg_yearly_return)}</td></tr>
  </table>`;
}

function drawSparkline(canvas, points) {
  const ctx = canvas.getContext("2d");
  const w = (canvas.width = canvas.clientWidth || 600);
  const h = (canvas.height = 120);
  ctx.clearRect(0, 0, w, h);
  if (!points.length) return;
  const min = Math.min(...points), max = Math.max(...points);
  const span = max - min || 1;
  ctx.strokeStyle = points[points.length - 1] >= 1 ? "#3ee87e" : "#ff6b6b";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  points.forEach((p, i) => {
    const x = (i / (points.length - 1)) * w;
    const y = h - ((p - min) / span) * (h - 10) - 5;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

async function showDetail(sid) {
  const row = await api(`/api/vault/${sid}`);
  const genome = JSON.parse(row.genome_json);
  const panel = document.getElementById("vault-detail");
  const body = document.getElementById("detail-body");
  body.innerHTML = `
    <h3 style="color:var(--primary);letter-spacing:2px">${esc(row.name)} — ${esc(row.symbol)} ${esc(row.timeframe)}</h3>
    ${geneTags(genome)}
    <p>${stageBadge(row.stage)}${reusedMark(row)}${demoLine(row)}</p>
    <div class="metric-cols">
      <div><h4>IN-SAMPLE</h4>${metricsTable(row.is_metrics)}</div>
      <div><h4>SELECT (used for selection)</h4>${metricsTable(row.oos_metrics)}</div>
      ${holdoutDetail(row)}
    </div>
    <h4 style="color:var(--text-dim)">SELECT EQUITY (used for selection)</h4>
    <canvas id="detail-spark" height="120"></canvas>
    <div class="control-actions">
      <button class="btn-primary" id="deploy-btn">⇪ DEPLOY TO MT5</button>
      <button class="pill" id="optimize-btn" title="breed a new campaign seeded from this genome">★ OPTIMIZE</button>
      <button class="btn-danger" id="delete-btn">✕ DELETE</button>
    </div>
    <div id="deploy-result" class="terminal" style="height:auto;min-height:40px;margin-top:10px" hidden></div>`;
  panel.hidden = false;
  drawSparkline(document.getElementById("detail-spark"), row.equity);

  document.getElementById("deploy-btn").addEventListener("click", () => deployFlow(sid));
  document.getElementById("optimize-btn").addEventListener("click", () => {
    document.dispatchEvent(new CustomEvent("alglory:optimize", {
      detail: {
        id: row.id, name: row.name, symbol: row.symbol,
        timeframe: row.timeframe, tribe: row.tribe,
      },
    }));
    toast(`Optimize seeded from ${row.name} — pick a data source and LAUNCH.`);
  });
  document.getElementById("delete-btn").addEventListener("click", async () => {
    if (!confirm(`Delete ${row.name} permanently?`)) return;
    await api(`/api/vault/${sid}`, { method: "DELETE" });
    panel.hidden = true;
    refreshVault();
  });
  panel.scrollIntoView({ behavior: "smooth" });
}

async function deployFlow(sid) {
  const preset = prompt(
    "Guardrail preset:\n  personal          (1% risk, 8% daily, 25% max DD)\n" +
    "  prop_conservative (0.5% risk, 3% daily, 8% max DD)\n" +
    "  prop_aggressive   (1% risk, 4% daily, 9% max DD)",
    "prop_conservative"
  );
  if (!preset) return;
  const out = document.getElementById("deploy-result");
  try {
    const res = await api(`/api/deploy/${sid}`, {
      method: "POST",
      body: JSON.stringify({ preset: preset.trim() }),
    });
    out.hidden = false;
    const where = res.mt5_detected
      ? "MT5 Experts folder (Navigator > Expert Advisors > Alglory)"
      : "a fallback folder — MT5 was NOT detected";
    out.textContent = `EA written to ${where}:\n${res.path}\n\n${res.instructions}`;
    toast(
      res.mt5_detected
        ? "EA in MT5 — compile it in MetaEditor (F7)."
        : "MT5 not detected — EA saved to fallback folder (see details)."
    );
  } catch (err) {
    out.hidden = false;
    out.textContent = `DEPLOY FAILED: ${err.message}`;
  }
}

export function initVault() {
  document.getElementById("vf-apply").addEventListener("click", () =>
    refreshVault().catch((e) => toast(e.message)));
  document.getElementById("vf-demo").addEventListener("click", refreshDemo);
  document.getElementById("detail-close").addEventListener("click", () => {
    document.getElementById("vault-detail").hidden = true;
  });
  document.getElementById("vf-rescore").addEventListener("click", async () => {
    const symbol = document.getElementById("vf-symbol").value.trim().toUpperCase();
    if (!symbol) { toast("Set a SYMBOL filter first (e.g. EURUSD)."); return; }
    const tf = prompt("Timeframe to rescore (M15 / H1 / H4 / D1):", "H1");
    if (!tf) return;
    toast("Rescoring — walk-forward across the vault. This can take a minute…");
    try {
      const out = await api(
        `/api/vault/rescore?symbol=${symbol}&timeframe=${tf.trim().toUpperCase()}`,
        { method: "POST" }
      );
      if (!out.scored) {
        toast(out.reason || "Nothing scored — no cached bars for that symbol/timeframe.");
        return;
      }
      toast(`Rescored ${out.scored} strateg${out.scored === 1 ? "y" : "ies"}. Sort by ROBUSTNESS ★ to rank them.`);
      document.getElementById("vf-sort").value = "robustness_score";
      refreshVault().catch((e) => toast(e.message));
    } catch (err) {
      toast(err.message);
    }
  });
}
