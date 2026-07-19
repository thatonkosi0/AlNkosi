// VAULT tab: strategy table, detail drawer, deploy flow.
import { api, toast } from "/js/app.js";

const PCT = (x) => `${(x * 100).toFixed(1)}%`;
const NUM = (x) => (x === null || x === undefined ? "-" : x.toFixed(2));

function cls(x) { return x >= 0 ? "pos" : "neg"; }

export async function refreshVault() {
  const params = new URLSearchParams();
  const symbol = document.getElementById("vf-symbol").value.trim();
  const tribe = document.getElementById("vf-tribe").value;
  const sort = document.getElementById("vf-sort").value;
  if (symbol) params.set("symbol", symbol.toUpperCase());
  if (tribe) params.set("tribe", tribe);
  params.set("sort", sort);
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
      <td>${r.name}</td><td>${r.tribe}</td><td>${r.symbol}</td><td>${r.timeframe}</td>
      <td class="${cls(r.oos_net_profit)}">${PCT(r.oos_net_profit)}</td>
      <td>${NUM(r.oos_profit_factor)}</td>
      <td>${PCT(r.oos_max_drawdown)}</td>
      <td>${NUM(r.oos_sharpe)}</td>
      <td>${r.oos_trades}</td>`;
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
    <h3 style="color:var(--primary);letter-spacing:2px">${row.name} — ${row.symbol} ${row.timeframe}</h3>
    ${geneTags(genome)}
    <div class="metric-cols">
      <div><h4>IN-SAMPLE</h4>${metricsTable(row.is_metrics)}</div>
      <div><h4>OUT-OF-SAMPLE (forward test)</h4>${metricsTable(row.oos_metrics)}</div>
    </div>
    <h4 style="color:var(--text-dim)">OOS EQUITY</h4>
    <canvas id="detail-spark" height="120"></canvas>
    <div class="control-actions">
      <button class="btn-primary" id="deploy-btn">⇪ DEPLOY TO MT5</button>
      <button class="btn-danger" id="delete-btn">✕ DELETE</button>
    </div>
    <div id="deploy-result" class="terminal" style="height:auto;min-height:40px;margin-top:10px" hidden></div>`;
  panel.hidden = false;
  drawSparkline(document.getElementById("detail-spark"), row.equity);

  document.getElementById("deploy-btn").addEventListener("click", () => deployFlow(sid));
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
    out.textContent = `EA written to:\n${res.path}\n\n${res.instructions}`;
    toast("EA generated — compile it in MetaEditor (F7).");
  } catch (err) {
    out.hidden = false;
    out.textContent = `DEPLOY FAILED: ${err.message}`;
  }
}

export function initVault() {
  document.getElementById("vf-apply").addEventListener("click", () =>
    refreshVault().catch((e) => toast(e.message)));
  document.getElementById("detail-close").addEventListener("click", () => {
    document.getElementById("vault-detail").hidden = true;
  });
}
