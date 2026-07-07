// INSIGHTS tab: horizontal bar charts of vault aggregates.
import { api } from "/js/app.js";

function drawBars(canvas, rows) {
  const ctx = canvas.getContext("2d");
  const w = (canvas.width = canvas.clientWidth || 400);
  const h = (canvas.height = Math.max(160, rows.length * 34 + 20));
  ctx.clearRect(0, 0, w, h);
  ctx.font = "11px Consolas, monospace";
  if (!rows.length) {
    ctx.fillStyle = "#7a626a";
    ctx.fillText("NO DATA — RUN A CAMPAIGN", 10, 30);
    return;
  }
  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.avg_oos_net_profit || 0)), 0.001);
  rows.forEach((r, i) => {
    const y = 14 + i * 34;
    const val = r.avg_oos_net_profit || 0;
    const barW = (Math.abs(val) / maxAbs) * (w - 150);
    ctx.fillStyle = "#d8c8cc";
    ctx.fillText(`${r.key} (${r.count})`, 4, y + 10);
    ctx.fillStyle = val >= 0 ? "#3ee87e" : "#ff6b6b";
    ctx.fillRect(120, y, Math.max(barW, 2), 14);
    ctx.fillStyle = "#7a626a";
    ctx.fillText(`${(val * 100).toFixed(1)}%`, 126 + barW, y + 11);
  });
}

export async function refreshInsights() {
  const data = await api("/api/insights");
  drawBars(document.getElementById("ins-symbol"), data.by_symbol);
  drawBars(document.getElementById("ins-timeframe"), data.by_timeframe);
  drawBars(document.getElementById("ins-tribe"), data.by_tribe);
}

export function initInsights() {
  // charts draw on tab activation via refreshInsights
}
