// MAP tab: whole vault as a zoomable, pannable clustered galaxy.
import { api } from "/js/app.js";

const TRIBE_COLORS = {
  trend: "#e8324f",
  breakout: "#e8b53e",
  mean_reversion: "#3e9de8",
  momentum: "#b43ee8",
};

let canvas, ctx, callout;
let strategies = [];
let clusters = {};      // tribe -> {cx, cy, members: []}
let view = { x: 0, y: 0, zoom: 1 };
let dragging = false, dragStart = null;
let raf = null;

function layout() {
  clusters = {};
  const tribes = [...new Set(strategies.map((s) => s.tribe))];
  const w = canvas.width, h = canvas.height;
  tribes.forEach((tribe, i) => {
    const angle = (i / Math.max(tribes.length, 1)) * Math.PI * 2 - Math.PI / 3;
    const r = Math.min(w, h) * 0.28;
    clusters[tribe] = {
      cx: w / 2 + Math.cos(angle) * r,
      cy: h / 2 + Math.sin(angle) * r,
      members: [],
    };
  });
  strategies.forEach((s, i) => {
    const c = clusters[s.tribe];
    const angle = Math.random() * Math.PI * 2;
    const radius = 15 + Math.random() * 55;
    c.members.push({
      ...s,
      baseAngle: angle,
      radius,
      speed: 0.001 + Math.random() * 0.002,
      t: Math.random() * 1000,
    });
  });
}

function draw() {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.translate(view.x, view.y);
  ctx.scale(view.zoom, view.zoom);

  for (const [tribe, c] of Object.entries(clusters)) {
    // cluster halo
    ctx.beginPath();
    ctx.arc(c.cx, c.cy, 78, 0, Math.PI * 2);
    ctx.strokeStyle = (TRIBE_COLORS[tribe] || "#888") + "22";
    ctx.stroke();
    ctx.fillStyle = (TRIBE_COLORS[tribe] || "#888") + "99";
    ctx.font = "10px Consolas, monospace";
    ctx.fillText(tribe.toUpperCase().replace("_", " "), c.cx - 30, c.cy - 86);

    for (const m of c.members) {
      m.t += m.speed;
      const x = c.cx + Math.cos(m.baseAngle + m.t) * m.radius;
      const y = c.cy + Math.sin(m.baseAngle + m.t) * m.radius;
      m._x = x; m._y = y;
      ctx.beginPath();
      const size = 2 + Math.min(Math.abs(m.oos_net_profit || 0) * 8, 4);
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fillStyle = (m.oos_net_profit || 0) >= 0 ? (TRIBE_COLORS[tribe] || "#888") : "#5a3a42";
      ctx.fill();
    }
  }
  ctx.restore();
  raf = requestAnimationFrame(draw);
}

function screenToWorld(px, py) {
  return { x: (px - view.x) / view.zoom, y: (py - view.y) / view.zoom };
}

function clusterAt(px, py) {
  const { x, y } = screenToWorld(px, py);
  for (const [tribe, c] of Object.entries(clusters)) {
    if (Math.hypot(x - c.cx, y - c.cy) < 85) return [tribe, c];
  }
  return null;
}

function showCallout(tribe, c) {
  const members = c.members;
  const profits = members.map((m) => m.oos_net_profit || 0);
  const best = members[profits.indexOf(Math.max(...profits))];
  const worst = members[profits.indexOf(Math.min(...profits))];
  const avg = profits.reduce((a, b) => a + b, 0) / (profits.length || 1);
  const symbols = {};
  members.forEach((m) => { symbols[m.symbol] = (symbols[m.symbol] || 0) + 1; });
  const domSymbol = Object.entries(symbols).sort((a, b) => b[1] - a[1])[0];
  callout.innerHTML = `
    <h4>${tribe.toUpperCase().replace("_", " ")}</h4>
    <div><span>STRATEGIES</span><span>${members.length}</span></div>
    <div><span>SHARE OF VAULT</span><span>${((members.length / strategies.length) * 100).toFixed(0)}%</span></div>
    <div><span>AVG OOS</span><span>${(avg * 100).toFixed(1)}%</span></div>
    <div><span>BEST</span><span>${best ? best.name : "-"}</span></div>
    <div><span>WORST</span><span>${worst ? worst.name : "-"}</span></div>
    <div><span>DOMINANT MKT</span><span>${domSymbol ? domSymbol[0] : "-"}</span></div>`;
  callout.hidden = false;
}

export async function refreshMap() {
  strategies = await api("/api/vault");
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  layout();
  if (raf) cancelAnimationFrame(raf);
  raf = requestAnimationFrame(draw);
}

export function initMap() {
  canvas = document.getElementById("map-canvas");
  ctx = canvas.getContext("2d");
  callout = document.getElementById("map-callout");

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    const before = screenToWorld(e.offsetX, e.offsetY);
    view.zoom = Math.min(Math.max(view.zoom * factor, 0.4), 4);
    view.x = e.offsetX - before.x * view.zoom;
    view.y = e.offsetY - before.y * view.zoom;
  }, { passive: false });

  canvas.addEventListener("mousedown", (e) => {
    dragging = true;
    dragStart = { x: e.offsetX - view.x, y: e.offsetY - view.y };
    canvas.style.cursor = "grabbing";
  });
  canvas.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    view.x = e.offsetX - dragStart.x;
    view.y = e.offsetY - dragStart.y;
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
    if (canvas) canvas.style.cursor = "grab";
  });
  canvas.addEventListener("click", (e) => {
    const hit = clusterAt(e.offsetX, e.offsetY);
    if (hit) showCallout(hit[0], hit[1]);
    else callout.hidden = true;
  });
}
