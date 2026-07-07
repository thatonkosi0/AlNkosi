// COMMAND DECK: canvas evolution field + telemetry + generation leader chart.
// Idle: vault strategies orbit in tribe clusters. Campaign: nodes implode,
// survivors divide, culled fall as embers, vaulted flash green.
import { api, fxMode } from "/js/app.js";

const TRIBE_COLORS = {
  trend: "#e8324f",
  breakout: "#e8b53e",
  mean_reversion: "#3e9de8",
  momentum: "#b43ee8",
};

let canvas, ctx, eqCanvas, terminal;
let nodes = [];        // {x,y,tx,ty,angle,radius,color,size,alpha,vx,vy,state}
let embers = [];       // falling culled nodes
let sparks = [];       // green vault sparks
let mode = "idle";     // idle | campaign
let campaignStart = null;
let raf = null;
let lastEventStats = {};

function resize() {
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
}

function center() {
  return { cx: canvas.width / 2, cy: canvas.height / 2 };
}

function tribeAnchor(tribe, i = 0, total = 4) {
  const { cx, cy } = center();
  const keys = Object.keys(TRIBE_COLORS);
  const idx = keys.indexOf(tribe);
  const angle = (idx / keys.length) * Math.PI * 2 - Math.PI / 4;
  const r = Math.min(cx, cy) * 0.5;
  return { x: cx + Math.cos(angle) * r, y: cy + Math.sin(angle) * r };
}

// ---- idle constellation ----------------------------------------------
export async function loadIdleConstellation() {
  try {
    const rows = await api("/api/vault");
    nodes = rows.map((r, i) => {
      const anchor = tribeAnchor(r.tribe);
      return {
        angle: Math.random() * Math.PI * 2,
        radius: 20 + Math.random() * 60,
        anchor,
        color: TRIBE_COLORS[r.tribe] || "#888",
        size: 2 + Math.min(Math.abs(r.oos_net_profit || 0) * 10, 4),
        alpha: 0.85,
        speed: 0.002 + Math.random() * 0.004,
      };
    });
  } catch (_) { nodes = []; }
}

// ---- campaign visuals --------------------------------------------------
function campaignNodes(population, tribe) {
  const { cx, cy } = center();
  nodes = Array.from({ length: population }, () => ({
    angle: Math.random() * Math.PI * 2,
    radius: 30 + Math.random() * 90,
    anchor: { x: cx, y: cy },
    color: TRIBE_COLORS[tribe] || "#e8324f",
    size: 2.5,
    alpha: 0.9,
    speed: 0.004 + Math.random() * 0.008,
    top: false,
  }));
}

function divideSurvivors(survivors, culled, topIdx) {
  const fx = fxMode();
  culled.forEach((i) => {
    const n = nodes[i];
    if (!n) return;
    if (fx !== "OFF") {
      embers.push({
        x: n.anchor.x + Math.cos(n.angle) * n.radius,
        y: n.anchor.y + Math.sin(n.angle) * n.radius,
        vx: (Math.random() - 0.5) * 0.6,
        vy: 0.4 + Math.random() * 0.8,
        alpha: 0.9,
        color: n.color,
      });
    }
  });
  // survivors pulse and spawn offspring beside them
  const survivorNodes = survivors.map((i) => nodes[i]).filter(Boolean);
  nodes = survivorNodes.flatMap((n) => {
    const parent = { ...n, size: 3, top: false };
    const child = {
      ...n,
      radius: n.radius + 12 + Math.random() * 10,
      angle: n.angle + (Math.random() - 0.5) * 0.5,
      size: 1.8,
      alpha: 0.7,
      top: false,
    };
    return [parent, child];
  });
  if (nodes[topIdx]) nodes[topIdx].top = true;
  if (nodes[0]) nodes[0].top = true;
}

function vaultSpark() {
  if (fxMode() === "OFF") return;
  const { cx, cy } = center();
  sparks.push({ x: cx, y: cy, t: 0 });
}

// ---- render loop -------------------------------------------------------
function draw() {
  const fx = fxMode();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const { cx, cy } = center();

  // core
  const pulse = 6 + Math.sin(Date.now() / 400) * (mode === "campaign" ? 3 : 1);
  ctx.beginPath();
  ctx.arc(cx, cy, pulse, 0, Math.PI * 2);
  ctx.fillStyle = mode === "campaign" ? "#e8324f" : "#6e2038";
  if (fx === "FULL") { ctx.shadowColor = "#e8324f"; ctx.shadowBlur = 18; }
  ctx.fill();
  ctx.shadowBlur = 0;

  // nodes
  for (const n of nodes) {
    n.angle += n.speed;
    const x = n.anchor.x + Math.cos(n.angle) * n.radius;
    const y = n.anchor.y + Math.sin(n.angle) * n.radius;
    ctx.beginPath();
    ctx.arc(x, y, n.size, 0, Math.PI * 2);
    ctx.globalAlpha = n.alpha;
    ctx.fillStyle = n.top ? "#ffffff" : n.color;
    if (fx === "FULL" && n.top) { ctx.shadowColor = "#fff"; ctx.shadowBlur = 10; }
    ctx.fill();
    ctx.shadowBlur = 0;
    if (n.top) {
      ctx.beginPath();
      ctx.arc(x, y, n.size + 4, 0, Math.PI * 2);
      ctx.strokeStyle = "#ffffff88";
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  // embers fall
  embers = embers.filter((e) => e.alpha > 0.02);
  for (const e of embers) {
    e.x += e.vx; e.y += e.vy; e.alpha *= 0.97;
    ctx.globalAlpha = e.alpha;
    ctx.fillStyle = e.color;
    ctx.fillRect(e.x, e.y, 2, 2);
    ctx.globalAlpha = 1;
  }

  // vault sparks fly to top-right
  sparks = sparks.filter((s) => s.t < 1);
  for (const s of sparks) {
    s.t += 0.03;
    const x = s.x + (canvas.width - 40 - s.x) * s.t;
    const y = s.y + (30 - s.y) * s.t;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = "#3ee87e";
    if (fx === "FULL") { ctx.shadowColor = "#3ee87e"; ctx.shadowBlur = 10; }
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  raf = requestAnimationFrame(draw);
}

function startLoop() {
  if (raf) cancelAnimationFrame(raf);
  if (fxMode() === "OFF") {
    // one static draw per event instead of a continuous loop
    const once = () => { raf = null; };
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    draw();
    cancelAnimationFrame(raf);
    once();
    return;
  }
  raf = requestAnimationFrame(draw);
}

// ---- equity chart ------------------------------------------------------
function drawEquityChart(curves) {
  const w = (eqCanvas.width = eqCanvas.clientWidth || 300);
  const h = eqCanvas.height;
  const c = eqCanvas.getContext("2d");
  c.clearRect(0, 0, w, h);
  if (!curves || !curves.length) return;
  const all = curves.flatMap((cu) => cu.points);
  const min = Math.min(...all), max = Math.max(...all);
  const span = max - min || 1;
  const palette = ["#e8324f", "#3ee87e", "#3e9de8", "#e8b53e", "#b43ee8"];
  curves.forEach((cu, ci) => {
    c.strokeStyle = palette[ci % palette.length];
    c.lineWidth = 1;
    c.beginPath();
    cu.points.forEach((p, i) => {
      const x = (i / (cu.points.length - 1)) * w;
      const y = h - ((p - min) / span) * (h - 20) - 10;
      i === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
    });
    c.stroke();
    c.fillStyle = palette[ci % palette.length];
    c.font = "9px Consolas, monospace";
    c.fillText(cu.id, 4, 10 + ci * 10);
  });
}

// ---- telemetry ---------------------------------------------------------
const $ = (id) => document.getElementById(id);
function fmtDur(ms) {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

function updateStats(event) {
  if (event.type === "generation") {
    $("st-state").textContent = "EVOLVING";
    $("st-gen").textContent = `${event.gen}/${event.of}`;
    $("st-tribe").textContent = `${event.symbol} · ${event.tribe}`;
    $("st-survivors").textContent = event.survivors.length;
    $("st-fitness").textContent = event.top_fitness.toFixed(4);
    $("st-vaulted").textContent = event.vaulted_count_total;
    if (campaignStart && event.units_done) {
      const elapsed = Date.now() - campaignStart;
      $("st-elapsed").textContent = fmtDur(elapsed);
      const left = (elapsed / event.units_done) * (event.total_units - event.units_done);
      $("st-eta").textContent = fmtDur(left);
    }
  }
}

function logLine(line) {
  terminal.textContent += line + "\n";
  if (terminal.textContent.length > 8000) {
    terminal.textContent = terminal.textContent.slice(-6000);
  }
  terminal.scrollTop = terminal.scrollHeight;
}

// ---- event entry point ---------------------------------------------------
export function deckHandleEvent(event) {
  if (!canvas) return;
  switch (event.type) {
    case "hello":
      if (event.campaign) {
        mode = "campaign";
        campaignStart = Date.now();
        $("st-state").textContent = "EVOLVING";
      }
      break;
    case "campaign_started":
      mode = "campaign";
      campaignStart = Date.now();
      embers = []; sparks = [];
      campaignNodes(Math.min(event.config.population, 120), event.config.tribes[0]);
      logLine(`>> campaign #${event.campaign_id} initialized`);
      startLoop();
      break;
    case "log":
      logLine(`   ${event.line}`);
      break;
    case "generation": {
      updateStats(event);
      lastEventStats = event;
      if (nodes.length === 0 || nodes[0].anchor.x !== center().cx) {
        campaignNodes(Math.min(event.population, 120), event.tribe);
      }
      const scale = nodes.length / event.population;
      const survivors = event.survivors.map((i) => Math.floor(i * scale));
      const culled = event.culled.map((i) => Math.floor(i * scale));
      divideSurvivors(survivors, culled, 0);
      nodes.forEach((n) => { n.color = TRIBE_COLORS[event.tribe] || n.color; });
      drawEquityChart(event.top_equity);
      if (fxMode() === "OFF") startLoop();
      break;
    }
    case "strategy_vaulted":
      vaultSpark();
      logLine(`   ++ ${event.name} vaulted`);
      break;
    case "campaign_finished":
      mode = "idle";
      $("st-state").textContent = event.status.toUpperCase();
      logLine(`>> ${event.status}: ${event.vaulted_total} strategies vaulted`);
      loadIdleConstellation().then(startLoop);
      break;
  }
}

export function initDeck() {
  canvas = $("deck-canvas");
  ctx = canvas.getContext("2d");
  eqCanvas = $("equity-canvas");
  terminal = $("deck-terminal");
  resize();
  window.addEventListener("resize", resize);
  loadIdleConstellation().then(startLoop);
}
