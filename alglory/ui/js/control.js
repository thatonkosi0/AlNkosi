// CONTROL tab: campaign form, cancel, live feed terminal.
import { api, toast, fxMode } from "/js/app.js";

let terminal, launchBtn, cancelBtn, errorBox;
let typeQueue = [];
let typing = false;

function typeLine(line) {
  if (fxMode() !== "FULL") {
    terminal.textContent += line + "\n";
    terminal.scrollTop = terminal.scrollHeight;
    return;
  }
  typeQueue.push(line);
  if (!typing) drainQueue();
}

function drainQueue() {
  const line = typeQueue.shift();
  if (line === undefined) { typing = false; return; }
  typing = true;
  let i = 0;
  const tick = () => {
    // type in chunks so long campaigns never lag behind
    const chunk = typeQueue.length > 3 ? line.length : 3;
    terminal.textContent += line.slice(i, i + chunk);
    i += chunk;
    if (i < line.length) { setTimeout(tick, 8); }
    else {
      terminal.textContent += "\n";
      terminal.scrollTop = terminal.scrollHeight;
      drainQueue();
    }
  };
  tick();
}

function setRunning(running) {
  launchBtn.disabled = running;
  cancelBtn.disabled = !running;
}

export function controlHandleEvent(event) {
  if (!terminal) return;
  switch (event.type) {
    case "hello":
      if (event.campaign) {
        setRunning(true);
        typeLine(`>> reattached to campaign #${event.campaign.campaign_id}`);
        (event.campaign.recent_events || []).forEach(controlHandleEvent);
      }
      break;
    case "campaign_started":
      setRunning(true);
      typeLine(`>> campaign #${event.campaign_id} started (${event.total_units} generation units)`);
      break;
    case "log":
      typeLine(`   ${event.line}`);
      break;
    case "generation":
      typeLine(
        `   [${event.symbol}/${event.tribe}] gen ${event.gen}/${event.of}` +
        ` fitness=${event.top_fitness.toFixed(4)} vaulted=${event.vaulted_count_total}`
      );
      break;
    case "strategy_vaulted":
      typeLine(`   ++ VAULTED ${event.name} (${event.tribe}, ${(event.oos_net_profit * 100).toFixed(1)}% OOS)`);
      break;
    case "campaign_finished":
      setRunning(false);
      typeLine(`>> campaign finished: ${event.status.toUpperCase()}${event.error ? " — " + event.error : ""}`);
      break;
  }
}

export function initControl() {
  terminal = document.getElementById("control-terminal");
  launchBtn = document.getElementById("launch-btn");
  cancelBtn = document.getElementById("cancel-btn");
  errorBox = document.getElementById("control-error");

  document.getElementById("campaign-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    errorBox.hidden = true;
    const form = new FormData(e.target);
    const tribes = form.getAll("tribes");
    if (tribes.length === 0) {
      errorBox.textContent = "Select at least one tribe.";
      errorBox.hidden = false;
      return;
    }
    const payload = {
      symbols: form.get("symbols").split(",").map((s) => s.trim().toUpperCase()).filter(Boolean),
      timeframe: form.get("timeframe"),
      source: form.get("source"),
      tribes,
      population: Number(form.get("population")),
      generations: Number(form.get("generations")),
      oos_fraction: Number(form.get("oos_fraction")),
      min_trades: Number(form.get("min_trades")),
      dd_cap: Number(form.get("dd_cap")),
    };
    if (form.get("seed")) payload.seed = Number(form.get("seed"));
    try {
      await api("/api/campaigns", { method: "POST", body: JSON.stringify(payload) });
      setRunning(true);
      location.hash = "#deck";
    } catch (err) {
      errorBox.textContent = err.message;
      errorBox.hidden = false;
    }
  });

  cancelBtn.addEventListener("click", async () => {
    try {
      await api("/api/campaigns/cancel", { method: "POST" });
      toast("Cancellation requested — finishing current generation.");
    } catch (err) {
      toast(err.message);
    }
  });

  document.addEventListener("alglory:status", (e) => {
    setRunning(e.detail.campaign.running);
  });
}
