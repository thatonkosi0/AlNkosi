// EXECUTOR tab: arm form, kill switch, per-strategy status, live intent feed.
import { api, toast } from "/js/app.js";

let terminal, armBtn, killBtn, errorBox, statusTable, idleNote, modeSel, confirmWrap, liveWarn;

function line(text) {
  if (!terminal) return;
  terminal.textContent += text + "\n";
  terminal.scrollTop = terminal.scrollHeight;
}

const DIR = (d) => (d === 1 ? "LONG" : d === -1 ? "SHORT" : "");

export function executorHandleEvent(event) {
  if (!terminal) return;
  if (event.type === "executor_intent") {
    const lots = event.lots ? `${event.lots.toFixed(2)}lot ` : "";
    line(`   [${event.symbol}#${event.magic}] ${event.action.toUpperCase()} ${DIR(event.direction)} ${lots}— ${event.reason}`);
    if (["open_long", "open_short", "close", "halt"].includes(event.action)) {
      refreshExecutorStatus().catch(() => {});
    }
  } else if (event.type === "executor_error") {
    line(`   !! executor error: ${event.error}`);
  }
}

function renderStatus(s) {
  if (!statusTable) return;
  const running = Boolean(s.running || s.armed);
  killBtn.disabled = !running;
  armBtn.disabled = running;
  const rows = s.strategies || [];
  idleNote.hidden = rows.length > 0;
  const tbody = statusTable.querySelector("tbody");
  tbody.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${r.magic}</td><td>${r.symbol}</td><td>${r.open_positions}</td>` +
      `<td>${r.halted ? "YES" : "no"}</td><td>${(r.peak_equity || 0).toFixed(2)}</td>`;
    tbody.appendChild(tr);
  }
}

export async function refreshExecutorStatus() {
  renderStatus(await api("/api/executor/status"));
}

export function initExecutor() {
  terminal = document.getElementById("executor-terminal");
  armBtn = document.getElementById("arm-btn");
  killBtn = document.getElementById("kill-btn");
  errorBox = document.getElementById("executor-error");
  statusTable = document.getElementById("executor-status-table");
  idleNote = document.getElementById("executor-idle");
  modeSel = document.getElementById("exec-mode");
  confirmWrap = document.getElementById("confirm-live-wrap");
  liveWarn = document.getElementById("live-warning");

  modeSel.addEventListener("change", () => {
    const live = modeSel.value === "live";
    confirmWrap.hidden = !live;
    liveWarn.hidden = !live;
  });

  document.getElementById("executor-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    errorBox.hidden = true;
    const form = new FormData(e.target);
    const ids = (form.get("strategy_ids") || "")
      .split(",")
      .map((s) => parseInt(s.trim(), 10))
      .filter((n) => !Number.isNaN(n));
    if (ids.length === 0) {
      errorBox.textContent = "Enter at least one vault strategy id.";
      errorBox.hidden = false;
      return;
    }
    const payload = {
      strategy_ids: ids,
      mode: form.get("mode"),
      preset: form.get("preset"),
      poll_seconds: Number(form.get("poll_seconds")) || 5,
      confirm_live: form.get("confirm_live") === "on",
    };
    const cap = form.get("max_open_positions");
    if (cap) payload.max_open_positions = Number(cap);
    try {
      const s = await api("/api/executor/arm", { method: "POST", body: JSON.stringify(payload) });
      line(`>> armed ${ids.length} strateg${ids.length > 1 ? "ies" : "y"} in ${payload.mode.toUpperCase()} mode`);
      renderStatus(s);
      toast(`Executor armed (${payload.mode}).`);
    } catch (err) {
      errorBox.textContent = err.message;
      errorBox.hidden = false;
    }
  });

  killBtn.addEventListener("click", async () => {
    try {
      await api("/api/executor/stop", { method: "POST" });
      line(">> KILL — all strategies flattened and halted");
      toast("Executor stopped.");
      await refreshExecutorStatus();
    } catch (err) {
      toast(err.message);
    }
  });

  refreshExecutorStatus().catch(() => {});
}
