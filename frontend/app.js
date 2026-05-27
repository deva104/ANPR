const API = "";
const WS_URL = `ws://${window.location.host}/ws`;

let searchDebounceTimer = null;

document.addEventListener("DOMContentLoaded", () => {
  loadStats();
  loadLogs();
  connectWebSocket();
  setupSearch();
  setupExport();
  setupRegisterBtn();
});

async function loadStats() {
  try {
    const res = await fetch(`${API}/api/stats`);
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    document.getElementById("allowed-count").textContent = data.allowed ?? 0;
    document.getElementById("denied-count").textContent = data.denied ?? 0;
    document.getElementById("passes-count").textContent = data.active_passes ?? 0;
    document.getElementById("expiring-count").textContent = data.expiring_soon ?? 0;
  } catch (err) {
    console.error("loadStats:", err);
  }
}

async function loadLogs(search = "") {
  try {
    const url = search.trim()
      ? `${API}/api/logs/search/${encodeURIComponent(search.trim())}`
      : `${API}/api/logs?limit=50`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    renderLogs(data);
  } catch (err) {
    console.error("loadLogs:", err);
  }
}

function formatLogTime(detectedOn) {
  if (!detectedOn) return "—";
  const d = new Date(detectedOn);
  if (Number.isNaN(d.getTime())) return detectedOn;
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())} ${pad(d.getDate())}/${pad(d.getMonth() + 1)}`;
}

function createLogRow(log) {
  const tr = document.createElement("tr");
  const granted = log.access_granted === true;
  tr.className = granted ? "row-allowed" : "row-denied";

  const statusBadge = granted
    ? '<span class="badge badge-allowed">✅ Allowed</span>'
    : '<span class="badge badge-denied">❌ Denied</span>';

  tr.innerHTML = `
    <td>${formatLogTime(log.detected_on)}</td>
    <td>${escapeHtml(log.plate_number || "—")}</td>
    <td>${escapeHtml(log.owner_name || "Unknown")}</td>
    <td>${escapeHtml(log.vehicle_type || "Unknown")}</td>
    <td>${statusBadge}</td>
    <td>${escapeHtml(log.denial_reason || "—")}</td>
  `;
  return tr;
}

function renderLogs(logs) {
  const tbody = document.getElementById("logs-tbody");
  tbody.innerHTML = "";
  if (!logs || logs.length === 0) {
    const tr = document.createElement("tr");
    tr.className = "empty-row";
    tr.innerHTML = '<td colspan="6">No access logs yet</td>';
    tbody.appendChild(tr);
    return;
  }
  logs.forEach((log) => tbody.appendChild(createLogRow(log)));
}

function connectWebSocket() {
  const ws = new WebSocket(WS_URL);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "detection") {
        updateDetectionCard(data);
        prependLogRow(data);
        loadStats();
        if (data.reason === "not_registered") {
          showUnknownPopup(data.plate_number);
        }
      }
    } catch (err) {
      console.error("WebSocket message:", err);
    }
  };

  ws.onclose = () => {
    setTimeout(connectWebSocket, 3000);
  };

  ws.onerror = (err) => {
    console.error("WebSocket error:", err);
  };
}

function updateDetectionCard(data) {
  const card = document.getElementById("detection-card");
  const statusEl = document.getElementById("detection-status");
  const plateEl = document.getElementById("detection-plate");
  const detailsEl = document.getElementById("detection-details");
  const registerBtn = document.getElementById("register-btn");

  card.classList.remove("allowed", "denied", "expired", "fade-in");
  void card.offsetWidth;

  if (data.granted) {
    card.classList.add("allowed");
    statusEl.textContent = "✅ ALLOWED";
    registerBtn.classList.add("hidden");
  } else if (data.reason === "expired") {
    card.classList.add("expired");
    statusEl.textContent = "⚠️ PASS EXPIRED";
    registerBtn.classList.add("hidden");
  } else if (data.reason === "not_registered") {
    card.classList.add("denied");
    statusEl.textContent = "❌ UNKNOWN VEHICLE";
    registerBtn.classList.remove("hidden");
    registerBtn.onclick = () => {
      window.location.href = `/vehicles?plate=${encodeURIComponent(data.plate_number)}`;
    };
  } else {
    card.classList.add("denied");
    statusEl.textContent = "❌ DENIED";
    registerBtn.classList.add("hidden");
  }

  plateEl.textContent = data.plate_number || "—";

  let details = "";
  if (data.owner_name && data.owner_name !== "Unknown") {
    details += `Owner: ${data.owner_name}`;
  }
  if (data.vehicle_type) {
    details += details ? ` | Type: ${data.vehicle_type}` : `Type: ${data.vehicle_type}`;
  }
  if (data.time) {
    details += details ? ` | ${data.time}` : data.time;
  }
  detailsEl.textContent = details;

  card.classList.add("fade-in");
}

function prependLogRow(data) {
  const tbody = document.getElementById("logs-tbody");
  const empty = tbody.querySelector(".empty-row");
  if (empty) empty.remove();

  const log = {
    detected_on: new Date().toISOString(),
    plate_number: data.plate_number,
    owner_name: data.owner_name,
    vehicle_type: data.vehicle_type,
    access_granted: data.granted,
    denial_reason: data.reason,
  };

  const tr = createLogRow(log);
  tbody.insertBefore(tr, tbody.firstChild);

  while (tbody.rows.length > 50) {
    tbody.removeChild(tbody.lastChild);
  }
}

function showUnknownPopup(plate) {
  const popup = document.getElementById("unknown-popup");
  document.getElementById("popup-plate").textContent = plate;

  document.getElementById("popup-register-btn").onclick = () => {
    window.location.href = `/vehicles?plate=${encodeURIComponent(plate)}`;
  };

  document.getElementById("popup-close-btn").onclick = () => {
    popup.classList.add("hidden");
  };

  popup.classList.remove("hidden");
}

function setupRegisterBtn() {
  const btn = document.getElementById("register-btn");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const plate = document.getElementById("detection-plate").textContent;
    if (plate && plate !== "---") {
      window.location.href = `/vehicles?plate=${encodeURIComponent(plate)}`;
    }
  });
}

function setupSearch() {
  const input = document.getElementById("log-search");
  if (!input) return;
  input.addEventListener("input", () => {
    clearTimeout(searchDebounceTimer);
    searchDebounceTimer = setTimeout(() => {
      loadLogs(input.value);
    }, 500);
  });
}

function setupExport() {
  const btn = document.getElementById("export-btn");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    try {
      const res = await fetch(`${API}/api/logs?limit=1000`);
      if (!res.ok) throw new Error(res.statusText);
      const logs = await res.json();
      const header = ["Time", "Plate", "Owner", "Type", "Status", "Reason"];
      const rows = logs.map((log) => [
        formatLogTime(log.detected_on),
        log.plate_number || "",
        log.owner_name || "",
        log.vehicle_type || "",
        log.access_granted ? "Allowed" : "Denied",
        log.denial_reason || "",
      ]);
      const csv = [header, ...rows]
        .map((row) =>
          row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(",")
        )
        .join("\n");
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "access_logs.csv";
      link.click();
      URL.revokeObjectURL(link.href);
    } catch (err) {
      console.error("export:", err);
      alert("Failed to export logs.");
    }
  });
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
