const API = "";
const WS_URL = `ws://${window.location.host}/ws`;

let allVehicles = [];
let currentFilter = "all";
let editingPlate = null;
let searchDebounceTimer = null;

document.addEventListener("DOMContentLoaded", () => {
  loadVehicles();
  setupForm();
  setupFilters();
  setupSearch();
  checkPrefilledPlate();
  connectWebSocket();
});

function checkPrefilledPlate() {
  const params = new URLSearchParams(window.location.search);
  const plate = params.get("plate");
  if (!plate) return;
  const input = document.getElementById("plate-input");
  input.value = plate.toUpperCase();
  input.classList.add("highlight");
  input.scrollIntoView({ behavior: "smooth", block: "center" });
  setTimeout(() => input.classList.remove("highlight"), 2000);
}

async function loadVehicles(filter = currentFilter, search = "") {
  currentFilter = filter;
  try {
    const res = await fetch(`${API}/api/vehicles`);
    if (!res.ok) throw new Error(res.statusText);
    allVehicles = await res.json();
    let filtered = [...allVehicles];
    if (filter !== "all") {
      filtered = filtered.filter(
        (v) => (v.vehicle_type || "").toLowerCase() === filter
      );
    }
    const q = search.trim().toUpperCase();
    if (q) {
      filtered = filtered.filter((v) =>
        (v.plate_number || "").toUpperCase().includes(q)
      );
    }
    renderVehicles(filtered);
  } catch (err) {
    console.error("loadVehicles:", err);
    showFormMessage("Failed to load vehicles.", "error");
  }
}

function parseDate(value) {
  if (!value) return null;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

function formatDateOnly(value) {
  const d = parseDate(value);
  if (!d) return "—";
  return d.toLocaleDateString("en-GB");
}

function validUntilCell(vehicle) {
  const vtype = (vehicle.vehicle_type || "").toLowerCase();
  if (vtype === "owner" || vtype === "renter") {
    return "Always";
  }
  const until = parseDate(vehicle.valid_until);
  if (!until) return "—";
  const now = new Date();
  const twoDays = new Date(now.getTime() + 2 * 24 * 60 * 60 * 1000);
  const text = formatDateOnly(vehicle.valid_until);
  if (until < now) {
    return `<span class="text-expired">${text} (Expired)</span>`;
  }
  if (until <= twoDays) {
    return `<span class="text-expiring">${text} (Soon)</span>`;
  }
  return text;
}

function typeBadge(type) {
  const t = (type || "unknown").toLowerCase();
  return `<span class="badge badge-${t}">${escapeHtml(type || "—")}</span>`;
}

function renderVehicles(vehicles) {
  const tbody = document.getElementById("vehicles-tbody");
  tbody.innerHTML = "";
  if (!vehicles.length) {
    const tr = document.createElement("tr");
    tr.className = "empty-row";
    tr.innerHTML = '<td colspan="6">No vehicles registered</td>';
    tbody.appendChild(tr);
    return;
  }
  vehicles.forEach((vehicle) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(vehicle.plate_number || "—")}</td>
      <td>${escapeHtml(vehicle.owner_name || "—")}</td>
      <td>${typeBadge(vehicle.vehicle_type)}</td>
      <td>${validUntilCell(vehicle)}</td>
      <td>${escapeHtml(vehicle.purpose || "—")}</td>
      <td class="action-cell">
        <button type="button" class="btn-edit">Edit</button>
        <button type="button" class="btn-danger">Delete</button>
      </td>
    `;
    const editBtn = tr.querySelector(".btn-edit");
    const delBtn = tr.querySelector(".btn-danger");
    const plateNum = vehicle.plate_number;
    editBtn.addEventListener("click", () => editVehicle(plateNum));
    delBtn.addEventListener("click", () => deleteVehicle(plateNum));
    tbody.appendChild(tr);
  });
}

function setupForm() {
  const typeSelect = document.getElementById("type-select");
  const temporal = document.getElementById("temporal-fields");
  const today = new Date().toISOString().slice(0, 10);

  typeSelect.addEventListener("change", () => {
    const v = typeSelect.value;
    if (v === "visitor" || v === "relative") {
      temporal.classList.remove("hidden");
      document.getElementById("valid-from-input").min = today;
      document.getElementById("valid-until-input").min = today;
    } else {
      temporal.classList.add("hidden");
    }
  });

  document.getElementById("add-vehicle-btn").addEventListener("click", submitVehicle);
}

async function submitVehicle() {
  const plate = document.getElementById("plate-input").value.trim().toUpperCase();
  const name = document.getElementById("name-input").value.trim();
  const type = document.getElementById("type-select").value;
  const validFrom = document.getElementById("valid-from-input").value || null;
  const validUntil = document.getElementById("valid-until-input").value || null;
  const purpose = document.getElementById("purpose-input").value.trim() || null;

  if (!plate) {
    showFormMessage("Plate number required", "error");
    return;
  }
  if (!name) {
    showFormMessage("Owner name required", "error");
    return;
  }
  if ((type === "visitor" || type === "relative") && !validUntil) {
    showFormMessage("Valid until date required", "error");
    return;
  }

  const body = {
    plate_number: plate,
    owner_name: name,
    vehicle_type: type,
    valid_from: type === "visitor" || type === "relative" ? validFrom : null,
    valid_until: type === "visitor" || type === "relative" ? validUntil : null,
    purpose: type === "visitor" || type === "relative" ? purpose : null,
  };

  try {
    let res;
    if (editingPlate) {
      res = await fetch(`${API}/api/vehicles/${encodeURIComponent(editingPlate)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          owner_name: body.owner_name,
          vehicle_type: body.vehicle_type,
          valid_from: body.valid_from,
          valid_until: body.valid_until,
          purpose: body.purpose,
        }),
      });
    } else {
      res = await fetch(`${API}/api/vehicles`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    }

    if (res.status === 409) {
      showFormMessage("Plate already registered", "error");
      return;
    }
    if (!res.ok) {
      const errText = await res.text();
      showFormMessage(`Error: ${res.status} ${errText}`, "error");
      return;
    }

    showFormMessage(
      editingPlate ? "Vehicle updated successfully" : "Vehicle added successfully",
      "success"
    );
    clearForm();
    loadVehicles();
  } catch (err) {
    console.error("submitVehicle:", err);
    showFormMessage("Request failed. Check backend connection.", "error");
  }
}

function clearForm() {
  editingPlate = null;
  document.getElementById("plate-input").value = "";
  document.getElementById("plate-input").disabled = false;
  document.getElementById("name-input").value = "";
  document.getElementById("type-select").value = "owner";
  document.getElementById("valid-from-input").value = "";
  document.getElementById("valid-until-input").value = "";
  document.getElementById("purpose-input").value = "";
  document.getElementById("temporal-fields").classList.add("hidden");
  document.getElementById("add-vehicle-btn").textContent = "Add Vehicle";
}

function editVehicle(plate) {
  const vehicle = allVehicles.find(
    (v) => (v.plate_number || "").toUpperCase() === plate.toUpperCase()
  );
  if (!vehicle) return;

  editingPlate = vehicle.plate_number;
  document.getElementById("plate-input").value = vehicle.plate_number;
  document.getElementById("plate-input").disabled = true;
  document.getElementById("name-input").value = vehicle.owner_name || "";
  document.getElementById("type-select").value = vehicle.vehicle_type || "owner";
  document.getElementById("type-select").dispatchEvent(new Event("change"));

  if (vehicle.valid_from) {
    document.getElementById("valid-from-input").value = String(vehicle.valid_from).slice(0, 10);
  }
  if (vehicle.valid_until) {
    document.getElementById("valid-until-input").value = String(vehicle.valid_until).slice(0, 10);
  }
  document.getElementById("purpose-input").value = vehicle.purpose || "";
  document.getElementById("add-vehicle-btn").textContent = "Update Vehicle";
  document.getElementById("plate-input").scrollIntoView({ behavior: "smooth", block: "center" });
}

async function deleteVehicle(plate) {
  if (!confirm(`Remove ${plate} from database?`)) return;
  try {
    const res = await fetch(`${API}/api/vehicles/${encodeURIComponent(plate)}`, {
      method: "DELETE",
    });
    if (!res.ok) throw new Error(res.statusText);
    if (editingPlate === plate) clearForm();
    loadVehicles();
    showFormMessage("Vehicle removed.", "success");
  } catch (err) {
    console.error("deleteVehicle:", err);
    showFormMessage("Failed to delete vehicle.", "error");
  }
}

function setupFilters() {
  document.querySelectorAll(".filter-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const search = document.getElementById("vehicle-search").value;
      loadVehicles(btn.dataset.filter, search);
    });
  });
}

function setupSearch() {
  const input = document.getElementById("vehicle-search");
  if (!input) return;
  input.addEventListener("input", () => {
    clearTimeout(searchDebounceTimer);
    searchDebounceTimer = setTimeout(() => {
      loadVehicles(currentFilter, input.value);
    }, 500);
  });
}

function showFormMessage(text, type) {
  const el = document.getElementById("form-message");
  el.textContent = text;
  el.className = type === "success" ? "success" : "error";
}

function connectWebSocket() {
  const ws = new WebSocket(WS_URL);
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "detection" && data.reason === "not_registered") {
        showUnknownPopup(data.plate_number);
      }
    } catch (err) {
      console.error("WebSocket:", err);
    }
  };
  ws.onclose = () => setTimeout(connectWebSocket, 3000);
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

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
