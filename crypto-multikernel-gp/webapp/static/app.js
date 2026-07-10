const API_BASE = "";

const state = { asset: null, chart: null };

const statusMessages = [
  "fetching live market, on-chain, and sentiment data",
  "engineering fundamental, technical, and sentiment features",
  "fitting multi-kernel additive Gaussian process (SVI)",
  "computing predictive distribution and uncertainty decomposition",
];

const assetPicker = document.getElementById("assetPicker");
const runButton = document.getElementById("runButton");
const statusPanel = document.getElementById("statusPanel");
const statusText = document.getElementById("statusText");
const errorPanel = document.getElementById("errorPanel");
const errorText = document.getElementById("errorText");
const resultsPanel = document.getElementById("resultsPanel");

assetPicker.addEventListener("click", (e) => {
  const btn = e.target.closest(".asset-btn");
  if (!btn) return;
  document.querySelectorAll(".asset-btn").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  state.asset = btn.dataset.asset;
  runButton.disabled = false;
  runButton.textContent = `fit live model on ${state.asset}`;
});

runButton.addEventListener("click", () => {
  if (!state.asset) return;
  runLiveFit(state.asset);
});

async function runLiveFit(asset) {
  runButton.disabled = true;
  errorPanel.classList.add("hidden");
  resultsPanel.classList.add("hidden");
  statusPanel.classList.remove("hidden");

  let msgIndex = 0;
  statusText.textContent = statusMessages[0];
  const rotator = setInterval(() => {
    msgIndex = Math.min(msgIndex + 1, statusMessages.length - 1);
    statusText.textContent = statusMessages[msgIndex];
  }, 3500);

  try {
    const resp = await fetch(`${API_BASE}/api/train/${asset}`, { method: "POST" });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `request failed with status ${resp.status}`);
    }
    const data = await resp.json();
    clearInterval(rotator);
    statusPanel.classList.add("hidden");
    renderResults(data);
  } catch (err) {
    clearInterval(rotator);
    statusPanel.classList.add("hidden");
    errorText.textContent = err.message;
    errorPanel.classList.remove("hidden");
  } finally {
    runButton.disabled = false;
  }
}

function renderResults(data) {
  document.getElementById("metricRmse").textContent = data.metrics.rmse.toFixed(4);
  document.getElementById("metricMae").textContent = data.metrics.mae.toFixed(4);
  document.getElementById("metricDir").textContent = (data.metrics.directional_accuracy * 100).toFixed(1) + "%";
  document.getElementById("metricDof").textContent = data.dof.toFixed(1);
  document.getElementById("metricWindow").textContent = `${data.window_train}d / ${data.horizon_test}d`;
  document.getElementById("caveatText").textContent = data.caveat;

  renderChart(data.predictions);
  renderPredictionsTable(data.predictions);
  renderCalibrationTable(data.calibration);

  resultsPanel.classList.remove("hidden");
}

function renderChart(predictions) {
  const ctx = document.getElementById("decompChart").getContext("2d");
  const labels = predictions.map((p) => p.date);
  const fundamental = predictions.map((p) => p.var_fundamental);
  const technical = predictions.map((p) => p.var_technical);
  const sentiment = predictions.map((p) => p.var_sentiment);
  const interaction = predictions.map((p) => p.var_interaction);

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (state.chart) {
    state.chart.destroy();
  }

  state.chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "fundamental",
          data: fundamental,
          borderColor: "#4C9F70",
          backgroundColor: "rgba(76,159,112,0.55)",
          fill: true,
          stack: "decomp",
          tension: 0.25,
          pointRadius: 0,
        },
        {
          label: "technical",
          data: technical,
          borderColor: "#4C86C8",
          backgroundColor: "rgba(76,134,200,0.55)",
          fill: true,
          stack: "decomp",
          tension: 0.25,
          pointRadius: 0,
        },
        {
          label: "sentiment",
          data: sentiment,
          borderColor: "#D97B4F",
          backgroundColor: "rgba(217,123,79,0.55)",
          fill: true,
          stack: "decomp",
          tension: 0.25,
          pointRadius: 0,
        },
        {
          label: "interaction",
          data: interaction,
          borderColor: "#9B8FC9",
          backgroundColor: "transparent",
          borderDash: [5, 4],
          fill: false,
          tension: 0.25,
          pointRadius: 0,
        },
      ],
    },
    options: {
      animation: prefersReducedMotion ? false : { duration: 900, easing: "easeOutQuart" },
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#8A96AC", font: { family: "IBM Plex Mono", size: 10 } }, grid: { color: "#1B2740" } },
        y: {
          stacked: true,
          ticks: { color: "#8A96AC", font: { family: "IBM Plex Mono", size: 10 } },
          grid: { color: "#1B2740" },
        },
      },
    },
  });
}

function renderPredictionsTable(predictions) {
  const tbody = document.querySelector("#predictionsTable tbody");
  tbody.innerHTML = "";
  predictions.forEach((p) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${p.date}</td>
      <td>${p.y_true.toFixed(4)}</td>
      <td>${p.pred_mean.toFixed(4)}</td>
      <td>[${p.pred_lower_95.toFixed(4)}, ${p.pred_upper_95.toFixed(4)}]</td>
      <td class="badge-cell"><span class="badge badge-${p.dominant_modality}">${p.dominant_modality}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function renderCalibrationTable(calibration) {
  const tbody = document.querySelector("#calibrationTable tbody");
  tbody.innerHTML = "";
  calibration.forEach((row) => {
    const diff = row.coverage - row.level;
    const flag = Math.abs(diff) < 0.1 ? "well-calibrated" : diff > 0 ? "conservative" : "overconfident";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${(row.level * 100).toFixed(0)}%</td>
      <td>${(row.coverage * 100).toFixed(1)}%</td>
      <td style="color: var(--text-muted); font-family: var(--font-body); font-size: 12.5px;">${flag}</td>
    `;
    tbody.appendChild(tr);
  });
}

(async function init() {
  try {
    const resp = await fetch(`${API_BASE}/api/assets`);
    if (!resp.ok) return;
    await resp.json();
  } catch (err) {
    // backend not reachable yet; asset buttons still work once it is
  }
})();
