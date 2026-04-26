const LOWER_BETTER = new Set(["mae", "mse", "rmse"]);

const RADAR_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
  '#bcbd22', '#17becf'
];

async function loadLeaderboard() {
  const root = document.getElementById("leaderboard-app");
  if (!root) return;

  const raw = await fetch("../assets/leaderboard.csv").then(r => r.text());
  const rows = parseCSV(raw);

  const seas = unique(rows, "sea");

  const pairs = uniquePairs(rows, "forecast_len", "step");
  pairs.sort((a, b) => a.forecast_len - b.forecast_len || a.step.localeCompare(b.step));

  const models = unique(rows, "model");
  const metrics = unique(rows, "metric");

  const state = {
    sea: seas[0],
    forecast_len: pairs[0].forecast_len,
    step: pairs[0].step,
  };

  root.innerHTML = renderLayout(seas, pairs);

  renderSummary(rows, seas, models, metrics);
  bind(state, rows, models, metrics);
  render(state, rows, models, metrics);
}

function renderLayout(seas, pairs) {
  return `
    <div id="summary"></div>
    <div class="lb-controls">
      <div class="lb-field">
        <label for="sea-select">Sea</label>
        <select id="sea-select">
          ${seas.map(s => `<option value="${s}">${s}</option>`).join("")}
        </select>
      </div>
      <div class="lb-field">
        <label for="settings-select">Forecast length / Step</label>
        <select id="settings-select">
          ${pairs.map(p => `<option value="${p.forecast_len}|${p.step}">${p.forecast_len}d / ${p.step}</option>`).join("")}
        </select>
      </div>
    </div>
    <div id="snippet"></div>
    <canvas id="radar"></canvas>
    <div id="tables"></div>
  `;
}

function bind(state, rows, models, metrics) {
  document.getElementById("sea-select").onchange = e => {
    state.sea = e.target.value;
    render(state, rows, models, metrics);
  };
  document.getElementById("settings-select").onchange = e => {
    const [forecast_len, step] = e.target.value.split("|");
    state.forecast_len = +forecast_len;
    state.step = step;
    render(state, rows, models, metrics);
  };
}

function render(state, rows, models, metrics) {
  const filtered = getFilteredRows(rows, state);
  renderSnippet(state);
  renderAllTables(filtered);
  renderRadar(filtered, state, models);
}

function getFilteredRows(rows, state) {
  return rows.filter(r =>
    r.sea === state.sea &&
    r.forecast_len === state.forecast_len &&
    r.step === state.step
  );
}

function renderSummary(rows, seas, models, metrics) {
  const summaryContainer = document.getElementById("summary");
  if (!summaryContainer) return;

  const summary = [];

  models.forEach(model => {
    seas.forEach(sea => {
      const modelRows = rows.filter(r => r.model === model && r.sea === sea);
      
      metrics.forEach(metric => {
        const metricRows = modelRows.filter(r => r.metric === metric);
        if (metricRows.length === 0) return;
        
        const avgValue = metricRows.reduce((sum, r) => sum + r.value, 0) / metricRows.length;
        
        summary.push({
          model,
          sea,
          metric,
          value: avgValue
        });
      });
    });
  });

  const bestBySeaMetric = {};
  metrics.forEach(metric => {
    seas.forEach(sea => {
      const key = `${sea}_${metric}`;
      const candidates = summary.filter(r => r.sea === sea && r.metric === metric);
      if (candidates.length === 0) return;
      
      bestBySeaMetric[key] = LOWER_BETTER.has(metric.toLowerCase())
        ? candidates.reduce((best, r) => r.value < best.value ? r : best)
        : candidates.reduce((best, r) => r.value > best.value ? r : best);
    });
  });

  summaryContainer.innerHTML = `
    <div class="summary-card">
      <h3>Overview (averaged across all forecast lengths & steps)</h3>
      <div class="summary-grid">
        ${seas.map(sea => {
          const seaSummary = summary.filter(r => r.sea === sea);
          const modelsInSea = unique(seaSummary, "model");

          const wins = {};
          modelsInSea.forEach(model => {
            wins[model] = 0;
            metrics.forEach(metric => {
              const key = `${sea}_${metric}`;
              if (bestBySeaMetric[key]?.model === model) wins[model]++;
            });
          });
          
          const topModel = Object.entries(wins).sort((a, b) => b[1] - a[1])[0];
          
          return `
            <div class="summary-sea">
              <div class="summary-sea-name">${sea}</div>
              <div class="summary-best">
                <span class="summary-best-label">Top model:</span>
                <span class="best">${topModel ? topModel[0] : '—'}</span>
                <span class="summary-best-wins">${topModel ? `(${topModel[1]}/${metrics.length} metrics)` : ''}</span>
              </div>
              <div class="summary-metrics">
                ${metrics.map(metric => {
                  const key = `${sea}_${metric}`;
                  const best = bestBySeaMetric[key];
                  const arrow = LOWER_BETTER.has(metric.toLowerCase()) ? '↓' : '↑';
                  return best ? `
                    <div class="summary-metric">
                      <span class="summary-metric-name">${metric} ${arrow}</span>
                      <span class="summary-metric-value">${best.value.toFixed(4)}</span>
                      <span class="summary-metric-model">${best.model}</span>
                    </div>
                  ` : '';
                }).join("")}
              </div>
            </div>
          `;
        }).join("")}
      </div>
    </div>
  `;
}

function renderAllTables(rows) {
  const container = document.getElementById("tables");
  const metrics = unique(rows, "metric");

  container.innerHTML = `
    <div class="metrics-grid">
      ${metrics.map(metric => {
        const filtered = rows.filter(r => r.metric === metric);
        const sorted = sort(filtered, metric);
        const arrow = LOWER_BETTER.has(metric.toLowerCase()) ? ' ↓' : ' ↑';

        const values = sorted.map(r => r.value);
        const minVal = Math.min(...values);
        const maxVal = Math.max(...values);
        const range = maxVal - minVal || 1;

        return `
          <div class="metric-card">
            <div class="metric-title">
              <h2>${metric}${arrow}</h2>
            </div>
            <table>
              <colgroup>
                <col style="width: 5%">
                <col style="width: 35%">
                <col style="width: 15%">
                <col style="width: 45%">
              </colgroup>
              <thead>
                <tr><th>#</th><th>Model</th><th>Value</th><th></th></tr>
              </thead>
              <tbody>
                ${sorted.map((r, i) => {
                  const ratio = LOWER_BETTER.has(metric.toLowerCase())
                    ? 1 - (r.value - minVal) / range
                    : (r.value - minVal) / range;
                  const pct = Math.max(0, Math.min(100, ratio * 100)).toFixed(1);
                  return `
                <tr>
                  <td class="${i === 0 ? "best" : ""}">${i + 1}</td>
                  <td class="${i === 0 ? "best" : ""}">${r.model}</td>
                  <td class="${i === 0 ? "best" : ""} val">${r.value.toFixed(5)}</td>
                  <td class="bar-cell">
                    <div class="bar-track">
                      <div class="bar-fill" style="width:${pct}%"></div>
                    </div>
                  </td>
                </tr>`;
                }).join("")}
              </tbody>
            </table>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function renderRadar(rows, state, models) {
  const canvas = document.getElementById("radar");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  if (window.radarChart) window.radarChart.destroy();
  if (rows.length === 0) return;

  const metrics = unique(rows, "metric");
  const normalized = normalize(rows);

  const datasets = models.map((model, i) => {
    const color = RADAR_COLORS[i % RADAR_COLORS.length];
    const data = metrics.map(metric => {
      const r = normalized.find(x =>
        x.model === model &&
        x.sea === state.sea &&
        x.forecast_len === state.forecast_len &&
        x.step === state.step &&
        x.metric === metric
      );
      return r ? r.norm : 0;
    });

    return {
      label: model,
      data,
      backgroundColor: color + '20',
      borderColor: color,
      borderWidth: 2,
      pointBackgroundColor: color,
    };
  });

  window.radarChart = new Chart(ctx, {
    type: "radar",
    data: { labels: metrics, datasets },
    options: {
      responsive: true,
      scales: {
        r: {
          min: 0,
          max: 1,
          ticks: { display: false },
          pointLabels: {
            font: { size: 14, family: Chart.defaults.font.family || 'Inter' }
          }
        }
      },
      plugins: {
        legend: {
          labels: {
            font: { size: 14, family: Chart.defaults.font.family || 'Inter' },
            usePointStyle: true,
            boxWidth: 8
          }
        }
      },
      elements: {
        point: {
          radius: 3,
          borderWidth: 1
        }
      }
    }
  });
}

function renderSnippet(state) {
  const code = `\
aiice = AIICE(
    sea="${state.sea}",
    forecast_len=${state.forecast_len},
    step="${state.step}",
    start="2020-01-01"
)`.trim();

  document.getElementById("snippet").innerHTML = `
    <div class="admonition example">
      <p class="admonition-title">AIICE experiment config</p>
      <div class="highlight">
        <pre><code class="language-python">${code}</code></pre>
      </div>
    </div>`;
}

function normalize(rows) {
  const values = rows.map(r => r.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min + 1e-9;

  return rows.map(r => {
    let v = (r.value - min) / range;
    if (LOWER_BETTER.has(r.metric.toLowerCase())) v = 1 - v;
    return { ...r, norm: v };
  });
}

function sort(data, metric) {
  return [...data].sort((a, b) =>
    LOWER_BETTER.has(metric.toLowerCase())
      ? a.value - b.value
      : b.value - a.value
  );
}

function unique(arr, key) {
  return [...new Set(arr.map(x => x[key]))];
}

function uniquePairs(arr, key1, key2) {
  const seen = new Set();
  const result = [];
  arr.forEach(x => {
    const k = `${x[key1]}|${x[key2]}`;
    if (!seen.has(k)) {
      seen.add(k);
      result.push({ [key1]: x[key1], [key2]: x[key2] });
    }
  });
  return result;
}

function parseCSV(csv) {
  return csv.trim().split("\n").slice(1).map(line => {
    const [model, sea, metric, value, forecast_len, step] = line.split(",");
    return {
      model,
      sea,
      metric,
      value: +value,
      forecast_len: +forecast_len,
      step
    };
  });
}

function loadChart() {
  const script = document.createElement("script");
  script.src = "https://cdn.jsdelivr.net/npm/chart.js";
  script.onload = () => {
    const font = getComputedStyle(document.documentElement)
      .getPropertyValue("--md-text-font")
      .trim() || "sans-serif";
    Chart.defaults.font.family = font;
    loadLeaderboard();
  };
  document.head.appendChild(script);
}

document.addEventListener("DOMContentLoaded", loadChart);
