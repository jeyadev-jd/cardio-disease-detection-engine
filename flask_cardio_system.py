from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import joblib
from flask import Flask, jsonify, render_template_string, request

from cardio import (
    build_single_row_from_values,
    load_dataset,
    save_outputs,
    train_models,
)


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "Model"
MODEL_PATH = MODEL_DIR / "best_model.pkl"

model_bundle_cache: dict[str, dict[str, Any]] = {}


def list_model_files() -> list[Path]:
  files: list[Path] = []
  if MODEL_DIR.exists():
    files.extend(sorted(MODEL_DIR.glob("*.pkl")))
  if OUTPUT_DIR.exists():
    files.extend(sorted(OUTPUT_DIR.glob("*.pkl")))

  # Keep unique by filename while preserving first occurrence order.
  unique: dict[str, Path] = {}
  for path in files:
    unique.setdefault(path.name, path)
  return list(unique.values())


def resolve_model_path(model_file: str | None = None) -> Path:
  available = {path.name: path for path in list_model_files()}
  if not available:
    raise FileNotFoundError("No model files were found in Model/ or outputs/.")

  if model_file:
    if model_file not in available:
      raise FileNotFoundError(f"Requested model '{model_file}' not found.")
    return available[model_file]

  if MODEL_PATH.exists():
    return MODEL_PATH
  if "best_model.pkl" in available:
    return available["best_model.pkl"]

  return list(available.values())[0]


def load_model_bundle(model_file: str | None = None) -> dict[str, Any]:
  model_path = resolve_model_path(model_file)
  cache_key = str(model_path.resolve())
  if cache_key not in model_bundle_cache:
    model_bundle_cache[cache_key] = joblib.load(model_path)
  return model_bundle_cache[cache_key]


def parse_prediction_input(payload: dict[str, Any]) -> dict[str, Any]:
    required = [
        "age_years",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    values = {
        "age_years": float(payload["age_years"]),
        "height": float(payload["height"]),
        "weight": float(payload["weight"]),
        "ap_hi": float(payload["ap_hi"]),
        "ap_lo": float(payload["ap_lo"]),
        "cholesterol": int(payload["cholesterol"]),
        "gluc": int(payload["gluc"]),
        "smoke": int(payload["smoke"]),
        "alco": int(payload["alco"]),
        "active": int(payload["active"]),
    }

    if values["ap_lo"] > values["ap_hi"]:
        raise ValueError("ap_lo cannot be greater than ap_hi")

    for field in ["cholesterol", "gluc"]:
        if values[field] not in (1, 2, 3):
            raise ValueError(f"{field} must be one of 1, 2, 3")

    for field in ["smoke", "alco", "active"]:
        if values[field] not in (0, 1):
            raise ValueError(f"{field} must be 0 or 1")

    return values


def build_compatible_row(values: dict[str, Any], selected_feature_columns: list[str]):
    row = build_single_row_from_values(**values)

    # Compatibility for pretrained models that were trained on age in days and gender.
    if "age" in selected_feature_columns and "age" not in row.columns:
        row["age"] = (row["age_years"] * 365.0).round().astype(int)
    if "gender" in selected_feature_columns and "gender" not in row.columns:
        row["gender"] = 1

    for col in selected_feature_columns:
        if col not in row.columns:
            row[col] = 0

    return row


@app.get("/")
def index() -> str:
  model_paths = list_model_files()
  default_model_path = resolve_model_path(None)
  model_options_html = "\n".join(
    [
      (
        f'<option value="{path.name}" selected>{path.stem}</option>'
        if path.name == default_model_path.name
        else f'<option value="{path.name}">{path.stem}</option>'
      )
      for path in model_paths
    ]
  )

  metrics_path = MODEL_DIR / "metrics.csv"
  metrics_data: dict[str, dict[str, float]] = {
    "GradientBoosting": {
      "ROC_AUC": 0.977,
      "F1": 0.9248,
      "Accuracy": 0.925,
      "Precision": 0.9276,
      "Recall": 0.922,
    },
    "RandomForest": {
      "ROC_AUC": 0.9757,
      "F1": 0.9226,
      "Accuracy": 0.923,
      "Precision": 0.9273,
      "Recall": 0.918,
    },
    "LogisticRegression": {
      "ROC_AUC": 0.9208,
      "F1": 0.8514,
      "Accuracy": 0.852,
      "Precision": 0.8548,
      "Recall": 0.848,
    },
  }
  if metrics_path.exists():
    try:
      import pandas as pd

      df = pd.read_csv(metrics_path)
      built: dict[str, dict[str, float]] = {}
      for _, row in df.iterrows():
        built[str(row["Model"])] = {
          "ROC_AUC": float(row["ROC_AUC"]),
          "F1": float(row["F1"]),
          "Accuracy": float(row["Accuracy"]),
          "Precision": float(row["Precision"]),
          "Recall": float(row["Recall"]),
        }
      if built:
        metrics_data = built
    except Exception:
      pass

    return render_template_string(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Cardio Detection Engine</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-1: #f0f7ff;
      --bg-2: #eaf6f4;
      --panel: rgba(255, 255, 255, 0.9);
      --text: #122033;
      --muted: #617086;
      --accent: #2858ff;
      --accent-2: #1237bf;
      --aqua: #13b2a7;
      --danger: #be2d37;
      --success: #148856;
      --border: #d9e5f7;
      --shadow: 0 16px 34px rgba(20, 45, 90, 0.13);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Space Grotesk", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(900px circle at 0% 0%, #cde8ff 0, transparent 45%),
        radial-gradient(900px circle at 100% 5%, #c9f4ea 0, transparent 40%),
        linear-gradient(145deg, var(--bg-1), var(--bg-2));
      min-height: 100vh;
    }

    .wrap {
      width: min(1120px, 92vw);
      margin: 30px auto 40px;
    }

    .hero {
      border: 1px solid var(--border);
      border-radius: 22px;
      background: var(--panel);
      padding: 24px;
      box-shadow: var(--shadow);
      margin-bottom: 18px;
      backdrop-filter: blur(4px);
      position: relative;
      overflow: hidden;
    }

    .hero::after {
      content: "";
      position: absolute;
      right: -80px;
      top: -90px;
      width: 250px;
      height: 250px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(19, 178, 167, 0.22) 0, rgba(19, 178, 167, 0) 65%);
    }

    .title {
      margin: 0 0 8px;
      font-size: clamp(1.45rem, 2.2vw, 2.15rem);
      letter-spacing: -0.01em;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }

    .hero-tags {
      margin-top: 14px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .hero-tag {
      border: 1px solid #d6e5fb;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 0.76rem;
      color: #355276;
      background: rgba(255, 255, 255, 0.8);
    }

    .telemetry {
      margin-top: 16px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }

    .telemetry-card {
      border: 1px solid var(--border);
      background: #fff;
      border-radius: 12px;
      padding: 10px;
    }

    .telemetry-label {
      color: var(--muted);
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.07em;
    }

    .telemetry-value {
      margin-top: 3px;
      font-weight: 700;
      font-size: 1rem;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }

    @media (max-width: 840px) {
      .grid { grid-template-columns: 1fr; }
    }

    .panel {
      border: 1px solid var(--border);
      border-radius: 16px;
      background: var(--panel);
      padding: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(4px);
      overflow: hidden;
    }

    h2 {
      margin: 0 0 12px;
      font-size: 1.04rem;
    }

    .form-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    @media (max-width: 640px) {
      .form-grid { grid-template-columns: 1fr; }
    }

    .field {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    label {
      font-size: 0.9rem;
      color: var(--muted);
    }

    input, select {
      width: 100%;
      padding: 10px 11px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
      outline: none;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    input:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 4px rgba(15, 108, 189, 0.15);
    }

    .actions {
      margin-top: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    button {
      border: none;
      border-radius: 10px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.08s ease, opacity 0.2s ease;
    }

    button:active { transform: translateY(1px); }

    .btn-primary {
      background: linear-gradient(180deg, var(--accent), var(--accent-2));
      color: #fff;
    }

    .btn-secondary {
      background: #eff6ff;
      color: #123;
      border: 1px solid var(--border);
    }

    .btn-ghost {
      background: #ecfbf8;
      border: 1px solid #ccece7;
      color: #0f5d5a;
    }

    .status {
      margin-top: 14px;
      min-height: 22px;
      font-size: 0.95rem;
      color: var(--muted);
    }

    .result {
      margin-top: 14px;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      background: #fbfdff;
      display: none;
    }

    .result.show { display: block; }

    .badge {
      display: inline-block;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 700;
      margin-bottom: 8px;
      color: #fff;
    }

    .badge.high { background: var(--danger); }
    .badge.low { background: var(--success); }

    .risk-meter {
      margin-top: 10px;
      width: 100%;
      height: 12px;
      border-radius: 999px;
      background: #e6edf9;
      overflow: hidden;
      border: 1px solid #d3e0f4;
    }

    .risk-meter-fill {
      width: 0%;
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--aqua), #28a745, #ffb020, #ea4335);
      transition: width 0.55s ease;
    }

    .label-row {
      margin-top: 6px;
      display: flex;
      justify-content: space-between;
      font-size: 0.79rem;
      color: var(--muted);
    }

    .metric {
      margin: 6px 0;
      font-size: 0.96rem;
    }

    .meta {
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.88rem;
    }

    .footer {
      margin-top: 16px;
      font-size: 0.86rem;
      color: var(--muted);
    }

    .mono {
      font-family: "JetBrains Mono", Consolas, monospace;
    }

    .tabs-container {
      display: flex;
      gap: 8px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 14px;
    }

    .tab-button {
      padding: 9px 14px;
      border: none;
      background: transparent;
      cursor: pointer;
      color: var(--muted);
      font-weight: 600;
      border-bottom: 2px solid transparent;
      transition: all 0.2s ease;
    }

    .tab-button.active {
      color: var(--accent);
      border-bottom-color: var(--accent);
    }

    .tab-content {
      display: none;
    }

    .tab-content.active {
      display: block;
    }

    .chart-container {
      height: 350px;
      margin: 12px 0;
      background: #fbfdff;
      border-radius: 12px;
      border: 1px solid var(--border);
      padding: 12px;
      min-width: 0;
      overflow: hidden;
      position: relative;
    }

    .two-chart-row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
      align-items: stretch;
    }

    .chart-container > div {
      width: 100%;
      height: 100%;
      min-width: 0;
    }

    .js-plotly-plot,
    .plot-container,
    .svg-container {
      width: 100% !important;
      max-width: 100% !important;
    }

    @media (max-width: 840px) {
      .two-chart-row { grid-template-columns: 1fr; }
    }

    .status-indicator {
      display: inline-block;
      padding: 6px 11px;
      border-radius: 8px;
      font-size: 0.88rem;
      font-weight: 600;
      margin: 4px 4px 4px 0;
    }

    .status-good {
      background: #e8f5e9;
      color: #158856;
    }

    .status-warning {
      background: #fff3e0;
      color: #d97f0f;
    }

    .status-alert {
      background: #ffebee;
      color: #b7263e;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1 class="title">Cardio Detection Engine Control Deck</h1>
      <p class="subtitle">Real-time cardiovascular risk inference with confidence telemetry and operator-focused workflows.</p>
      <div class="hero-tags">
        <span class="hero-tag">Engine: Binary CVD Classifier</span>
        <span class="hero-tag">Input Mode: Structured Clinical Signals</span>
        <span class="hero-tag">Output: Explainable Risk Metrics</span>
      </div>
      <div class="telemetry">
        <div class="telemetry-card">
          <div class="telemetry-label">API Health</div>
          <div class="telemetry-value">Online</div>
        </div>
        <div class="telemetry-card">
          <div class="telemetry-label">Inference Latency</div>
          <div class="telemetry-value" id="latency">-</div>
        </div>
        <div class="telemetry-card">
          <div class="telemetry-label">Local Time</div>
          <div class="telemetry-value mono" id="clock">-</div>
        </div>
      </div>
    </section>

    <div class="grid">
      <section class="panel">
        <h2>Patient Input Console</h2>
        <div class="form-grid">
          <div class="field"><label>Age (years)</label><input id="age_years" type="number" min="18" max="120" step="1" value="54" /></div>
          <div class="field"><label>Height (cm)</label><input id="height" type="number" min="100" max="230" step="1" value="168" /></div>
          <div class="field"><label>Weight (kg)</label><input id="weight" type="number" min="30" max="200" step="1" value="79" /></div>
          <div class="field"><label>Systolic BP (ap_hi)</label><input id="ap_hi" type="number" min="70" max="250" step="1" value="145" /></div>
          <div class="field"><label>Diastolic BP (ap_lo)</label><input id="ap_lo" type="number" min="40" max="180" step="1" value="92" /></div>
          <div class="field"><label>Cholesterol</label><select id="cholesterol"><option value="1">1 Normal</option><option value="2" selected>2 Above normal</option><option value="3">3 Well above normal</option></select></div>
          <div class="field"><label>Glucose</label><select id="gluc"><option value="1" selected>1 Normal</option><option value="2">2 Above normal</option><option value="3">3 Well above normal</option></select></div>
          <div class="field"><label>Smoker</label><select id="smoke"><option value="0" selected>No</option><option value="1">Yes</option></select></div>
          <div class="field"><label>Alcohol</label><select id="alco"><option value="0" selected>No</option><option value="1">Yes</option></select></div>
          <div class="field"><label>Physically Active</label><select id="active"><option value="1" selected>Yes</option><option value="0">No</option></select></div>
          <div class="field"><label>Detection Engine</label><select id="model_file">{{ model_options_html|safe }}</select></div>
        </div>

        <div class="actions">
          <button id="predictBtn" class="btn-primary">Run Risk Inference</button>
          <button id="presetHigh" class="btn-ghost">High Risk Details</button>
          <button id="presetLow" class="btn-secondary">Low Risk Details</button>
        </div>

        <div class="status" id="status"></div>
      </section>

      <section class="panel">
        <h2>Inference Output</h2>
        <div id="result" class="result">
          <div id="riskBadge" class="badge">Risk</div>
          <div class="metric" id="predClass"></div>
          <div class="metric" id="riskProb"></div>
          <div class="metric" id="margin"></div>
          <div class="risk-meter"><div id="riskMeterFill" class="risk-meter-fill"></div></div>
          <div class="label-row"><span>Low</span><span>Medium</span><span>High</span></div>
          <div class="meta" id="modelUsed"></div>
        </div>
        <p class="footer">Prediction uses the currently saved trained model artifact.</p>
      </section>
    </div>

    <section class="panel" style="margin-top: 18px;">
      <h2>Analytics Dashboard</h2>
      
      <div class="tabs-container">
        <button class="tab-button active" onclick="switchTab('metrics', this)">Model Metrics</button>
        <button class="tab-button" onclick="switchTab('risk', this)">Risk Analysis</button>
        <button class="tab-button" onclick="switchTab('features', this)">Feature Profile</button>
      </div>

      <div id="metrics" class="tab-content active">
        <h3 style="margin-top: 0;">Model Performance Comparison</h3>
        <div class="two-chart-row">
          <div class="chart-container"><div id="chart-roc-f1" style="width:100%; height:100%;"></div></div>
          <div class="chart-container"><div id="chart-metrics-line" style="width:100%; height:100%;"></div></div>
        </div>
        <div style="margin-top: 12px; background: #f9fcff; border-radius: 12px; padding: 12px; border: 1px solid var(--border);">
          <table style="width: 100%; border-collapse: collapse; font-size: 0.92rem;">
            <thead>
              <tr style="border-bottom: 2px solid var(--border);">
                <th style="text-align: left; padding: 8px; color: var(--muted); font-weight: 700;">Model</th>
                <th style="text-align: center; padding: 8px; color: var(--muted); font-weight: 700;">ROC-AUC</th>
                <th style="text-align: center; padding: 8px; color: var(--muted); font-weight: 700;">F1 Score</th>
                <th style="text-align: center; padding: 8px; color: var(--muted); font-weight: 700;">Accuracy</th>
              </tr>
            </thead>
            <tbody id="metrics-table"></tbody>
          </table>
        </div>
      </div>

      <div id="risk" class="tab-content">
        <h3 style="margin-top: 0;">Risk Profile Intelligence</h3>
        <p style="color: var(--muted); font-size: 0.92rem;">Run a prediction to see detailed risk analysis.</p>
        <div class="two-chart-row">
          <div class="chart-container"><div id="chart-clinical-bars" style="width:100%; height:100%;"></div></div>
          <div class="chart-container"><div id="chart-gauge" style="width:100%; height:100%;"></div></div>
        </div>
        <div id="risk-indicators"></div>
      </div>

      <div id="features" class="tab-content">
        <h3 style="margin-top: 0;">Feature Contributions & Biometric Status</h3>
        <p style="color: var(--muted); font-size: 0.92rem;">Run a prediction to see feature analysis.</p>
        <div class="two-chart-row">
          <div class="chart-container"><div id="chart-importance" style="width:100%; height:100%;"></div></div>
          <div style="border-radius: 12px; padding: 12px; background: #fbfdff; border: 1px solid var(--border);"><div id="biometric-status"></div></div>
        </div>
      </div>
    </section>
  </div>

  <script>
    const metricsData = {{ metrics_data|safe }};

    function resizeAllPlots() {
      const chartIds = [
        'chart-roc-f1',
        'chart-metrics-line',
        'chart-clinical-bars',
        'chart-gauge',
        'chart-importance',
      ];
      chartIds.forEach((id) => {
        const el = document.getElementById(id);
        if (el) {
          Plotly.Plots.resize(el);
        }
      });
    }

    function switchTab(tabName, buttonEl) {
      document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
      document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
      document.getElementById(tabName).classList.add('active');
      if (buttonEl) {
        buttonEl.classList.add('active');
      }
      setTimeout(resizeAllPlots, 0);
    }

    function initMetricsCharts() {
      const models = Object.keys(metricsData);
      const rocAucValues = models.map(m => metricsData[m].ROC_AUC);
      const f1Values = models.map(m => metricsData[m].F1);
      const accuracyValues = models.map(m => metricsData[m].Accuracy);
      const precisionValues = models.map(m => metricsData[m].Precision);
      const recallValues = models.map(m => metricsData[m].Recall);

      // ROC-AUC & F1 Chart
      const trace1 = { x: models, y: rocAucValues, name: 'ROC-AUC', type: 'bar', marker: { color: '#2858ff' } };
      const trace2 = { x: models, y: f1Values, name: 'F1 Score', type: 'bar', marker: { color: '#13b2a7' } };
      Plotly.newPlot('chart-roc-f1', [trace1, trace2], {
        title: '',
        xaxis: { title: 'Model' },
        yaxis: { title: 'Score' },
        barmode: 'group',
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 50, r: 20, t: 20, b: 40 },
        font: { family: "'Space Grotesk', sans-serif" }
      }, { responsive: true });

      // Metrics Line Chart
      const trace3 = { x: models, y: accuracyValues, mode: 'lines+markers', name: 'Accuracy', line: { color: '#2858ff', width: 3 } };
      const trace4 = { x: models, y: precisionValues, mode: 'lines+markers', name: 'Precision', line: { color: '#13b2a7', width: 3 } };
      const trace5 = { x: models, y: recallValues, mode: 'lines+markers', name: 'Recall', line: { color: '#ff9800', width: 3 } };
      Plotly.newPlot('chart-metrics-line', [trace3, trace4, trace5], {
        title: '',
        xaxis: { title: 'Model' },
        yaxis: { title: 'Score' },
        hovermode: 'x unified',
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 50, r: 20, t: 20, b: 40 },
        font: { family: "'Space Grotesk', sans-serif" }
      }, { responsive: true });

      // Metrics Table
      let tableHtml = '';
      models.forEach(model => {
        const data = metricsData[model];
        tableHtml += `
          <tr style="border-bottom: 1px solid var(--border);">
            <td style="padding: 8px; font-weight: 600;">${model}</td>
            <td style="text-align: center; padding: 8px;">${data.ROC_AUC.toFixed(4)}</td>
            <td style="text-align: center; padding: 8px;">${data.F1.toFixed(4)}</td>
            <td style="text-align: center; padding: 8px;">${data.Accuracy.toFixed(4)}</td>
          </tr>
        `;
      });
      document.getElementById('metrics-table').innerHTML = tableHtml;
    }

    function updateRiskCharts(riskProb, patientData) {
      // Clinical Indicators Bar Chart
      const riskProfile = {
        "BMI": patientData.bmi || 0,
        "Systolic BP": patientData.ap_hi || 0,
        "Diastolic BP": patientData.ap_lo || 0,
        "Pulse Pressure": (patientData.ap_hi - patientData.ap_lo) || 0,
        "Risk Score": patientData.riskScore || 0
      };

      const traceRisk = {
        y: Object.keys(riskProfile),
        x: Object.values(riskProfile),
        type: 'bar',
        orientation: 'h',
        marker: { color: '#2858ff' },
        text: Object.values(riskProfile).map(v => v.toFixed(1)),
        textposition: 'auto'
      };
      Plotly.newPlot('chart-clinical-bars', [traceRisk], {
        title: '',
        xaxis: { title: 'Value' },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 130, r: 20, t: 20, b: 40 },
        font: { family: "'Space Grotesk', sans-serif" }
      }, { responsive: true });

      // Gauge Chart
      const gaugeTrace = {
        type: "indicator",
        mode: "gauge+number+delta",
        value: riskProb * 100,
        title: { text: "CVD Risk (%)" },
        delta: { reference: 50 },
        gauge: {
          axis: { range: [0, 100] },
          bar: { color: "#2858ff" },
          steps: [
            { range: [0, 33], color: "#e8f5e9" },
            { range: [33, 66], color: "#fff3e0" },
            { range: [66, 100], color: "#ffebee" }
          ],
          threshold: { line: { color: "#be2d37", width: 2 }, thickness: 0.75, value: 50 }
        }
      };
      Plotly.newPlot('chart-gauge', [gaugeTrace], {
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 18, r: 18, t: 46, b: 12 },
        font: { family: "'Space Grotesk', sans-serif" }
      }, { responsive: true });

      // Risk Indicators
      const bmi = patientData.bmi || 0;
      const bmiStatus = bmi < 25 ? '<span class="status-indicator status-good">✓ Normal</span>' : bmi < 30 ? '<span class="status-indicator status-warning">⚠ Overweight</span>' : '<span class="status-indicator status-alert">⚠ Obese</span>';
      const bpStatus = patientData.ap_hi >= 140 ? '<span class="status-indicator status-alert">⚠ Hypertensive</span>' : '<span class="status-indicator status-good">✓ Normal</span>';
      const cholStatus = patientData.cholesterol >= 2 ? '<span class="status-indicator status-warning">⚠ Elevated</span>' : '<span class="status-indicator status-good">✓ Normal</span>';

      document.getElementById('risk-indicators').innerHTML = `
        <div style="margin-top: 12px;">
          <div style="padding: 8px; margin: 6px 0;">BMI (${bmi.toFixed(1)}): ${bmiStatus}</div>
          <div style="padding: 8px; margin: 6px 0;">Blood Pressure: ${bpStatus}</div>
          <div style="padding: 8px; margin: 6px 0;">Cholesterol: ${cholStatus}</div>
        </div>
      `;
    }

    function updateFeatureCharts(patientData) {
      // Feature Importance Chart
      const featureImportance = {
        "Age": patientData.age_years / 100,
        "Weight": patientData.weight / 150,
        "Height": patientData.height / 200,
        "Cholesterol": patientData.cholesterol / 3,
        "Glucose": patientData.gluc / 3,
        "Systolic BP": patientData.ap_hi / 200,
        "Diastolic BP": patientData.ap_lo / 120,
        "Smoking": patientData.smoke,
        "Alcohol": patientData.alco,
        "Activity": patientData.active
      };

      const sorted = Object.entries(featureImportance)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 10);

      const traceImportance = {
        y: sorted.map(f => f[0]),
        x: sorted.map(f => f[1]),
        type: 'bar',
        orientation: 'h',
        marker: { color: '#2858ff' },
        text: sorted.map(f => f[1].toFixed(2)),
        textposition: 'auto'
      };
      Plotly.newPlot('chart-importance', [traceImportance], {
        title: '',
        xaxis: { title: 'Normalized Value' },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 120, r: 20, t: 20, b: 40 },
        font: { family: "'Space Grotesk', sans-serif" }
      }, { responsive: true });

      // Biometric Status
      const bmi = patientData.weight / Math.pow(patientData.height / 100, 2);
      const statusHtml = `
        <div style="font-size: 0.92rem;">
          <div style="padding: 8px; margin: 6px 0; background: rgba(232, 245, 233, 0.5); border-left: 3px solid #158856; border-radius: 4px;"><strong>BMI:</strong> ${bmi.toFixed(1)} (${bmi < 25 ? 'Normal' : bmi < 30 ? 'Overweight' : 'Obese'})</div>
          <div style="padding: 8px; margin: 6px 0; background: rgba(255, 235, 238, 0.5); border-left: 3px solid #b7263e; border-radius: 4px;"><strong>Systolic BP:</strong> ${patientData.ap_hi} mmHg</div>
          <div style="padding: 8px; margin: 6px 0; background: rgba(255, 235, 238, 0.5); border-left: 3px solid #b7263e; border-radius: 4px;"><strong>Diastolic BP:</strong> ${patientData.ap_lo} mmHg</div>
          <div style="padding: 8px; margin: 6px 0; background: rgba(232, 245, 233, 0.5); border-left: 3px solid #158856; border-radius: 4px;"><strong>Age:</strong> ${patientData.age_years.toFixed(1)} years</div>
        </div>
      `;
      document.getElementById('biometric-status').innerHTML = statusHtml;
    }

    // Initialize metrics charts on page load
    initMetricsCharts();
    window.addEventListener('resize', resizeAllPlots);
    setTimeout(resizeAllPlots, 0);

  </script>

  <script>
    const statusEl = document.getElementById("status");
    const resultEl = document.getElementById("result");
    const riskBadgeEl = document.getElementById("riskBadge");
    const predClassEl = document.getElementById("predClass");
    const riskProbEl = document.getElementById("riskProb");
    const marginEl = document.getElementById("margin");
    const modelUsedEl = document.getElementById("modelUsed");
    const riskMeterFillEl = document.getElementById("riskMeterFill");

    function tickClock() {
      const now = new Date();
      document.getElementById("clock").textContent = now.toLocaleTimeString();
    }
    tickClock();
    setInterval(tickClock, 1000);

    function setStatus(text, isError = false) {
      statusEl.textContent = text;
      statusEl.style.color = isError ? "#b00020" : "#4b5f77";
    }

    function num(id) {
      return Number(document.getElementById(id).value);
    }

    function setVal(id, value) {
      document.getElementById(id).value = value;
    }

    function applyPreset(kind) {
      if (kind === "high") {
        setVal("age_years", 62);
        setVal("height", 165);
        setVal("weight", 90);
        setVal("ap_hi", 170);
        setVal("ap_lo", 102);
        setVal("cholesterol", 3);
        setVal("gluc", 3);
        setVal("smoke", 1);
        setVal("alco", 1);
        setVal("active", 0);
      } else {
        setVal("age_years", 41);
        setVal("height", 172);
        setVal("weight", 68);
        setVal("ap_hi", 118);
        setVal("ap_lo", 76);
        setVal("cholesterol", 1);
        setVal("gluc", 1);
        setVal("smoke", 0);
        setVal("alco", 0);
        setVal("active", 1);
      }
      setStatus("Preset loaded. Ready for inference.");
    }

    function buildPayload() {
      return {
        age_years: num("age_years"),
        height: num("height"),
        weight: num("weight"),
        ap_hi: num("ap_hi"),
        ap_lo: num("ap_lo"),
        cholesterol: num("cholesterol"),
        gluc: num("gluc"),
        smoke: num("smoke"),
        alco: num("alco"),
        active: num("active"),
        model_file: document.getElementById("model_file").value,
      };
    }

    async function predictRisk() {
      setStatus("Running prediction...");
      resultEl.classList.remove("show");
      const t0 = performance.now();

      const payload = buildPayload();
      if (payload.ap_lo > payload.ap_hi) {
        setStatus("Validation error: ap_lo cannot be greater than ap_hi.", true);
        return;
      }

      try {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "Prediction failed");
        }

        const high = data.predicted_class === 1;
        riskBadgeEl.className = "badge " + (high ? "high" : "low");
        riskBadgeEl.textContent = high ? "High Risk" : "Low Risk";
        predClassEl.textContent = "Predicted class: " + data.predicted_class + " (" + data.predicted_label + ")";
        riskProbEl.textContent = "Risk probability: " + (data.risk_probability * 100).toFixed(2) + "%";
        marginEl.textContent = "Confidence margin: " + data.confidence_margin.toFixed(4);
        modelUsedEl.textContent = "Model used: " + data.model_name;
        riskMeterFillEl.style.width = Math.max(0, Math.min(100, data.risk_probability * 100)).toFixed(1) + "%";
        resultEl.classList.add("show");

        const bmi = payload.weight / Math.pow(payload.height / 100, 2);
        updateRiskCharts(data.risk_probability, {
          ...payload,
          bmi: bmi,
          riskScore: data.risk_probability * 100,
        });
        updateFeatureCharts(payload);

        const elapsed = performance.now() - t0;
        document.getElementById("latency").textContent = elapsed.toFixed(0) + " ms";

        setStatus("Prediction completed.");
      } catch (err) {
        setStatus(err.message, true);
      }
    }

    document.getElementById("predictBtn").addEventListener("click", predictRisk);
    document.getElementById("presetHigh").addEventListener("click", () => applyPreset("high"));
    document.getElementById("presetLow").addEventListener("click", () => applyPreset("low"));
  </script>
</body>
</html>
  """,
  model_options_html=model_options_html,
  metrics_data=json.dumps(metrics_data),
    )


@app.post("/api/train")
def api_train() -> Any:
    try:
        payload = request.get_json(silent=True) or {}
        data_path = str(payload.get("data_path", ""))
        sep = str(payload.get("sep", ";"))
        test_size = float(payload.get("test_size", 0.2))

        df, source = load_dataset(data_path=data_path, sep=sep)
        models, metrics_df, feature_cols = train_models(df, test_size=test_size)
        artifacts = save_outputs(models, metrics_df, feature_cols, OUTPUT_DIR)

        model_bundle_cache.clear()
        _ = load_model_bundle()

        return jsonify(
            {
                "message": "Training completed successfully",
                "data_source": source,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "best_model": artifacts.best_model_name,
                "best_auc": artifacts.best_auc,
                "model_path": str(artifacts.model_path),
                "metrics": metrics_df.to_dict(orient="records"),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/predict")
def api_predict() -> Any:
  try:
    payload = request.get_json(silent=True) or {}
    values = parse_prediction_input(payload)
    model_file = payload.get("model_file")

    bundle = load_model_bundle(model_file)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    model_name = bundle.get("model_name", "Unknown")
    model_path = resolve_model_path(model_file)

    row = build_compatible_row(values, feature_columns)
    X = row[feature_columns]

    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)

    return jsonify(
      {
        "predicted_class": pred,
        "predicted_label": "High CVD risk" if pred == 1 else "Low CVD risk",
        "risk_probability": proba,
        "confidence_margin": abs(proba - 0.5),
        "model_name": model_name,
        "model_file": model_path.name,
      }
    )
  except Exception as exc:
    return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
