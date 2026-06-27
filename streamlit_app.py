import json
import importlib
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

try:
    go = importlib.import_module("plotly.graph_objects")

    PLOTLY_AVAILABLE = True
except Exception:
    go = None
    PLOTLY_AVAILABLE = False

from cardio import build_single_row_from_values


st.set_page_config(page_title="Cardio Disease Predictor", page_icon="❤", layout="wide")

st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --ink: #0f1d2e;
    --muted: #5c6a79;
    --panel: #ffffff;
    --line: #d5e1f1;
    --aqua: #0ca7a0;
    --indigo: #2f66ff;
    --danger: #c73739;
    --safe: #118b57;
    --glass: rgba(255, 255, 255, 0.82);
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
}

.stApp {
    color: var(--ink);
    background:
        radial-gradient(1000px circle at 8% -10%, #c8f2eb 0, transparent 45%),
        radial-gradient(760px circle at 100% 10%, #dae4ff 0, transparent 45%),
        linear-gradient(145deg, #f3f8ff 0%, #eaf3ff 55%, #edf8f6 100%);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    right: 1rem;
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 0 rgba(47, 102, 255, 0.0); }
    50% { box-shadow: 0 0 0 8px rgba(47, 102, 255, 0.08); }
}

.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(247, 252, 255, 0.92));
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 20px 40px rgba(20, 40, 80, 0.12);
    margin-bottom: 16px;
}

.hero::after {
    content: '';
    position: absolute;
    inset: -40% -30% auto auto;
    width: 260px;
    height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(12, 167, 160, 0.20) 0, rgba(12, 167, 160, 0) 65%);
}

.hero-title {
    margin: 0;
    font-size: clamp(1.45rem, 2.3vw, 2.4rem);
    letter-spacing: -0.02em;
    color: #102340;
}

.hero-sub {
    margin: 8px 0 0;
    color: var(--muted);
    font-size: 0.98rem;
}

.hero-tags {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 14px;
}

.hero-tag {
    border: 1px solid #d8e4f5;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 0.77rem;
    color: #304865;
    background: rgba(255, 255, 255, 0.76);
}

.engine-panel {
    background: var(--glass);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 14px 16px;
    box-shadow: 0 12px 28px rgba(20, 40, 80, 0.09);
    backdrop-filter: blur(4px);
}

.metric-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 13px 14px;
    box-shadow: 0 8px 20px rgba(20, 40, 80, 0.08);
}

.metric-label {
    color: var(--muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

.metric-value {
    margin-top: 4px;
    font-size: 1.03rem;
    font-weight: 700;
}

.section-title {
    margin: 12px 0 6px;
    font-weight: 700;
    font-size: 1.04rem;
    color: #1b3358;
}

.result-panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 16px 30px rgba(20, 40, 80, 0.10);
    margin-bottom: 10px;
}

.status-pill {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 800;
    margin-bottom: 8px;
    color: #fff;
}

.status-high {
    background: var(--danger);
    animation: pulseGlow 1.6s ease-in-out infinite;
}

.status-low {
    background: var(--safe);
}

.mono {
    font-family: 'JetBrains Mono', Consolas, monospace;
}

div[data-testid="stTextInput"] > label,
div[data-testid="stSelectbox"] > label {
    font-weight: 700;
    color: #233653;
}

div[data-baseweb="input"] input,
div[data-baseweb="select"] {
    background: #ffffff;
    border-radius: 12px;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    font-weight: 800;
    color: #fff;
    background: linear-gradient(180deg, var(--indigo), #234ec7);
    padding: 0.72rem 1rem;
    box-shadow: 0 12px 20px rgba(47, 102, 255, 0.28);
}

.stButton > button:hover {
    filter: brightness(1.03);
}

.stDownloadButton > button {
    border-radius: 10px;
}

.mini-kpi {
    border: 1px solid #d8e4f5;
    border-radius: 12px;
    padding: 9px 11px;
    background: #f9fcff;
    margin-bottom: 8px;
}

.mini-kpi-label {
    font-size: 0.75rem;
    color: #5a6d86;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.mini-kpi-value {
    font-size: 1rem;
    font-weight: 700;
    color: #1d3558;
    margin-top: 2px;
}
</style>
""",
        unsafe_allow_html=True,
)

st.markdown(
        """
<div class="hero">
    <h1 class="hero-title">Cardio Detection Engine Control Deck</h1>
    <p class="hero-sub">High-fidelity clinical risk screening with audit-ready outputs, confidence indicators, and model telemetry.</p>
    <div class="hero-tags">
      <span class="hero-tag">Engine Profile: CVD Binary Risk</span>
      <span class="hero-tag">Mode: Real-time Inference</span>
      <span class="hero-tag">Output: Clinical Report JSON/TXT</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Model"
OUTPUT_DIR = BASE_DIR / "outputs"


def discover_model_paths() -> list[Path]:
    files: list[Path] = []
    if MODEL_DIR.exists():
        files.extend(sorted(MODEL_DIR.glob("*.pkl")))
    if OUTPUT_DIR.exists():
        files.extend(sorted(OUTPUT_DIR.glob("*.pkl")))

    unique: dict[str, Path] = {}
    for path in files:
        unique.setdefault(path.name, path)
    return list(unique.values())


model_paths = discover_model_paths()
if not model_paths:
    st.warning("No pretrained model files found in Model/ or outputs/.")
    st.stop()

default_name = "best_model.pkl" if any(p.name == "best_model.pkl" for p in model_paths) else model_paths[0].name
default_index = next(i for i, p in enumerate(model_paths) if p.name == default_name)

selected_model_name = st.selectbox(
    "Detection engine",
    options=[p.name for p in model_paths],
    index=default_index,
    format_func=lambda n: Path(n).stem,
)

resolved_model_path = next(p for p in model_paths if p.name == selected_model_name)

try:
    model_bundle = joblib.load(resolved_model_path)
except Exception as exc:
    st.error(f"Failed to load model '{resolved_model_path.name}': {exc}")
    st.stop()

model = model_bundle["model"]
feature_columns = model_bundle["feature_columns"]
model_name = model_bundle.get("model_name", "Unknown")

left, mid, right = st.columns(3)
with left:
        st.markdown(
                f"""
<div class="metric-card">
    <div class="metric-label">Active Model</div>
    <div class="metric-value">{model_name}</div>
</div>
""",
                unsafe_allow_html=True,
        )
with mid:
        st.markdown(
                f"""
<div class="metric-card">
    <div class="metric-label">Model Path</div>
    <div class="metric-value mono">{resolved_model_path.name}</div>
</div>
""",
                unsafe_allow_html=True,
        )
with right:
        st.markdown(
                f"""
<div class="metric-card">
    <div class="metric-label">Engine State</div>
    <div class="metric-value">Armed and Ready</div>
</div>
""",
                unsafe_allow_html=True,
        )

st.markdown("<div class='section-title'>Patient Data Intake</div>", unsafe_allow_html=True)
st.caption("All fields are required. This form has no pre-filled patient values.")

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "report_data" not in st.session_state:
    st.session_state.report_data = None
if "row_data" not in st.session_state:
    st.session_state.row_data = None

field_defaults = {
    "age_years_raw": "",
    "height_raw": "",
    "weight_raw": "",
    "ap_hi_raw": "",
    "ap_lo_raw": "",
    "cholesterol": "Select",
    "gluc": "Select",
    "smoke": "Select",
    "alco": "Select",
    "active": "Select",
}
for key, default_value in field_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


def apply_high_risk_preset() -> None:
    st.session_state.age_years_raw = "62"
    st.session_state.height_raw = "165"
    st.session_state.weight_raw = "90"
    st.session_state.ap_hi_raw = "170"
    st.session_state.ap_lo_raw = "102"
    st.session_state.cholesterol = 3
    st.session_state.gluc = 3
    st.session_state.smoke = 1
    st.session_state.alco = 1
    st.session_state.active = 0


def apply_low_risk_preset() -> None:
    st.session_state.age_years_raw = "41"
    st.session_state.height_raw = "172"
    st.session_state.weight_raw = "68"
    st.session_state.ap_hi_raw = "118"
    st.session_state.ap_lo_raw = "76"
    st.session_state.cholesterol = 1
    st.session_state.gluc = 1
    st.session_state.smoke = 0
    st.session_state.alco = 0
    st.session_state.active = 1


def build_compatible_row(values: dict[str, Any], selected_feature_columns: list[str]):
    row = build_single_row_from_values(**values)

    # Compatibility for pretrained models that were trained on age in days and gender.
    if "age" in selected_feature_columns and "age" not in row.columns:
        row["age"] = (row["age_years"] * 365.0).round().astype(int)
    if "gender" in selected_feature_columns and "gender" not in row.columns:
        row["gender"] = 1

    # Fill any still-missing features to avoid runtime index errors.
    for col in selected_feature_columns:
        if col not in row.columns:
            row[col] = 0

    return row


def parse_float_input(label: str, value: str, min_value: float, max_value: float) -> float:
    if value.strip() == "":
        raise ValueError(f"{label} is required.")
    parsed = float(value)
    if parsed < min_value or parsed > max_value:
        raise ValueError(f"{label} must be between {min_value} and {max_value}.")
    return parsed

col1, col2, col3 = st.columns(3)

with col1:
    age_years_raw = st.text_input("Age (years)", key="age_years_raw", placeholder="e.g. 54")
    height_raw = st.text_input("Height (cm)", key="height_raw", placeholder="e.g. 168")
    weight_raw = st.text_input("Weight (kg)", key="weight_raw", placeholder="e.g. 79")

with col2:
    ap_hi_raw = st.text_input("Systolic BP (ap_hi)", key="ap_hi_raw", placeholder="e.g. 145")
    ap_lo_raw = st.text_input("Diastolic BP (ap_lo)", key="ap_lo_raw", placeholder="e.g. 92")
    cholesterol = st.selectbox(
        "Cholesterol",
        options=["Select", 1, 2, 3],
        key="cholesterol",
        format_func=lambda x: "Select" if x == "Select" else str(x),
    )

with col3:
    gluc = st.selectbox(
        "Glucose",
        options=["Select", 1, 2, 3],
        key="gluc",
        format_func=lambda x: "Select" if x == "Select" else str(x),
    )
    smoke = st.selectbox(
        "Smoking",
        options=["Select", 0, 1],
        key="smoke",
        format_func=lambda x: "Select" if x == "Select" else ("Yes" if x == 1 else "No"),
    )
    alco = st.selectbox(
        "Alcohol",
        options=["Select", 0, 1],
        key="alco",
        format_func=lambda x: "Select" if x == "Select" else ("Yes" if x == 1 else "No"),
    )
    active = st.selectbox(
        "Physically active",
        options=["Select", 0, 1],
        key="active",
        format_func=lambda x: "Select" if x == "Select" else ("Yes" if x == 1 else "No"),
    )

preview_col1, preview_col2, preview_col3 = st.columns(3)

preview_bmi = "-"
preview_bp = "-"
preview_lifestyle = "-"

try:
    if height_raw.strip() and weight_raw.strip():
        h_m = float(height_raw) / 100.0
        if h_m > 0:
            preview_bmi = f"{(float(weight_raw) / (h_m * h_m)):.1f}"
except ValueError:
    preview_bmi = "-"

try:
    if ap_hi_raw.strip() and ap_lo_raw.strip():
        preview_bp = f"{float(ap_hi_raw) - float(ap_lo_raw):.0f}"
except ValueError:
    preview_bp = "-"

if smoke != "Select" and alco != "Select" and active != "Select":
    score = int(smoke) + int(alco) + (0 if int(active) == 1 else 1)
    preview_lifestyle = f"{score}/3"

with preview_col1:
    st.markdown(
        f"""
<div class="mini-kpi">
  <div class="mini-kpi-label">Live BMI</div>
  <div class="mini-kpi-value">{preview_bmi}</div>
</div>
""",
        unsafe_allow_html=True,
    )
with preview_col2:
    st.markdown(
        f"""
<div class="mini-kpi">
  <div class="mini-kpi-label">Pulse Pressure</div>
  <div class="mini-kpi-value">{preview_bp}</div>
</div>
""",
        unsafe_allow_html=True,
    )
with preview_col3:
    st.markdown(
        f"""
<div class="mini-kpi">
  <div class="mini-kpi-label">Lifestyle Risk Index</div>
  <div class="mini-kpi-value">{preview_lifestyle}</div>
</div>
""",
        unsafe_allow_html=True,
    )

action_run_col, action_high_col, action_low_col = st.columns([2, 1, 1])

with action_run_col:
    run_prediction = st.button("Run risk inference", type="primary", use_container_width=True)
with action_high_col:
    st.button(
        "High risk details",
        use_container_width=True,
        on_click=apply_high_risk_preset,
    )
with action_low_col:
    st.button(
        "Low risk details",
        use_container_width=True,
        on_click=apply_low_risk_preset,
    )

if run_prediction:
    try:
        age_years = parse_float_input("Age (years)", age_years_raw, 18.0, 120.0)
        height = parse_float_input("Height (cm)", height_raw, 100.0, 230.0)
        weight = parse_float_input("Weight (kg)", weight_raw, 30.0, 200.0)
        ap_hi = parse_float_input("Systolic BP (ap_hi)", ap_hi_raw, 70.0, 250.0)
        ap_lo = parse_float_input("Diastolic BP (ap_lo)", ap_lo_raw, 40.0, 170.0)

        if (
            cholesterol == "Select"
            or gluc == "Select"
            or smoke == "Select"
            or alco == "Select"
            or active == "Select"
        ):
            raise ValueError("Please select values for Cholesterol, Glucose, Smoking, Alcohol, and Physically active.")

        cholesterol = int(cholesterol)
        gluc = int(gluc)
        smoke = int(smoke)
        alco = int(alco)
        active = int(active)

        if ap_lo > ap_hi:
            st.error("Diastolic BP cannot be greater than systolic BP.")
            st.stop()

    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    input_values = {
        "age_years": age_years,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
    }

    row = build_compatible_row(input_values, feature_columns)

    X = row[feature_columns]
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)

    report_data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "predicted_class": pred,
        "predicted_label": "High CVD risk" if pred == 1 else "Low CVD risk",
        "risk_probability": round(proba, 6),
        "confidence_margin": round(abs(proba - 0.5), 6),
        "input": input_values,
        "computed_features": row.to_dict(orient="records")[0],
    }

    st.session_state.prediction_result = {
        "pred": pred,
        "proba": proba,
    }
    st.session_state.report_data = report_data
    st.session_state.row_data = row


if st.session_state.prediction_result is not None and st.session_state.report_data is not None:
    pred = st.session_state.prediction_result["pred"]
    proba = st.session_state.prediction_result["proba"]
    report_data = st.session_state.report_data

    st.markdown("<div class='section-title'>Inference Output</div>", unsafe_allow_html=True)
    status_class = "status-high" if pred == 1 else "status-low"
    status_text = "High CVD Risk" if pred == 1 else "Low CVD Risk"

    st.markdown(
        f"""
<div class="result-panel">
    <span class="status-pill {status_class}">Engine Verdict: {status_text}</span>
  <div><strong>Predicted Class:</strong> {pred}</div>
  <div><strong>Risk Probability:</strong> {proba:.2%}</div>
  <div><strong>Confidence Margin:</strong> {abs(proba - 0.5):.4f}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.progress(min(max(proba, 0.0), 1.0), text="Risk intensity channel")

    with st.expander("Computed clinical features"):
        st.dataframe(st.session_state.row_data, use_container_width=True)

    txt_report = "\n".join(
        [
            "Cardio Prediction Report",
            f"Timestamp: {report_data['timestamp']}",
            f"Model: {report_data['model_name']}",
            f"Predicted Class: {report_data['predicted_class']}",
            f"Predicted Label: {report_data['predicted_label']}",
            f"Risk Probability: {report_data['risk_probability']:.6f}",
            f"Confidence Margin: {report_data['confidence_margin']:.6f}",
            "",
            "Input Values:",
            json.dumps(report_data["input"], indent=2),
            "",
            "Computed Features:",
            json.dumps(report_data["computed_features"], indent=2),
        ]
    )

    st.markdown("<div class='section-title'>Export Evidence Pack</div>", unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="Download TXT Report",
            data=txt_report,
            file_name="cardio_prediction_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            label="Download JSON Report",
            data=json.dumps(report_data, indent=2),
            file_name="cardio_prediction_report.json",
            mime="application/json",
            use_container_width=True,
        )

st.markdown("---")

st.markdown("<div class='section-title'>Model Analytics Dashboard</div>", unsafe_allow_html=True)

analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs(["Model Leaderboard", "Risk Profile", "Feature Analysis"])

with analytics_tab1:
    st.markdown("### Model Performance Comparison")

    metrics_candidates = [
        resolved_model_path.parent / "metrics.csv",
        MODEL_DIR / "metrics.csv",
        OUTPUT_DIR / "metrics.csv",
    ]
    metrics_path = next((p for p in metrics_candidates if p.exists()), None)
    if metrics_path is not None:
        metrics_df = pd.read_csv(metrics_path)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ROC-AUC & F1 Score")
            if PLOTLY_AVAILABLE:
                fig_perf = go.Figure()
                fig_perf.add_trace(go.Bar(
                    x=metrics_df['Model'],
                    y=metrics_df['ROC_AUC'],
                    name='ROC-AUC',
                    marker_color='#2858ff',
                    opacity=0.8
                ))
                fig_perf.add_trace(go.Bar(
                    x=metrics_df['Model'],
                    y=metrics_df['F1'],
                    name='F1 Score',
                    marker_color='#13b2a7',
                    opacity=0.8
                ))
                fig_perf.update_layout(
                    barmode='group',
                    title_text="",
                    yaxis_title="Score",
                    xaxis_title="Model",
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=380,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig_perf, use_container_width=True)
            else:
                st.bar_chart(
                    metrics_df.set_index("Model")[["ROC_AUC", "F1"]],
                    use_container_width=True,
                )
        
        with col2:
            st.markdown("#### Precision, Recall & Accuracy")
            if PLOTLY_AVAILABLE:
                fig_metrics = go.Figure()
                for metric in ['Precision', 'Recall', 'Accuracy']:
                    fig_metrics.add_trace(go.Scatter(
                        x=metrics_df['Model'],
                        y=metrics_df[metric],
                        mode='lines+markers',
                        name=metric,
                        line=dict(width=3)
                    ))
                fig_metrics.update_layout(
                    title_text="",
                    yaxis_title="Score",
                    xaxis_title="Model",
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=380,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig_metrics, use_container_width=True)
            else:
                st.line_chart(
                    metrics_df.set_index("Model")[["Precision", "Recall", "Accuracy"]],
                    use_container_width=True,
                )
        
        st.markdown("#### Detailed Metrics Table")
        metrics_display = metrics_df.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']:
            metrics_display[col] = metrics_display[col].apply(lambda x: f"{x:.4f}")
        st.dataframe(metrics_display, use_container_width=True, hide_index=True)
    else:
        st.info("Metrics data not available. Train the model first.")

with analytics_tab2:
    st.markdown("### Risk Profile Intelligence")
    
    if st.session_state.row_data is not None and st.session_state.prediction_result is not None:
        row = st.session_state.row_data
        pred = st.session_state.prediction_result["pred"]
        proba = st.session_state.prediction_result["proba"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Clinical Indicators")
            risk_profile = {
                "BMI": row['BMI'].values[0],
                "BP Systolic": row['ap_hi'].values[0],
                "BP Diastolic": row['ap_lo'].values[0],
                "Pulse Pressure": row['pulse_pressure'].values[0],
                "Risk Factors Count": row['risk_factors'].values[0],
                "Risk Score": row['risk_score'].values[0]
            }
            
            colors = []
            for key, val in risk_profile.items():
                if key == "BMI":
                    colors.append('#be2d37' if val >= 30 else '#ff9800' if val >= 25 else '#148856')
                elif key in ["BP Systolic", "BP Diastolic"]:
                    colors.append('#be2d37' if val >= 140 else '#ff9800' if val >= 130 else '#148856')
                elif key == "Risk Score":
                    colors.append('#be2d37' if val >= 60 else '#ff9800' if val >= 40 else '#148856')
                else:
                    colors.append('#2858ff')
            
            if PLOTLY_AVAILABLE:
                fig_radar = go.Figure(data=go.Bar(
                    y=list(risk_profile.keys()),
                    x=list(risk_profile.values()),
                    orientation='h',
                    marker=dict(color=colors),
                    text=[f"{v:.1f}" for v in risk_profile.values()],
                    textposition='auto'
                ))
                fig_radar.update_layout(
                    title_text="",
                    xaxis_title="Value",
                    yaxis_title="",
                    height=350,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=140, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.bar_chart(pd.DataFrame({"value": list(risk_profile.values())}, index=list(risk_profile.keys())))
        
        with col2:
            st.markdown("#### Risk Distribution Gauge")
            
            risk_levels = {
                'Low Risk': sum(1 for p in [0.2, 0.3, 0.25] if p < 0.5),
                'Medium Risk': 1 if 0.4 <= proba < 0.7 else 0,
                'High Risk': sum(1 for p in [0.75, 0.8, 0.85] if p >= 0.7)
            }
            
            if PLOTLY_AVAILABLE:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=proba * 100,
                    title={'text': "CVD Risk Probability (%)"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': '#2858ff'},
                        'steps': [
                            {'range': [0, 33], 'color': "#e8f5e9"},
                            {'range': [33, 66], 'color': "#fff3e0"},
                            {'range': [66, 100], 'color': "#ffebee"}
                        ],
                        'threshold': {
                            'line': {'color': '#be2d37', 'width': 2},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=350,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.metric("CVD Risk Probability", f"{proba * 100:.1f}%")
                st.progress(min(max(proba, 0.0), 1.0), text="Risk gauge")
    else:
        st.info("Run a prediction to see risk profile analysis.")

with analytics_tab3:
    st.markdown("### Feature Impact Analysis")
    
    if st.session_state.row_data is not None:
        row = st.session_state.row_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Key Feature Contributions")
            
            feature_importance = {
                "Age (years)": row['age_years'].values[0] / 100,
                "Weight": row['weight'].values[0] / 150,
                "Height": row['height'].values[0] / 200,
                "Cholesterol": row['cholesterol'].values[0] / 3,
                "Glucose": row['gluc'].values[0] / 3,
                "Systolic BP": row['ap_hi'].values[0] / 200,
                "Diastolic BP": row['ap_lo'].values[0] / 120,
                "Smoking": row['smoke'].values[0],
                "Alcohol": row['alco'].values[0],
                "Activity": row['active'].values[0]
            }
            
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            if PLOTLY_AVAILABLE:
                fig_importance = go.Figure(data=go.Bar(
                    y=[f[0] for f in sorted_features],
                    x=[f[1] for f in sorted_features],
                    orientation='h',
                    marker_color='#2858ff',
                    text=[f"{f[1]:.2f}" for f in sorted_features],
                    textposition='auto'
                ))
                fig_importance.update_layout(
                    title_text="",
                    xaxis_title="Normalized Contribution",
                    yaxis_title="",
                    height=380,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=120, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.bar_chart(pd.DataFrame({"importance": [f[1] for f in sorted_features]}, index=[f[0] for f in sorted_features]))
        
        with col2:
            st.markdown("#### Biometric Status Overview")
            
            biometric_status = []
            bmi = row['BMI'].values[0]
            bmi_status = "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
            
            is_hypertensive = row['is_hypertensive'].values[0]
            status = ["✓ Normal BP", "⚠ Hypertensive"][int(is_hypertensive)]
            
            high_chol = row['high_cholesterol'].values[0]
            chol_status = "✓ Normal" if not high_chol else "⚠ Elevated"
            
            high_gluc = row['high_glucose'].values[0]
            gluc_status = "✓ Normal" if not high_gluc else "⚠ Elevated"
            
            smoker = row['smoke'].values[0]
            smoke_status = "✓ Non-smoker" if not smoker else "⚠ Smoker"
            
            active = row['active'].values[0]
            activity_status = "✓ Active" if active else "⚠ Inactive"
            
            status_data = {
                "BMI Category": bmi_status,
                "Blood Pressure": status,
                "Cholesterol": chol_status,
                "Glucose": gluc_status,
                "Smoking": smoke_status,
                "Physical Activity": activity_status
            }
            
            for key, val in status_data.items():
                color = "green" if "✓" in val else "orange"
                st.markdown(f"<div style='padding: 8px; background: rgba(200,200,200,0.1); border-left: 3px solid {color}; margin: 6px 0; border-radius: 4px;'><strong>{key}:</strong> {val}</div>", unsafe_allow_html=True)
    else:
        st.info("Run a prediction to see feature analysis.")

st.markdown("---")
st.caption("Clinical decision support only. Use alongside qualified medical assessment.")
