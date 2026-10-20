"""
Simulador Ejecutivo de Stress Testing Macroeconómico para Riesgo de Crédito
===========================================================================
Aplicación Streamlit: app/streamlit_app.py
Cartera de Consumo del Sistema Bancario Peruano (Enfoque IFRS 9 / Basilea III)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Inyectar raíz del proyecto en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.models import SARIMAXModelWrapper, VARModelWrapper
from src.stress_engine import (
    DEFAULT_SCENARIOS,
    BalanceSheetImpactEngine,
    FutureExogBuilder,
    MacroScenario,
    StressTester
)

# Configuración de página Streamlit
st.set_page_config(
    page_title="Macro-Credit Stress Testing Engine | Cartera Consumo Perú",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS para estética ejecutiva institucional bancaria
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 700;
        font-size: 2.1rem;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 400;
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.2rem 0;
    }
    .metric-delta-pos {
        color: #dc2626;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .metric-delta-neg {
        color: #16a34a;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        background-color: #e0f2fe;
        color: #0369a1;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# CARGA DE DATOS Y CACHÉ DEL MOTOR
# =====================================================================

@st.cache_resource(show_spinner="Inicializando modelos econométricos (SARIMAX y VAR)...")
def get_stress_tester() -> StressTester:
    return StressTester(
        processed_data_path=PROJECT_ROOT / "data/processed/macro_npl_processed.csv",
        raw_data_path=PROJECT_ROOT / "data/raw/macro_npl_raw.csv"
    )

@st.cache_data
def load_historical_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = pd.read_csv(PROJECT_ROOT / "data/raw/macro_npl_raw.csv", index_col="date", parse_dates=True)
    proc_df = pd.read_csv(PROJECT_ROOT / "data/processed/macro_npl_processed.csv", index_col="date", parse_dates=True)
    return raw_df, proc_df


# =====================================================================
# SIDEBAR: CONFIGURACIÓN DE ESCENARIOS Y PARÁMETROS BANCARIOS
# =====================================================================

st.sidebar.markdown("### 🏛️ Parámetros de la Cartera")
portfolio_size_input = st.sidebar.number_input(
    "Exposición Total Cartera (S/ Millones):",
    min_value=500.0,
    max_value=100000.0,
    value=5000.0,
    step=250.0,
    help="Exposición total de créditos de consumo y tarjetas de la institución."
)

lgd_input = st.sidebar.slider(
    "Severidad de Pérdida (LGD %):",
    min_value=40.0,
    max_value=95.0,
    value=75.0,
    step=1.0,
    help="Loss Given Default estimada para la cartera minorista no minorista / consumo."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Shocks Macroeconómicos")

scenario_mode = st.sidebar.selectbox(
    "Selección de Escenario a Analizar:",
    options=["Comparativa Multi-Escenario", "Personalizado (What-If)", "Escenario Base", "Adverso Moderado", "Adverso Severo"],
    index=0
)

# Sliders interactivos de shocks
if scenario_mode == "Personalizado (What-If)":
    st.sidebar.markdown("**Parámetros del Escenario Personalizado:**")
    custom_gdp = st.sidebar.slider("Crecimiento PBI Interanual Mes 12 (%):", -6.0, 6.0, -1.0, 0.25)
    custom_inf = st.sidebar.slider("Inflación 12 meses (%):", 0.0, 12.0, 6.5, 0.25)
    custom_rate = st.sidebar.slider("Variación Tasa BCRP (bps):", -200, 500, 200, 25)
    custom_fx = st.sidebar.slider("Depreciación Tipo de Cambio (%):", -5.0, 25.0, 8.0, 0.5)

    active_custom_scenario = MacroScenario(
        name="Escenario Personalizado (What-If)",
        gdp_growth_yoy=custom_gdp,
        inflation_12m=custom_inf,
        policy_rate_change_bps=float(custom_rate),
        fx_depreciation_pct=custom_fx,
        description=f"Shock simulado: PBI {custom_gdp}%, Inflación {custom_inf}%, Tasa {custom_rate:+d} bps, TC {custom_fx:+}%."
    )
else:
    active_custom_scenario = None

st.sidebar.markdown("---")
st.sidebar.caption("📌 Fuente Oficial: **BCRPData** & **SBS Perú**")
st.sidebar.caption("🔬 Modelos: **SARIMAX (1,0,1)(1,0,0)₁₂** con Regresores Rezagados + **VAR(3)**")


# =====================================================================
# EJECUCIÓN DEL MOTOR DE SIMULACIÓN
# =====================================================================

tester = get_stress_tester()
raw_data, processed_data = load_historical_data()

# Ejecutar proyecciones
sim_results = tester.run_all_scenarios(
    custom_scenario=active_custom_scenario,
    portfolio_size_mm=portfolio_size_input,
    lgd_pct=lgd_input,
    save_results=False
)

impact_df = pd.DataFrame(sim_results["impact_summary"])
traj_dict = sim_results["trajectories"]
traj_df = pd.DataFrame(traj_dict).T
traj_df.index = pd.to_datetime(traj_df.index)

initial_npl = sim_results["initial_npl_pct"]
initial_date_str = sim_results["initial_date"]


# =====================================================================
# ENCABEZADO PRINCIPAL
# =====================================================================

st.markdown("""
<div>
    <span class="badge-tag">IFRS 9 / NIIF 9</span>
    <span class="badge-tag">Basilea III</span>
    <span class="badge-tag">BCRP & SBS</span>
    <span class="badge-tag">Stress Testing Engine</span>
</div>
<h1 class="main-header">Motor de Stress Testing para Riesgo de Crédito</h1>
<div class="sub-header">
    Proyección de Morosidad (NPL) y Pérdida Crediticia Esperada (ECL) de la Cartera de Consumo
    en la Banca Múltiple Peruana ante Escenarios Macroeconómicos Adversos
</div>
""", unsafe_allow_html=True)


# =====================================================================
# TARJETAS DE KPIs EJECUTIVOS SUPERIORES
# =====================================================================

base_row = impact_df[impact_df["scenario_name"].str.contains("Base")].iloc[0]
mod_row = impact_df[impact_df["scenario_name"].str.contains("Moderado")].iloc[0]
sev_row = impact_df[impact_df["scenario_name"].str.contains("Severo")].iloc[0]

# Selección de escenario foco para los KPIs
if active_custom_scenario:
    focus_row = impact_df[impact_df["scenario_name"].str.contains("Personalizado")].iloc[0]
    focus_title = "Escenario Personalizado"
elif scenario_mode == "Adverso Moderado":
    focus_row = mod_row
    focus_title = "Escenario Adverso Moderado"
elif scenario_mode == "Adverso Severo":
    focus_row = sev_row
    focus_title = "Escenario Adverso Severo"
else:
    focus_row = sev_row
    focus_title = "Escenario Adverso Severo"

delta_vs_base_bps = (focus_row["projected_npl_month12_pct"] - base_row["projected_npl_month12_pct"]) * 100.0
add_prov_vs_base_mm = (focus_row["total_provisions_stressed_mm"] - base_row["total_provisions_stressed_mm"])

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">NPL Actual (Línea Base {initial_date_str[:7]})</div>
        <div class="metric-value">{initial_npl:.2f}%</div>
        <div style="font-size: 0.8rem; color: #64748b;">Cartera Consumo Banca Múltiple</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">NPL Proyectado (Mes 12)</div>
        <div class="metric-value">{focus_row['projected_npl_month12_pct']:.2f}%</div>
        <div class="{'metric-delta-pos' if focus_row['delta_npl_bps'] > 0 else 'metric-delta-neg'}">
            {focus_row['delta_npl_bps']:+.0f} bps vs. NPL Inicial
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Exceso de NPL vs. Base</div>
        <div class="metric-value">+{delta_vs_base_bps:.0f} bps</div>
        <div style="font-size: 0.8rem; color: #64748b;">Brecha de Severidad Macro</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Provisiones Adicionales (NIIF 9)</div>
        <div class="metric-value">S/ {focus_row['additional_provisions_required_mm']:.1f} MM</div>
        <div class="{'metric-delta-pos' if focus_row['additional_provisions_required_mm'] > 0 else 'metric-delta-neg'}">
            Impacto CET1: {focus_row['capital_cet1_impact_bps']:+.0f} bps
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =====================================================================
# SECCIÓN 1: FAN CHART DE PROYECCIÓN MULTI-ESCENARIO
# =====================================================================

st.markdown("### 📈 Fan Chart de Stress Testing (Horizonte 12 Meses)")

# Preparar serie histórica reciente (últimos 36 meses)
hist_slice = raw_data[["npl_consumer"]].tail(36)

fig = go.Figure()

# 1. Serie histórica observada
fig.add_trace(go.Scatter(
    x=hist_slice.index,
    y=hist_slice["npl_consumer"],
    mode="lines",
    name="Histórico NPL SBS",
    line=dict(color="#0f172a", width=2.5),
    hovertemplate="<b>Histórico</b><br>Fecha: %{x|%Y-%m}<br>Morosidad: %{y:.2f}%<extra></extra>"
))

# Punto de empalme
last_hist_date = hist_slice.index[-1]
last_hist_val = hist_slice["npl_consumer"].iloc[-1]

# 2. Escenario Base
if "base_npl" in traj_df.columns:
    base_dates = [last_hist_date] + list(traj_df.index)
    base_vals = [last_hist_val] + list(traj_df["base_npl"])
    base_lower = [last_hist_val] + list(traj_df["base_lower"])
    base_upper = [last_hist_val] + list(traj_df["base_upper"])

    # Banda de confianza Base
    fig.add_trace(go.Scatter(
        x=base_dates + base_dates[::-1],
        y=base_upper + base_lower[::-1],
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
        name="IC 95% Base"
    ))
    fig.add_trace(go.Scatter(
        x=base_dates,
        y=base_vals,
        mode="lines+markers",
        name="Escenario Base (Consenso BCRP)",
        line=dict(color="#10b981", width=2.5),
        marker=dict(size=4),
        hovertemplate="<b>Base</b><br>Fecha: %{x|%Y-%m}<br>NPL: %{y:.2f}%<extra></extra>"
    ))

# 3. Escenario Adverso Moderado
if "adverse_moderate_npl" in traj_df.columns:
    mod_dates = [last_hist_date] + list(traj_df.index)
    mod_vals = [last_hist_val] + list(traj_df["adverse_moderate_npl"])
    mod_lower = [last_hist_val] + list(traj_df["adverse_moderate_lower"])
    mod_upper = [last_hist_val] + list(traj_df["adverse_moderate_upper"])

    fig.add_trace(go.Scatter(
        x=mod_dates + mod_dates[::-1],
        y=mod_upper + mod_lower[::-1],
        fill="toself",
        fillcolor="rgba(245, 158, 11, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
        name="IC 95% Moderado"
    ))
    fig.add_trace(go.Scatter(
        x=mod_dates,
        y=mod_vals,
        mode="lines+markers",
        name="Adverso Moderado (Estanflación)",
        line=dict(color="#f59e0b", width=2.5, dash="dash"),
        marker=dict(size=4),
        hovertemplate="<b>Moderado</b><br>Fecha: %{x|%Y-%m}<br>NPL: %{y:.2f}%<extra></extra>"
    ))

# 4. Escenario Adverso Severo
if "adverse_severe_npl" in traj_df.columns:
    sev_dates = [last_hist_date] + list(traj_df.index)
    sev_vals = [last_hist_val] + list(traj_df["adverse_severe_npl"])
    sev_lower = [last_hist_val] + list(traj_df["adverse_severe_lower"])
    sev_upper = [last_hist_val] + list(traj_df["adverse_severe_upper"])

    fig.add_trace(go.Scatter(
        x=sev_dates + sev_dates[::-1],
        y=sev_upper + sev_lower[::-1],
        fill="toself",
        fillcolor="rgba(239, 68, 68, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
        name="IC 95% Severo"
    ))
    fig.add_trace(go.Scatter(
        x=sev_dates,
        y=sev_vals,
        mode="lines+markers",
        name="Adverso Severo (Crisis)",
        line=dict(color="#ef4444", width=3),
        marker=dict(size=5),
        hovertemplate="<b>Severo</b><br>Fecha: %{x|%Y-%m}<br>NPL: %{y:.2f}%<extra></extra>"
    ))

# 5. Escenario Personalizado (si está activo)
if "custom_npl" in traj_df.columns:
    cust_dates = [last_hist_date] + list(traj_df.index)
    cust_vals = [last_hist_val] + list(traj_df["custom_npl"])
    fig.add_trace(go.Scatter(
        x=cust_dates,
        y=cust_vals,
        mode="lines+markers",
        name="Personalizado (What-If)",
        line=dict(color="#8b5cf6", width=3, dash="dot"),
        marker=dict(size=5),
        hovertemplate="<b>Personalizado</b><br>Fecha: %{x|%Y-%m}<br>NPL: %{y:.2f}%<extra></extra>"
    ))

# Línea vertical del origen de proyección
fig.add_vline(
    x=last_hist_date,
    line_width=1.5,
    line_dash="dot",
    line_color="#64748b",
    annotation_text="Punto de Empalme (T₀)",
    annotation_position="top left"
)

fig.update_layout(
    height=480,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    hovermode="x unified",
    xaxis=dict(
        title="Horizonte Temporal (Meses)",
        showgrid=True,
        gridcolor="#f1f5f9"
    ),
    yaxis=dict(
        title="Tasa de Morosidad NPL (%)",
        ticksuffix="%",
        showgrid=True,
        gridcolor="#f1f5f9"
    ),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff"
)

st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# PESTAÑAS DETALLADAS: BALANCE NIIF 9, ECONOMETRÍA Y METODOLOGÍA
# =====================================================================

tab_balance, tab_econometrics, tab_methodology = st.tabs([
    "📊 Impacto en Balance y Pérdida Esperada (NIIF 9)",
    "🔬 Explicabilidad Econométrica (SARIMAX & VAR)",
    "📖 Marco Metodológico y Fuentes BCRP/SBS"
])


# ---------------------------------------------------------------------
# PESTAÑA 1: BALANCE NIIF 9
# ---------------------------------------------------------------------
with tab_balance:
    st.markdown("#### Cuantificación de Provisiones Contables y Ratios de Solvencia")
    
    col_t1_left, col_t1_right = st.columns([3, 2])

    with col_t1_left:
        # Tabla comparativa de escenarios
        table_data = []
        for _, r in impact_df.iterrows():
            table_data.append({
                "Escenario": r["scenario_name"],
                "NPL Mes 12 (%)": f"{r['projected_npl_month12_pct']:.2f}%",
                "NPL Pico (%)": f"{r['peak_npl_pct']:.2f}%",
                "Δ NPL (bps)": f"{r['delta_npl_bps']:+.0f} bps",
                "Δ Exposición Default": f"S/ {r['delta_defaulted_exposure_mm']:,.1f} MM",
                "Provisiones Adicionales": f"S/ {r['additional_provisions_required_mm']:,.1f} MM",
                "Total Provisiones": f"S/ {r['total_provisions_stressed_mm']:,.1f} MM",
                "Hit CET1": f"{r['capital_cet1_impact_bps']:+.0f} bps"
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    with col_t1_right:
        # Gráfico de barras de provisiones requeridas
        prov_fig = go.Figure()
        colors = ["#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
        for idx, row in impact_df.iterrows():
            c = colors[idx % len(colors)]
            prov_fig.add_trace(go.Bar(
                x=[row["scenario_name"].split("(")[0].strip()],
                y=[row["total_provisions_stressed_mm"]],
                marker_color=c,
                text=f"S/ {row['total_provisions_stressed_mm']:.1f}M",
                textposition="auto",
                name=row["scenario_name"]
            ))

        prov_fig.update_layout(
            title="Stock de Provisiones Estresadas (S/ MM)",
            height=280,
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis=dict(title="S/ Millones", showgrid=True, gridcolor="#f1f5f9"),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff"
        )
        st.plotly_chart(prov_fig, use_container_width=True)

    st.info(
        f"💡 **Interpretación Regulatoria:** Para una cartera de consumo de **S/ {portfolio_size_input:,.0f} MM** con una severidad "
        f"LGD del **{lgd_input:.0f}%**, un shock severo requeriría provisiones totales por **S/ {sev_row['total_provisions_stressed_mm']:.1f} MM**, "
        f"representando una absorción de capital de solvencia de **{sev_row['capital_cet1_impact_bps']:.0f} puntos básicos** sobre el ratio CET1."
    )


# ---------------------------------------------------------------------
# PESTAÑA 2: EXPLICABILIDAD ECONOMÉTRICA
# ---------------------------------------------------------------------
with tab_econometrics:
    st.markdown("#### Coeficientes y Diagnósticos del Modelo SARIMAX")

    col_e1, col_e2 = st.columns([3, 2])

    with col_e1:
        st.markdown("**Tabla de Regresores Macroeconómicos Rezagados:**")
        coef_df = tester.sarimax_model.get_coefficients_table()
        # Filtrar regresores macroeconómicos
        macro_coefs = coef_df[~coef_df.index.str.contains("ar|ma|sigma", regex=True)]
        st.dataframe(macro_coefs.style.format({
            "Coeficiente": "{:+.5f}",
            "Std_Error": "{:.5f}",
            "z_stat": "{:+.3f}",
            "p_value": "{:.4f}",
            "IC_Lower_95": "{:+.5f}",
            "IC_Upper_95": "{:+.5f}"
        }), use_container_width=True)

    with col_e2:
        st.markdown("**Diagnósticos Regulatorios de Residuos:**")
        diag = tester.sarimax_model.run_diagnostics()
        st.dataframe(diag.summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Funciones de Impulso-Respuesta Ortogonalizadas (VAR IRF - Cholesky)")
    st.caption("Respuesta dinámica del cambio en la Morosidad (Δ NPL) ante un shock de 1 desviación estándar en las variables macroeconómicas:")

    try:
        irf_res = tester.var_model.compute_irf(periods=18)
        # Extraer respuesta de npl_diff
        target_name = "npl_diff"
        target_idx = tester.var_model.variables.index(target_name)
        irf_periods = np.arange(19)

        irf_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Shock en Crecimiento PBI (lag 3)",
                "Shock en Tasa BCRP (lag 8)",
                "Shock en Aceleración Inflación (lag 9)",
                "Shock en Depreciación TC (lag 12)"
            ]
        )

        coord_map = [(1, 1), (1, 2), (2, 1), (2, 2)]
        var_subset = [v for v in tester.var_model.variables if v != target_name][:4]

        for i, var_name in enumerate(var_subset):
            v_idx = tester.var_model.variables.index(var_name)
            resp = irf_res.irfs[:, target_idx, v_idx]
            stderr = irf_res.stderr()[:, target_idx, v_idx]
            lower_b = resp - 1.96 * stderr
            upper_b = resp + 1.96 * stderr

            r, c = coord_map[i]
            # Banda de error
            irf_fig.add_trace(go.Scatter(
                x=list(irf_periods) + list(irf_periods[::-1]),
                y=list(upper_b) + list(lower_b[::-1]),
                fill="toself",
                fillcolor="rgba(148, 163, 184, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False
            ), row=r, col=c)

            # Traza de respuesta
            irf_fig.add_trace(go.Scatter(
                x=irf_periods,
                y=resp,
                mode="lines+markers",
                line=dict(color="#0284c7", width=2),
                showlegend=False
            ), row=r, col=c)

            irf_fig.add_hline(y=0.0, line_dash="dash", line_color="#94a3b8", row=r, col=c)

        irf_fig.update_layout(
            height=460,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff"
        )
        st.plotly_chart(irf_fig, use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudieron generar los gráficos IRF interactivos: {e}")


# ---------------------------------------------------------------------
# PESTAÑA 3: METODOLOGÍA Y FUENTES
# ---------------------------------------------------------------------
with tab_methodology:
    st.markdown("""
    ### 📐 Rigor Econométrico y Fuentes de Datos Oficiales

    #### 1. Formulación del Modelo SARIMAX Multiplicativo:
    $$\\phi_p(B) \\Phi_P(B^{12}) (1 - B)^d (1 - B^{12})^D Y_t = \\sum_{j=1}^{M} \\beta_j X_{j, t-k_j} + \\theta_q(B) \\Theta_Q(B^{12}) \\varepsilon_t$$
    Donde:
    - $Y_t$: Variación mensual de la tasa de morosidad (\\(\\Delta \\text{NPL}_t\\)).
    - $X_{j, t-k_j}$: Regresores macroeconómicos rezagados con orden $k^*$ determinado mediante **Correlación Cruzada (CCF)** con bandas de Bartlett al 95%.
    - Multicolinealidad controlada garantizando $VIF < 5.0$.

    #### 2. Vector Autorregresivo (VAR):
    $$Y_t = c + \\sum_{l=1}^p A_l Y_{t-l} + u_t$$
    - Condición de estabilidad: Todos los valores propios de la matriz compañera tienen módulo estrictamente inferior a 1.0 ($|\\lambda_i| < 1.0$).
    - Funciones de Impulso-Respuesta (IRF) identificadas mediante **Descomposición de Cholesky** ordenada de la variable macroeconómica más exógena al target bancario.

    #### 3. Diccionario de Series Oficiales de BCRPData:
    | Variable Económica | Código BCRP | Frecuencia | Descripción |
    | :--- | :--- | :--- | :--- |
    | **Tasa de Política Monetaria** | `PD04722MM` | Mensual | Tasa de Interés de Referencia del BCRP (%) |
    | **Inflación 12 Meses** | `PN01273PM` | Mensual | IPC Lima Metropolitana (Variación % 12 meses) |
    | **Tipo de Cambio Venta** | `PN01207PM` | Mensual | Tipo de cambio promedio del periodo (S/ por US$) |
    | **Producción Nacional** | `PN01773AM` | Mensual | PBI Global Desestacionalizado (Índice 2007=100) |
    | **Morosidad SBS** | `sbs_npl` | Mensual | Tasa de Morosidad Cartera de Consumo Banca Múltiple |
    """)

st.markdown("---")
st.caption("Macro-Credit Risk Stress Testing Engine v1.0 | Google Antigravity IDE | Desarrollado para Portafolio Cuantitativo Bancario.")
