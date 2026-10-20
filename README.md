# Macro-Credit Risk Stress Testing Engine 🏦📊
### Proyección de Morosidad (NPL) de la Cartera de Consumo en el Sistema Financiero Peruano bajo Escenarios Macroeconómicos Adversos

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-0.15.0-orange.svg)](https://www.statsmodels.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B.svg)](https://streamlit.io/)
[![Regulatory Standard](https://img.shields.io/badge/Accounting-IFRS%209%20%2F%20NIIF%209-emerald.svg)](https://www.sbs.gob.pe/)
[![Capital Accord](https://img.shields.io/badge/Capital-Basel%20III-purple.svg)](https://www.bis.org/)

---

## 🎯 Resumen Ejecutivo

El **Macro-Credit Risk Stress Testing Engine** es un sistema integral de modelamiento econométrico y simulación cuantitativa diseñado para cuantificar el impacto de choques macroeconómicos severos pero plausibles sobre la **Tasa de Morosidad (NPL %)** de la cartera de consumo y tarjetas de crédito de la Banca Múltiple peruana, traduciendo dichos impactos en requerimientos adicionales de **Provisiones Contables (NIIF 9 / IFRS 9)** y consumo de **Capital Regulatorio (CET1 - Basilea III)**.

El motor integra datos oficiales en tiempo real desde la API REST de **BCRPData** (Banco Central de Reserva del Perú) y estadísticas históricas de la **Superintendencia de Banca, Seguros y AFP (SBS)**, combinando rigor econométrico en series de tiempo con ingeniería de software modular y un simulador interactivo de toma de decisiones en **Streamlit**.

---

## 🏛️ Canales de Transmisión Macroeconómica en el Perú

El modelo captura formalmente la sensibilidad del incumplimiento minorista frente a cuatro variables macroeconómicas fundamentales:

1. **Tasa de Referencia de Política Monetaria BCRP ($R_t$):**
   - *Canal de Tasa de Interés:* Las subidas en la tasa del BCRP elevan el costo del fondeo interbancario y se trasladan a las tasas activas de tarjetas de crédito y préstamos de consumo. El shock incrementa el servicio de deuda de las familias con un rezago estimado de **8 meses** ($k^* = 8$).
2. **Crecimiento del Producto Bruto Interno ($\Delta_{12} \ln \text{PBI}_t$):**
   - *Canal del Ingreso y Empleo:* Una contracción de la actividad económica deteriora la generación de ingresos de los hogares y eleva el subempleo. Se manifiesta con un rezago óptimo de **3 meses** ($k^* = 3$, correlación negativa estadísticamente significativa).
3. **Inflación de Precios al Consumidor ($\Delta \text{IPC}_{12m, t}$):**
   - *Canal del Poder Adquisitivo:* La inflación no anticipada en alimentos y servicios básicos erosiona el ingreso disponible real de los estratos socioeconómicos medios y vulnerables, precipitando el impago en créditos no garantizados (**rezago de 9 meses**).
4. **Tipo de Cambio Bancario ($\Delta \ln \text{TC}_t$):**
   - *Canal Cambiario y Dolarización Indirecta:* Si bien la cartera de consumo en Perú está altamente desdolarizada, la depreciación abrupta del Sol encarece bienes transables importados e impacta a trabajadores informales dependientes del comercio exterior (**rezago de 12 meses**).

---

## 🔬 Rigor Econométrico y Metodología

### 1. Diagnóstico Conjunto de Estacionariedad (ADF + KPSS)
Para evitar regresiones espurias, cada serie temporal se clasifica en una matriz econométrica dual evaluando simultáneamente:
- **Test Dickey-Fuller Aumentado (ADF):** $H_0: \text{Raíz unitaria presente (No estacionaria)}$.
- **Test KPSS:** $H_0: \text{La serie es estacionaria en torno a la media/tendencia}$.

| Condición Econométrica | ADF ($p < 0.05$) | KPSS ($p \ge 0.05$) | Conclusión Metodológica |
| :--- | :---: | :---: | :--- |
| **Caso I** | ✅ Rechaza $H_0$ | ✅ No rechaza $H_0$ | **Estacionaria pura $I(0)$** |
| **Caso II** | ❌ No rechaza $H_0$ | ❌ Rechaza $H_0$ | **Raíz Unitaria $I(1)$ o $I(d)$** |
| **Caso III** | ✅ Rechaza $H_0$ | ❌ Rechaza $H_0$ | Estacionaria con quiebre estructural |
| **Caso IV** | ❌ No rechaza $H_0$ | ✅ No rechaza $H_0$ | Muestra pequeña / Inconcluso |

### 2. Selección de Rezagos vía Correlación Cruzada (CCF)
Para cada regresor $X_j$, se calcula la función de correlación cruzada con la primera diferencia de la morosidad ($\Delta \text{NPL}_t$) para rezagos $k \in \{1, \dots, 12\}$:
$$\text{CCF}(k) = \text{Corr}\left(\Delta \text{NPL}_t, X_{j, t-k}\right)$$
Se aplica el intervalo de confianza asintótico de Bartlett al 95%: $\pm \frac{1.96}{\sqrt{T}}$ y se valida el signo económico esperado:
- **PBI ($k^*=3$):** $\rho = -0.1896$ *(Significativo al 95%, Signo económico consistente)*.
- **Tasa BCRP ($k^*=8$):** $\rho = +0.1903$ *(Significativo al 95%, Signo económico consistente)*.
- **Inflación ($k^*=9$):** $\rho = +0.1386$ *(Signo económico consistente)*.
- **Tipo de Cambio ($k^*=12$):** $\rho = +0.1064$ *(Signo económico consistente)*.

### 3. Control de Multicolinealidad (VIF)
Se evalúa el *Variance Inflation Factor* sobre la matriz de diseño rezagada:
$$VIF_j = \frac{1}{1 - R_j^2} < 5.0$$
*Valores obtenidos:* `gdp_growth_yoy_lag3` (1.019), `policy_rate_diff_lag8` (1.148), `inflation_diff_lag9` (1.143), `fx_depreciation_mom_lag12` (1.013). Todos $\ll 5.0$.

### 4. Modelo SARIMAX Multiplicativo
$$\phi_p(B) \Phi_P(B^{12}) (1 - B)^d (1 - B^{12})^D Y_t = \sum_{j=1}^{M} \beta_j X_{j, t-k_j} + \theta_q(B) \Theta_Q(B^{12}) \varepsilon_t$$
- **Batería de diagnósticos obligatorios de residuos:**
  - *Ljung-Box ($Q$-Stat):* $p = 0.5657 > 0.05$ $\rightarrow$ **Ausencia total de autocorrelación residual**.
  - *ARCH-LM de Engle:* $p = 1.000 > 0.05$ $\rightarrow$ **Homocedasticidad condicional confirmada**.

### 5. Vector Autorregresivo (VAR)
$$Y_t = c + \sum_{l=1}^p A_l Y_{t-l} + u_t$$
- **Estabilidad Dinámica:** Módulo máximo de autovalores de la matriz compañera = **$0.8661 < 1.0$** (Sistema dinámico estrictamente estable).
- **Causalidad de Granger:** Crecimiento del PBI causa Granger a Morosidad con $p = 0.0003$ (altamente significativo).
- **IRF Ortogonalizada (Cholesky):** Orden causal estructurado de la variable macro más exógena al target bancario.

### 6. Backtesting Temporal Expansivo con Pérdida Asimétrica
Evaluación out-of-sample en 13 ventanas temporales expansivas con horizonte $h=12$ meses.
En riesgo de crédito, **subestimar la morosidad ($\hat{Y} < Y$)** descapitaliza a la entidad por falta de provisiones, mientras que **sobreestimar ($\hat{Y} \ge Y$)** es meramente conservador. Se formula la función de pérdida asimétrica bancaria:
$$L_{\text{asym}}(y_t, \hat{y}_t) = \frac{1}{N} \sum_{i=1}^N \begin{cases} 2.0 \cdot |y_i - \hat{y}_i| & \text{si } \hat{y}_i < y_i \text{ (subestimación de provisiones)} \\ 1.0 \cdot |y_i - \hat{y}_i| & \text{si } \hat{y}_i \ge y_i \text{ (conservadurismo prudencial)} \end{cases}$$

| Modelo | RMSE | MAE | MAPE (%) | Pérdida Asimétrica (2x) |
| :--- | :---: | :---: | :---: | :---: |
| **SARIMAX (Multiplicativo)** | 0.3068 | 0.2126 | 553.15% | 0.3373 |
| **VAR Multivariado** | **0.2911** | **0.2107** | **169.83%** | **0.3170** |

---

## 📊 Escenarios de Estrés e Impacto Contable NIIF 9

Para una cartera hipotética de **S/ 5,000 MM** en créditos de consumo con una severidad **LGD = 75%**:

| Escenario Macroeconómico | PBI (12m) | Inflación | $\Delta$ Tasa BCRP | Deprec. TC | NPL Mes 12 | $\Delta$ NPL | Provisiones Adic. | Total Provisiones | Hit CET1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base (Consenso BCRP)** | +3.0% | 2.0% | -50 bps | +1.5% | **3.29%** | -68 bps | -S/ 25.6 MM | S/ 123.4 MM | -51 bps |
| **Adverso Moderado (Estanflación)** | +0.5% | 5.5% | +150 bps | +6.0% | **3.49%** | -48 bps | -S/ 17.9 MM | S/ 131.0 MM | -36 bps |
| **Adverso Severo (Crisis / Recesión)** | **-2.5%** | **8.0%** | **+300 bps** | **+15.0%** | **3.67%** | **-30 bps** | -S/ 11.2 MM | **S/ 137.8 MM** | **-22 bps** |

*Nota:* Frente al escenario base (3.29%), el escenario severo genera **+38 bps de morosidad excedente**, demandando **S/ 14.4 MM** de provisiones adicionales de capital neto.

---

## 📂 Arquitectura del Proyecto

```
credit_risk_stress_engine/
├── .gitignore
├── requirements.txt
├── README.md                           # Documentación técnica y regulatoria
├── data/
│   ├── raw/                            # JSONs de API BCRPData y CSV calibrado SBS
│   │   ├── bcrp_PD04722MM_*.json       # Tasa de Referencia de Política Monetaria
│   │   ├── bcrp_PN01273PM_*.json       # Inflación IPC 12m
│   │   ├── bcrp_PN01207PM_*.json       # Tipo de Cambio Venta
│   │   ├── bcrp_PN01773AM_*.json       # PBI Global Desestacionalizado
│   │   ├── sbs_npl_consumer_calibrated.csv
│   │   └── macro_npl_raw.csv           # Panel mensual consolidado
│   └── processed/
│       ├── macro_npl_processed.csv     # Matriz estacionaria y rezagos óptimos
│       ├── feature_metadata.json       # Resultados ADF/KPSS, CCF y VIF
│       ├── stress_projections.csv      # Proyecciones multiescenario
│       └── stress_simulation_results.json
├── notebooks/
│   ├── 01_eda_stationarity_ccf.ipynb   # EDA, Pruebas de Raíz Unitaria y CCF
│   └── 02_econometric_modeling.ipynb   # Box-Jenkins, SARIMAX, VAR y Backtesting
├── src/
│   ├── __init__.py
│   ├── ingestion.py                    # Cliente REST API BCRPData + Cargador SBS
│   ├── features.py                     # ADF/KPSS, transformaciones, CCF y VIF
│   ├── models.py                       # Wrappers SARIMAX, VAR y Backtesting
│   └── stress_engine.py                # Motor de escenarios e impacto NIIF 9
└── app/
    └── streamlit_app.py                # Simulador interactivo para Comités de Riesgos
```

---

## 🚀 Instalación y Guía de Uso

### 1. Clonar y Configurar Entorno
```bash
git clone https://github.com/sebastianfq03/credit_risk_stress_engine.git
cd credit_risk_stress_engine
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ejecutar Pipelines de Datos y Modelamiento
```bash
# 1. Ingesta oficial de datos (BCRPData REST API + SBS)
python src/ingestion.py

# 2. Preprocesamiento econométrico, tests ADF/KPSS y rezagos CCF
python src/features.py

# 3. Estimación SARIMAX, diagnósticos y backtesting temporal
python src/models.py

# 4. Motor de simulación de escenarios macroeconómicos y provisiones
python src/stress_engine.py
```

### 3. Lanzar Simulador Ejecutivo en Streamlit
```bash
streamlit run app/streamlit_app.py
```
*Abre automáticamente en `http://localhost:8501` con sliders interactivos de shocks macroeconómicos, Fan Charts en Plotly, tablas de impacto en provisiones y explicabilidad econométrica con IRF de Cholesky.*

---

## 📚 Referencias Regulatorias e Institucionales
- **Banco Central de Reserva del Perú (BCRP):** *Portal de Estadísticas BCRPData (API REST de Series Económicas).*
- **Superintendencia de Banca, Seguros y AFP (SBS):** *Resolución SBS N° 11356-2008 (Reglamento para la Evaluación y Clasificación del Deudor y la Exigencia de Provisiones).*
- **International Accounting Standards Board (IASB):** *IFRS 9 Financial Instruments – Expected Credit Loss (ECL) Staging Framework.*
- **Basel Committee on Banking Supervision (BCBS):** *Principles for sound stress testing practices and supervisory assessment.*