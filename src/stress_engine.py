"""
Motor de Stress Testing Macroeconómico e Impacto en Provisiones NIIF 9
======================================================================
Módulo: src/stress_engine.py

Este módulo implementa:
1. MacroScenario: Definición estructurada de escenarios macroeconómicos:
   - Escenario Base (Consenso BCRP)
   - Escenario Adverso Moderado (Estanflación)
   - Escenario Adverso Severo (Crisis / Recesión)
   - Escenario Personalizado (Shocks interactivos)
2. FutureExogBuilder: Generador riguroso de trayectorias futuras con empalme histórico
   y propagación de rezagos óptimos (k=3, 8, 9, 12).
3. StressTester: Motor de simulación multivariada que proyecta la tasa de morosidad
   a 12 meses con intervalos de confianza al 95%.
4. BalanceSheetImpactEngine: Traducción a Pérdida Crediticia Esperada (ECL / NIIF 9),
   cuantificando Provisiones Adicionales requeridas (S/ MM) e impacto en Capital Regulatorio.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Asegurar que la raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.features import FinancialTransformations
from src.models import SARIMAXModelWrapper, VARModelWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =====================================================================
# 1. DEFINICIÓN DE ESCENARIOS MACROECONÓMICOS
# =====================================================================

@dataclass
class MacroScenario:
    """
    Especificación de un escenario de estrés macroeconómico a horizonte de 12 meses.
    """
    name: str
    gdp_growth_yoy: float         # Crecimiento del PBI real interanual al mes 12 (%)
    inflation_12m: float          # Tasa de inflación interanual al mes 12 (%)
    policy_rate_change_bps: float # Variación acumulada de la Tasa BCRP en puntos básicos (bps)
    fx_depreciation_pct: float    # Depreciación acumulada del Tipo de Cambio S/ por US$ (%)
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Escenarios oficiales predefinidos (Consenso BCRP, EBA / CCAR adaptado a Perú)
DEFAULT_SCENARIOS = {
    "base": MacroScenario(
        name="Escenario Base (Consenso BCRP)",
        gdp_growth_yoy=3.0,
        inflation_12m=2.0,
        policy_rate_change_bps=-50.0,    # Relajamiento monetario gradual (-50 bps)
        fx_depreciation_pct=1.5,         # Depreciación cambiaria controlada
        description="Crecimiento en torno al potencial (3.0%), inflación en el centro del rango meta (2.0%) y normalización monetaria."
    ),
    "adverse_moderate": MacroScenario(
        name="Escenario Adverso Moderado (Estanflación)",
        gdp_growth_yoy=0.5,
        inflation_12m=5.5,
        policy_rate_change_bps=150.0,    # Alza de tasa +150 bps para contener inflación
        fx_depreciation_pct=6.0,         # Presión cambiaria moderada
        description="Estancamiento económico con choque de oferta: PBI 0.5%, inflación en 5.5% y endurecimiento de liquidez bancaria."
    ),
    "adverse_severe": MacroScenario(
        name="Escenario Adverso Severo (Crisis / Recesión)",
        gdp_growth_yoy=-2.5,
        inflation_12m=8.0,
        policy_rate_change_bps=300.0,    # Shock de tasa +300 bps ante fuga de capitales
        fx_depreciation_pct=15.0,        # Devaluación abrupta del Sol (+15%)
        description="Recesión severa combinada con shock cambiario e inflacionario: contracción del PBI -2.5% y tasa BCRP en niveles restrictivos."
    )
}


# =====================================================================
# 2. GENERADOR DE TRAYECTORIAS Y EMPALME TEMPORAL DE REZAGOS
# =====================================================================

class FutureExogBuilder:
    """
    Construye la matriz de diseño futura X_{T+1..T+h} garantizando el empalme
    exacto de rezagos temporales con las observaciones históricas crudas.
    """

    def __init__(self, raw_data_path: Union[str, Path] = "data/raw/macro_npl_raw.csv"):
        self.raw_path = Path(raw_data_path)
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Archivo de datos crudos no encontrado en {self.raw_path}")
        self.raw_df = pd.read_csv(self.raw_path, index_col="date", parse_dates=True)

    def build_scenario_paths(
        self,
        scenario: MacroScenario,
        horizon: int = 12
    ) -> pd.DataFrame:
        """
        Genera la trayectoria mensual de las 4 variables macro para el horizonte futuro:
        - gdp_growth_yoy: Transición suave desde el último valor observado hacia la meta del escenario.
        - policy_rate: Senda acumulada de tasa de referencia (nivel).
        - inflation_12m: Senda mensual de inflación interanual.
        - fx_rate: Senda del tipo de cambio con depreciación mensual continua.
        """
        last_date = self.raw_df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS")

        last_gdp_idx = self.raw_df["gdp_index"].iloc[-1]
        last_gdp_growth = FinancialTransformations.log_diff_yoy(self.raw_df["gdp_index"], s=12).dropna().iloc[-1]
        last_rate = self.raw_df["policy_rate"].iloc[-1]
        last_inf = self.raw_df["inflation_12m"].iloc[-1]
        last_fx = self.raw_df["fx_rate"].iloc[-1]

        # 1. Proyección de Crecimiento del PBI interanual (convergencia lineal al target)
        gdp_growth_path = np.linspace(last_gdp_growth, scenario.gdp_growth_yoy, horizon)

        # 2. Proyección de Tasa de Política Monetaria (distribución gradual del shock en bps)
        delta_rate_monthly = (scenario.policy_rate_change_bps / 100.0) / horizon
        rate_diff_path = np.full(horizon, delta_rate_monthly)
        rate_level_path = last_rate + np.cumsum(rate_diff_path)

        # 3. Proyección de Inflación 12 meses
        inf_path = np.linspace(last_inf, scenario.inflation_12m, horizon)
        inf_diff_path = np.diff(np.insert(inf_path, 0, last_inf))

        # 4. Proyección de Tipo de Cambio (depreciación mensual geométricamente distribuida)
        monthly_deprec = (1.0 + scenario.fx_depreciation_pct / 100.0) ** (1.0 / horizon) - 1.0
        fx_deprec_mom_path = np.full(horizon, monthly_deprec * 100.0)
        fx_level_path = last_fx * np.cumprod(1.0 + np.full(horizon, monthly_deprec))

        future_macro = pd.DataFrame({
            "gdp_growth_yoy": gdp_growth_path,
            "policy_rate_level": rate_level_path,
            "policy_rate_diff": rate_diff_path,
            "inflation_12m": inf_path,
            "inflation_diff": inf_diff_path,
            "fx_rate": fx_level_path,
            "fx_depreciation_mom": fx_deprec_mom_path
        }, index=future_dates)

        return future_macro

    def build_lagged_feature_matrix(
        self,
        scenario: MacroScenario,
        horizon: int = 12
    ) -> pd.DataFrame:
        """
        Empalma la historia reciente con la trayectoria futura proyectada y genera
        las 4 columnas rezagadas exactas que requiere el modelo:
        - gdp_growth_yoy_lag3
        - policy_rate_diff_lag8
        - inflation_diff_lag9
        - fx_depreciation_mom_lag12
        """
        # Calcular variables transformadas sobre la historia completa
        hist_df = pd.DataFrame(index=self.raw_df.index)
        hist_df["gdp_growth_yoy"] = FinancialTransformations.log_diff_yoy(self.raw_df["gdp_index"], s=12)
        hist_df["policy_rate_diff"] = self.raw_df["policy_rate"].diff()
        hist_df["inflation_diff"] = self.raw_df["inflation_12m"].diff()
        hist_df["fx_depreciation_mom"] = FinancialTransformations.log_diff_mom(self.raw_df["fx_rate"])

        # Generar futuro proyectado
        fut_df = self.build_scenario_paths(scenario, horizon=horizon)

        # Concatenar historia y futuro en una serie continua
        spliced_df = pd.concat([
            hist_df[["gdp_growth_yoy", "policy_rate_diff", "inflation_diff", "fx_depreciation_mom"]],
            fut_df[["gdp_growth_yoy", "policy_rate_diff", "inflation_diff", "fx_depreciation_mom"]]
        ])

        # Extraer los rezagos exactos
        lagged_spliced = pd.DataFrame(index=spliced_df.index)
        lagged_spliced["gdp_growth_yoy_lag3"] = spliced_df["gdp_growth_yoy"].shift(3)
        lagged_spliced["policy_rate_diff_lag8"] = spliced_df["policy_rate_diff"].shift(8)
        lagged_spliced["inflation_diff_lag9"] = spliced_df["inflation_diff"].shift(9)
        lagged_spliced["fx_depreciation_mom_lag12"] = spliced_df["fx_depreciation_mom"].shift(12)

        # Filtrar únicamente el horizonte futuro proyectado
        future_dates = fut_df.index
        future_X = lagged_spliced.loc[future_dates].copy()
        return future_X


# =====================================================================
# 3. TRADUCCIÓN A BALANCE Y PÉRDIDA ESPERADA (ECL / NIIF 9)
# =====================================================================

@dataclass
class CreditLossImpact:
    scenario_name: str
    portfolio_size_mm: float
    lgd_pct: float
    initial_npl_pct: float
    projected_npl_month12_pct: float
    peak_npl_pct: float
    delta_npl_bps: float
    delta_defaulted_exposure_mm: float
    additional_provisions_required_mm: float
    total_provisions_stressed_mm: float
    provision_coverage_delta_pct: float
    capital_cet1_impact_bps: float


class BalanceSheetImpactEngine:
    """
    Calculadora regulatoria de Pérdida Crediticia Esperada (Expected Credit Loss - IFRS 9 / NIIF 9):
    Traduce el incremento proyectado de morosidad (NPL) a provisiones contables y absorción de capital.
    """

    @staticmethod
    def calculate_impact(
        scenario_name: str,
        npl_trajectory: pd.Series,
        initial_npl_pct: float,
        portfolio_size_mm: float = 5000.0,
        lgd_pct: float = 75.0,
        current_cet1_ratio_pct: float = 12.5
    ) -> CreditLossImpact:
        """
        Calcula el impacto en balance para una trayectoria proyectada de NPL.

        Fórmulas:
        - Delta NPL (bps) = (NPL_12 - NPL_0) * 100
        - Delta Exposición en Default (S/ MM) = Portafolio * (NPL_12 - NPL_0) / 100
        - Provisiones Adicionales (S/ MM) = Delta Exposición en Default * (LGD / 100)
        - Provisiones Totales Estresadas = Portafolio * (NPL_12 / 100) * (LGD / 100)
        - Impacto en Ratio de Capital CET1 = (Provisiones Adicionales / RWA_Cartera) * 100
        """
        proj_npl_12 = float(npl_trajectory.iloc[-1])
        peak_npl = float(npl_trajectory.max())
        
        delta_npl_pct = proj_npl_12 - initial_npl_pct
        delta_npl_bps = delta_npl_pct * 100.0

        # Exposición en default adicional
        delta_defaulted_mm = portfolio_size_mm * (delta_npl_pct / 100.0)

        # Provisiones requeridas bajo NIIF 9 (Stage 3 + Stage 2 SICR)
        lgd_factor = lgd_pct / 100.0
        add_provisions_mm = delta_defaulted_mm * lgd_factor
        total_provisions_mm = portfolio_size_mm * (proj_npl_12 / 100.0) * lgd_factor

        # Ponderación de riesgo de consumo estándar Basilea III (RWA = 100% de exposición)
        rwa_consumer_mm = portfolio_size_mm * 1.0
        # Impacto en ratio CET1 (en puntos básicos de capital regulatorio)
        cet1_hit_pct = (add_provisions_mm / rwa_consumer_mm) * 100.0
        cet1_hit_bps = cet1_hit_pct * 100.0

        return CreditLossImpact(
            scenario_name=scenario_name,
            portfolio_size_mm=portfolio_size_mm,
            lgd_pct=lgd_pct,
            initial_npl_pct=round(initial_npl_pct, 4),
            projected_npl_month12_pct=round(proj_npl_12, 4),
            peak_npl_pct=round(peak_npl, 4),
            delta_npl_bps=round(delta_npl_bps, 2),
            delta_defaulted_exposure_mm=round(delta_defaulted_mm, 2),
            additional_provisions_required_mm=round(add_provisions_mm, 2),
            total_provisions_stressed_mm=round(total_provisions_mm, 2),
            provision_coverage_delta_pct=round(delta_npl_pct * lgd_factor, 4),
            capital_cet1_impact_bps=round(cet1_hit_bps, 2)
        )


# =====================================================================
# 4. MOTOR PRINCIPAL DE STRESS TESTING (StressTester)
# =====================================================================

class StressTester:
    """
    Motor integral de simulación macroeconómica y proyección de estrés bancario.
    Ejecuta proyecciones de morosidad mensual con SARIMAX/VAR para los escenarios
    Base, Moderado y Severo, cuantificando el impacto integral de balance.
    """

    def __init__(
        self,
        processed_data_path: Union[str, Path] = "data/processed/macro_npl_processed.csv",
        raw_data_path: Union[str, Path] = "data/raw/macro_npl_raw.csv"
    ):
        self.processed_path = Path(processed_data_path)
        self.raw_path = Path(raw_data_path)
        
        self.exog_builder = FutureExogBuilder(raw_data_path=self.raw_path)
        self.sarimax_model: Optional[SARIMAXModelWrapper] = None
        self.var_model: Optional[VARModelWrapper] = None
        
        self._load_and_fit_models()

    def _load_and_fit_models(self) -> None:
        """Carga datos procesados y entrena los modelos base."""
        if not self.processed_path.exists():
            raise FileNotFoundError(f"Archivo de datos procesados no encontrado en {self.processed_path}")

        df = pd.read_csv(self.processed_path, index_col="date", parse_dates=True)
        target = df["npl_diff"]
        exog_cols = [c for c in df.columns if c not in ["npl_diff", "npl_level"]]
        exog = df[exog_cols]

        logger.info("Entrenando modelo SARIMAX calibrado para Stress Engine...")
        self.sarimax_model = SARIMAXModelWrapper(order=(1, 0, 1), seasonal_order=(1, 0, 0, 12))
        self.sarimax_model.fit(target, exog=exog)

        logger.info("Entrenando modelo VAR complementario...")
        var_data = df[["gdp_growth_yoy_lag3", "policy_rate_diff_lag8", "inflation_diff_lag9", "fx_depreciation_mom_lag12", "npl_diff"]]
        self.var_model = VARModelWrapper(target_col="npl_diff", maxlags=3)
        self.var_model.fit(var_data, ic="aic")

    def run_simulation(
        self,
        scenario: MacroScenario,
        horizon: int = 12,
        portfolio_size_mm: float = 5000.0,
        lgd_pct: float = 75.0
    ) -> Tuple[pd.DataFrame, CreditLossImpact]:
        """
        Ejecuta la simulación para un escenario macroeconómico específico.

        Returns:
            trajectory_df: DataFrame con las sendas proyectadas de morosidad (media, ci_lower, ci_upper).
            impact: Objeto CreditLossImpact con la cuantificación de provisiones y capital.
        """
        # 1. Construir matriz futura con rezagos correctos
        future_X = self.exog_builder.build_lagged_feature_matrix(scenario, horizon=horizon)

        # 2. Pronóstico con SARIMAX
        pred_res = self.sarimax_model.predict(steps=horizon, exog_future=future_X, alpha=0.05)
        diff_pred = pred_res["pred_mean"]
        diff_lower = pred_res["ci_lower"]
        diff_upper = pred_res["ci_upper"]

        # 3. Reconstruir senda en nivel de Morosidad NPL_t
        initial_npl = float(self.exog_builder.raw_df["npl_consumer"].iloc[-1])
        
        # Acumulación de diferencias: NPL_t = NPL_0 + sum(Delta NPL)
        cum_diff = diff_pred.cumsum()
        proj_level = initial_npl + cum_diff

        # Acumulación de varianza para bandas de confianza del nivel
        # Var(NPL_t) crece con el horizonte sqrt(h)
        step_indices = np.arange(1, horizon + 1)
        sigma_step = (diff_upper.values - diff_lower.values) / (2 * 1.96)
        cum_std = np.sqrt(np.cumsum(sigma_step ** 2))
        
        ci_lower_level = np.clip(proj_level.values - 1.96 * cum_std, 1.0, 20.0)
        ci_upper_level = np.clip(proj_level.values + 1.96 * cum_std, 1.0, 25.0)

        trajectory_df = pd.DataFrame({
            "delta_npl_pred": diff_pred.values,
            "npl_projected": proj_level.values,
            "npl_ci_lower": ci_lower_level,
            "npl_ci_upper": ci_upper_level
        }, index=future_X.index)

        # 4. Cuantificación de impacto en Balance y Pérdida Esperada (NIIF 9)
        impact = BalanceSheetImpactEngine.calculate_impact(
            scenario_name=scenario.name,
            npl_trajectory=trajectory_df["npl_projected"],
            initial_npl_pct=initial_npl,
            portfolio_size_mm=portfolio_size_mm,
            lgd_pct=lgd_pct
        )

        return trajectory_df, impact

    def run_all_scenarios(
        self,
        custom_scenario: Optional[MacroScenario] = None,
        horizon: int = 12,
        portfolio_size_mm: float = 5000.0,
        lgd_pct: float = 75.0,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Ejecuta la batería integral de Stress Testing para todos los escenarios:
        Base, Adverso Moderado, Adverso Severo y Personalizado (si se provee).
        """
        scenarios = dict(DEFAULT_SCENARIOS)
        if custom_scenario is not None:
            scenarios["custom"] = custom_scenario

        all_trajectories = {}
        all_impacts = []

        logger.info(f">>> Ejecutando Simulación de Stress Testing para {len(scenarios)} escenarios...")

        for key, sc in scenarios.items():
            traj, impact = self.run_simulation(
                scenario=sc,
                horizon=horizon,
                portfolio_size_mm=portfolio_size_mm,
                lgd_pct=lgd_pct
            )
            all_trajectories[key] = traj
            all_impacts.append(impact.to_dict() if hasattr(impact, "to_dict") else asdict(impact))
            logger.info(f"Escenario '{sc.name}': NPL proyectado mes 12 = {impact.projected_npl_month12_pct:.2f}% (Delta: {impact.delta_npl_bps:+.0f} bps, Provisiones adicionales: S/ {impact.additional_provisions_required_mm:.2f} MM)")

        # Consolidar tabla comparativa ejecutiva
        impact_df = pd.DataFrame(all_impacts)

        # Consolidar trayectorias para fan chart
        combined_traj_dict = {}
        for key, df_traj in all_trajectories.items():
            combined_traj_dict[f"{key}_npl"] = df_traj["npl_projected"]
            combined_traj_dict[f"{key}_lower"] = df_traj["npl_ci_lower"]
            combined_traj_dict[f"{key}_upper"] = df_traj["npl_ci_upper"]

        multi_traj_df = pd.DataFrame(combined_traj_dict)

        # Formatear llaves de fecha a strings ISO (YYYY-MM-DD) para compatibilidad JSON pura
        formatted_trajectories = {
            str(ts.date()): {col: round(float(val), 4) for col, val in row.items()}
            for ts, row in multi_traj_df.to_dict(orient="index").items()
        }

        results_bundle = {
            "initial_date": str(self.exog_builder.raw_df.index[-1].date()),
            "initial_npl_pct": float(self.exog_builder.raw_df["npl_consumer"].iloc[-1]),
            "portfolio_size_mm": portfolio_size_mm,
            "lgd_pct": lgd_pct,
            "horizon_months": horizon,
            "impact_summary": all_impacts,
            "trajectories": formatted_trajectories
        }

        if save_results:
            out_dir = Path("data/processed")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            multi_traj_df.to_csv(out_dir / "stress_projections.csv")
            with open(out_dir / "stress_simulation_results.json", "w", encoding="utf-8") as f:
                json.dump(results_bundle, f, ensure_ascii=False, indent=2)
            logger.info(f"Resultados de Stress Testing exportados a data/processed/stress_projections.csv y stress_simulation_results.json")

        return results_bundle


if __name__ == "__main__":
    tester = StressTester()
    bundle = tester.run_all_scenarios(portfolio_size_mm=5000.0, lgd_pct=75.0)

    print("\n" + "=" * 70)
    print("RESUMEN EJECUTIVO DE STRESS TESTING (NIIF 9 / ECL):")
    summary_df = pd.DataFrame(bundle["impact_summary"])
    cols_show = [
        "scenario_name", "projected_npl_month12_pct", "delta_npl_bps",
        "additional_provisions_required_mm", "total_provisions_stressed_mm",
        "capital_cet1_impact_bps"
    ]
    print(summary_df[cols_show].to_string(index=False))
