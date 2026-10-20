"""
Módulo de Modelamiento Econométrico y Validación Regulatoria (src/models.py)
===========================================================================
Este módulo implementa:
1. SARIMAXModelWrapper: Modelo SARIMAX multiplicativo con regresores macroeconómicos
   rezagados, búsqueda en grilla (AIC/BIC) y batería de diagnósticos regulatorios
   (Ljung-Box, Jarque-Bera y ARCH-LM de Engle).
2. VARModelWrapper: Modelo de Vectores Autorregresivos (VAR) con verificación de estabilidad
   dinámica, test de causalidad de Granger, Funciones de Impulso-Respuesta (IRF Cholesky)
   y Descomposición de Varianza del Error de Predicción (FEVD).
3. TimeDimensionBacktester: Backtesting temporal mediante ventana expansiva (Rolling Origin)
   con horizonte h=12 meses, calculando RMSE, MAE, MAPE y Función de Pérdida Financiera
   Asimétrica (penalización 2x por subestimación de morosidad).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =====================================================================
# 1. EVALUACIÓN DE PÉRDIDA FINANCIERA ASIMÉTRICA
# =====================================================================

def asymmetric_loss(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    underestimation_penalty: float = 2.0
) -> float:
    """
    Función de Pérdida Financiera Asimétrica (Asymmetric Credit Risk Loss).
    En gestión de riesgo crediticio bancario y NIIF 9:
    - Subestimar la morosidad (y_pred < y_true) conlleva un riesgo crítico de descapitalización
      y déficit de provisiones ante shocks macroeconómicos. Se penaliza con un factor de 2.0x.
    - Sobreestimar la morosidad (y_pred >= y_true) representa una postura prudente/conservadora.
      Se penaliza con un factor estándar de 1.0x.

    L(e_i) = underestimation_penalty * |y_true - y_pred| si y_pred < y_true
             1.0 * |y_true - y_pred|                     si y_pred >= y_true
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    errors = yp - yt  # error = pred - true. Si yp < yt => error < 0 (subestimación)
    abs_errors = np.abs(yt - yp)
    weights = np.where(yp < yt, underestimation_penalty, 1.0)
    return float(np.mean(weights * abs_errors))


def evaluate_forecast_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Calcula métricas estándar y regulatorias de precisión predictiva.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    
    diff = yp - yt
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    
    # Manejar posibles valores cercanos a cero para MAPE
    denom = np.where(np.abs(yt) < 1e-4, 1e-4, yt)
    mape = float(np.mean(np.abs(diff / denom)) * 100.0)
    
    asym_mae = asymmetric_loss(yt, yp, underestimation_penalty=2.0)
    asym_mse = float(np.mean(np.where(yp < yt, 2.0, 1.0) * (diff ** 2)))

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE_pct": round(mape, 2),
        "Asymmetric_Loss_2x": round(asym_mae, 4),
        "Asymmetric_MSE_2x": round(asym_mse, 4)
    }


# =====================================================================
# 2. MODELO SARIMAX MULTIPLICATIVO CON DIAGNÓSTICOS REGULATORIOS
# =====================================================================

@dataclass
class SARIMAXDiagnostics:
    ljung_box_stat: float
    ljung_box_pvalue: float
    no_autocorrelation: bool
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    residuals_normal: bool
    arch_lm_stat: float
    arch_lm_pvalue: float
    no_arch_effects: bool
    summary_df: pd.DataFrame


class SARIMAXModelWrapper:
    """
    Wrapper econométrico para el modelo SARIMAX:
    phi_p(B) Phi_P(B^s) (1 - B)^d (1 - B^s)^D Y_t = sum_j beta_j X_{j, t-k_j} + theta_q(B) Theta_Q(B^s) eps_t
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 0, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 0, 0, 12)
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model: Optional[SARIMAX] = None
        self.results: Optional[SARIMAXResults] = None
        self.feature_names: List[str] = []
        self.target_name: str = ""

    def fit_grid_search(
        self,
        y: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        p_values: List[int] = [0, 1, 2],
        d_values: List[int] = [0],
        q_values: List[int] = [0, 1, 2],
        P_values: List[int] = [0, 1],
        D_values: List[int] = [0],
        Q_values: List[int] = [0, 1],
        s: int = 12,
        criterion: str = "aic"
    ) -> SARIMAXModelWrapper:
        """
        Búsqueda informada en grilla para seleccionar la especificación (p, d, q) x (P, D, Q)_s
        que minimice el criterio de información (AIC o BIC).
        """
        best_ic = np.inf
        best_order = self.order
        best_seasonal_order = self.seasonal_order
        best_res = None

        logger.info(f"Iniciando búsqueda en grilla SARIMAX (minimizando {criterion.upper()})...")

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    for P in P_values:
                        for D in D_values:
                            for Q in Q_values:
                                current_order = (p, d, q)
                                current_seasonal = (P, D, Q, s)
                                try:
                                    temp_model = SARIMAX(
                                        y,
                                        exog=exog,
                                        order=current_order,
                                        seasonal_order=current_seasonal,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False
                                    )
                                    temp_res = temp_model.fit(disp=False, maxiter=200)
                                    current_ic = temp_res.aic if criterion.lower() == "aic" else temp_res.bic

                                    if current_ic < best_ic:
                                        best_ic = current_ic
                                        best_order = current_order
                                        best_seasonal_order = current_seasonal
                                        best_res = temp_res
                                except Exception:
                                    continue

        if best_res is not None:
            self.order = best_order
            self.seasonal_order = best_seasonal_order
            self.results = best_res
            self.feature_names = list(exog.columns) if exog is not None else []
            self.target_name = y.name or "target"
            logger.info(f"SARIMAX óptimo encontrado: Order={self.order}, Seasonal={self.seasonal_order} | {criterion.upper()}={best_ic:.3f}")
        else:
            logger.warning("Fallo en búsqueda en grilla. Ajustando modelo base de respaldo...")
            self.fit(y, exog=exog)

        return self

    def fit(self, y: pd.Series, exog: Optional[pd.DataFrame] = None) -> SARIMAXModelWrapper:
        """
        Ajusta el modelo SARIMAX con el orden especificado.
        """
        self.target_name = y.name or "target"
        self.feature_names = list(exog.columns) if exog is not None else []

        self.model = SARIMAX(
            y,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.results = self.model.fit(disp=False, maxiter=300)
        logger.info(f"SARIMAX {self.order}x{self.seasonal_order} ajustado exitosamente. AIC={self.results.aic:.3f}, BIC={self.results.bic:.3f}")
        return self

    def run_diagnostics(self, lags_lb: int = 10, lags_arch: int = 5) -> SARIMAXDiagnostics:
        """
        Ejecuta la batería regulatoria completa sobre los residuos estandarizados:
        1. Ljung-Box Q-Test: H0: No autocorrelación serial en residuos (deseable p > 0.05).
        2. Jarque-Bera Test: H0: Residuos normales (deseable p > 0.05).
        3. ARCH-LM de Engle: H0: No heterocedasticidad condicional autorregresiva (deseable p > 0.05).
        """
        if self.results is None:
            raise RuntimeError("El modelo debe ser ajustado previamente llamando a fit().")

        resid = self.results.resid.dropna()

        # 1. Ljung-Box Test
        lb_res = acorr_ljungbox(resid, lags=[lags_lb], return_df=True)
        lb_stat = float(lb_res["lb_stat"].iloc[0])
        lb_pval = float(lb_res["lb_pvalue"].iloc[0])

        # 2. Jarque-Bera Test
        jb_stat, jb_pval, _, _ = jarque_bera(resid)
        jb_stat = float(jb_stat)
        jb_pval = float(jb_pval)

        # 3. ARCH-LM Test
        arch_res = het_arch(resid, nlags=lags_arch)
        arch_stat = float(arch_res[0])
        arch_pval = float(arch_res[1])

        summary_data = [
            {
                "Prueba": "Ljung-Box (Autocorrelación residual)",
                "Estadístico": round(lb_stat, 4),
                "p-value": round(lb_pval, 4),
                "H0": "No existe autocorrelación residual",
                "Cumple (p > 0.05)": "Sí" if lb_pval > 0.05 else "No (alerta)"
            },
            {
                "Prueba": "Jarque-Bera (Normalidad de residuos)",
                "Estadístico": round(jb_stat, 4),
                "p-value": round(jb_pval, 4),
                "H0": "Los residuos distribuyen normalmente",
                "Cumple (p > 0.05)": "Sí" if jb_pval > 0.05 else "No (colas pesadas)"
            },
            {
                "Prueba": "ARCH-LM (Homocedasticidad condicional)",
                "Estadístico": round(arch_stat, 4),
                "p-value": round(arch_pval, 4),
                "H0": "No hay efectos ARCH en la varianza",
                "Cumple (p > 0.05)": "Sí" if arch_pval > 0.05 else "No (volatilidad clustered)"
            }
        ]

        return SARIMAXDiagnostics(
            ljung_box_stat=lb_stat,
            ljung_box_pvalue=lb_pval,
            no_autocorrelation=(lb_pval > 0.05),
            jarque_bera_stat=jb_stat,
            jarque_bera_pvalue=jb_pval,
            residuals_normal=(jb_pval > 0.05),
            arch_lm_stat=arch_stat,
            arch_lm_pvalue=arch_pval,
            no_arch_effects=(arch_pval > 0.05),
            summary_df=pd.DataFrame(summary_data)
        )

    def predict(
        self,
        steps: int = 12,
        exog_future: Optional[pd.DataFrame] = None,
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Genera proyecciones out-of-sample con intervalos de confianza al 95%.
        """
        if self.results is None:
            raise RuntimeError("El modelo debe ser ajustado antes de proyectar.")

        forecast_res = self.results.get_forecast(steps=steps, exog=exog_future)
        pred_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=alpha)

        pred_df = pd.DataFrame({
            "pred_mean": pred_mean,
            "ci_lower": conf_int.iloc[:, 0],
            "ci_upper": conf_int.iloc[:, 1]
        }, index=pred_mean.index)
        return pred_df

    def get_coefficients_table(self) -> pd.DataFrame:
        """
        Retorna la tabla formal de coeficientes estimados, errores estándar,
        estadísticos z, p-values e intervalos de confianza.
        """
        if self.results is None:
            raise RuntimeError("El modelo debe ser ajustado antes de extraer coeficientes.")

        params = self.results.params
        bse = self.results.bse
        pvals = self.results.pvalues
        ci = self.results.conf_int()

        coef_df = pd.DataFrame({
            "Coeficiente": params.round(5),
            "Std_Error": bse.round(5),
            "z_stat": (params / bse).round(3),
            "p_value": pvals.round(4),
            "IC_Lower_95": ci.iloc[:, 0].round(5),
            "IC_Upper_95": ci.iloc[:, 1].round(5)
        })
        return coef_df


# =====================================================================
# 3. MODELO DE VECTORES AUTORREGRESIVOS (VAR)
# =====================================================================

@dataclass
class VARDiagnostics:
    optimal_lag: int
    is_stable: bool
    max_eigenvalue_modulus: float
    granger_causality_results: Dict[str, float]
    aic: float
    bic: float


class VARModelWrapper:
    """
    Wrapper econométrico para el modelo de Vectores Autorregresivos (VAR):
    Y_t = c + sum_{l=1}^p A_l Y_{t-l} + u_t
    donde Y_t = [gdp_growth, policy_rate_diff, inflation_diff, fx_depreciation, npl_diff]'
    """

    def __init__(self, target_col: str = "npl_diff", maxlags: int = 6):
        self.target_col = target_col
        self.maxlags = maxlags
        self.model: Optional[VAR] = None
        self.results: Optional[Any] = None
        self.optimal_lag: int = 1
        self.variables: List[str] = []

    def fit(self, data: pd.DataFrame, ic: str = "aic") -> VARModelWrapper:
        """
        Ajusta el sistema VAR seleccionando automáticamente el orden óptimo de rezagos
        mediante criterios de información multivariados (AIC, BIC, HQIC).
        """
        clean_data = data.dropna()
        self.variables = list(clean_data.columns)
        self.model = VAR(clean_data)

        # Selección de rezagos
        lag_order_res = self.model.select_order(maxlags=self.maxlags)
        selected_lag = getattr(lag_order_res, ic.lower(), None)
        
        # Salvaguarda: al menos 1 rezago y menor o igual a maxlags
        if selected_lag is None or selected_lag == 0:
            selected_lag = 1
        self.optimal_lag = int(min(max(selected_lag, 1), self.maxlags))

        logger.info(f"VAR: Rezago óptimo seleccionado p={self.optimal_lag} vía {ic.upper()}.")
        self.results = self.model.fit(self.optimal_lag)
        return self

    def check_dynamic_stability(self) -> Tuple[bool, float]:
        """
        Verifica la condición de estabilidad dinámica del sistema VAR:
        Todas las raíces del polinomio característico inverso deben yacer estrictamente
        DENTRO del círculo unitario (módulo de los valores propios de la matriz compañera < 1).
        """
        if self.results is None:
            raise RuntimeError("El modelo VAR debe ser ajustado primero.")

        is_stable = bool(self.results.is_stable(verbose=False))
        # Calcular módulos de los valores propios de la matriz compañera
        roots = self.results.roots
        # Invertir raíces si statsmodels define raíces del polinomio inverso
        moduli = np.abs(roots)
        max_modulus = float(np.max(1.0 / moduli)) if len(moduli) > 0 else 0.0

        logger.info(f"Estabilidad dinámica VAR: {'ESTABLE' if is_stable else 'INESTABLE'} (Módulo Máximo = {max_modulus:.4f} < 1.0)")
        return is_stable, max_modulus

    def test_granger_causality(self) -> Dict[str, float]:
        """
        Ejecuta pruebas de Causalidad de Granger para evaluar si cada variable macroeconómica
        mejora el pronóstico de la morosidad (NPL) más allá del pasado del propio NPL.
        """
        if self.results is None:
            raise RuntimeError("El modelo VAR debe ser ajustado primero.")

        gc_results = {}
        for col in self.variables:
            if col == self.target_col:
                continue
            try:
                test = self.results.test_causality(caused=self.target_col, causing=[col], kind="f")
                gc_results[col] = round(float(test.pvalue), 4)
            except Exception as e:
                logger.warning(f"No se pudo calcular Granger Causality para {col}: {e}")
                gc_results[col] = np.nan
        return gc_results

    def compute_irf(self, periods: int = 18, orth: bool = True) -> Any:
        """
        Calcula las Funciones de Impulso-Respuesta (IRF) ortogonalizadas mediante
        descomposición de Cholesky.
        """
        if self.results is None:
            raise RuntimeError("El modelo VAR debe ser ajustado primero.")
        return self.results.irf(periods=periods)

    def compute_fevd(self, periods: int = 18) -> Any:
        """
        Calcula la Descomposición de Varianza del Error de Predicción (FEVD).
        """
        if self.results is None:
            raise RuntimeError("El modelo VAR debe ser ajustado primero.")
        return self.results.fevd(periods=periods)

    def forecast(self, steps: int = 12) -> pd.DataFrame:
        """
        Genera pronósticos dinámicos out-of-sample del sistema VAR completo.
        """
        if self.results is None:
            raise RuntimeError("El modelo VAR debe ser ajustado primero.")

        last_obs = self.results.endog[-self.optimal_lag:]
        forecast_vals = self.results.forecast(y=last_obs, steps=steps)
        forecast_df = pd.DataFrame(forecast_vals, columns=self.variables)
        return forecast_df


# =====================================================================
# 4. BACKTESTING TEMPORAL CON VENTANA EXPANSIVA (ROLLING ORIGIN)
# =====================================================================

class TimeDimensionBacktester:
    """
    Evaluador out-of-sample mediante ventana temporal expansiva (Expanding Window Time Series Split).
    Evalúa SARIMAX vs VAR en múltiples orígenes históricos con un horizonte de prueba h=12 meses.
    """

    def __init__(
        self,
        min_train_periods: int = 84,  # Mínimo 7 años para entrenamiento inicial
        forecast_horizon: int = 12,   # Horizonte estándar de estrés = 1 año
        step_size: int = 6            # Desplazamiento cada semestre
    ):
        self.min_train = min_train_periods
        self.horizon = forecast_horizon
        self.step_size = step_size

    def evaluate_models(
        self,
        target_series: pd.Series,
        exog_df: pd.DataFrame,
        var_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Ejecuta el backtesting comparativo y reporta métricas de error y pérdida asimétrica.
        """
        n_obs = len(target_series)
        sarimax_metrics_list = []
        var_metrics_list = []
        origins = []

        logger.info(f"Iniciando Backtesting Temporal Expansivo (T={n_obs}, min_train={self.min_train}, h={self.horizon})...")

        for split_point in range(self.min_train, n_obs - self.horizon + 1, self.step_size):
            train_dates = target_series.index[:split_point]
            test_dates = target_series.index[split_point: split_point + self.horizon]
            origins.append(str(train_dates[-1].date()))

            # 1. Datos Train / Test
            y_train = target_series.iloc[:split_point]
            y_test = target_series.iloc[split_point: split_point + self.horizon]

            exog_train = exog_df.iloc[:split_point]
            exog_test = exog_df.iloc[split_point: split_point + self.horizon]

            var_train = var_data.iloc[:split_point]

            # 2. SARIMAX
            try:
                sarimax = SARIMAXModelWrapper(order=(1, 0, 1), seasonal_order=(1, 0, 0, 12))
                sarimax.fit(y_train, exog=exog_train)
                pred_sarimax = sarimax.predict(steps=len(y_test), exog_future=exog_test)["pred_mean"].values
                m_sarimax = evaluate_forecast_metrics(y_test.values, pred_sarimax)
                sarimax_metrics_list.append(m_sarimax)
            except Exception as e:
                logger.warning(f"Error evaluando SARIMAX en split {split_point}: {e}")

            # 3. VAR
            try:
                var_wrapper = VARModelWrapper(target_col=target_series.name, maxlags=3)
                var_wrapper.fit(var_train, ic="aic")
                var_fc = var_wrapper.forecast(steps=len(y_test))
                pred_var = var_fc[target_series.name].values
                m_var = evaluate_forecast_metrics(y_test.values, pred_var)
                var_metrics_list.append(m_var)
            except Exception as e:
                logger.warning(f"Error evaluando VAR en split {split_point}: {e}")

        # Consolidar promedios de métricas
        df_sarimax = pd.DataFrame(sarimax_metrics_list)
        df_var = pd.DataFrame(var_metrics_list)

        summary_rows = [
            {
                "Modelo": "SARIMAX (Multiplicativo + Regresores Rezagados)",
                "RMSE": round(df_sarimax["RMSE"].mean(), 4),
                "MAE": round(df_sarimax["MAE"].mean(), 4),
                "MAPE (%)": round(df_sarimax["MAPE_pct"].mean(), 2),
                "Pérdida Asimétrica (2x Subest.)": round(df_sarimax["Asymmetric_Loss_2x"].mean(), 4)
            },
            {
                "Modelo": "VAR (Vector Autorregresivo Multivariado)",
                "RMSE": round(df_var["RMSE"].mean(), 4),
                "MAE": round(df_var["MAE"].mean(), 4),
                "MAPE (%)": round(df_var["MAPE_pct"].mean(), 2),
                "Pérdida Asimétrica (2x Subest.)": round(df_var["Asymmetric_Loss_2x"].mean(), 4)
            }
        ]

        summary_table = pd.DataFrame(summary_rows)
        logger.info(f"Backtesting completado en {len(origins)} ventanas temporales.")
        return summary_table


if __name__ == "__main__":
    # Prueba de integración y verificación del módulo
    data_path = Path("data/processed/macro_npl_processed.csv")
    if not data_path.exists():
        print("Ejecuta primero src/features.py para generar data/processed/macro_npl_processed.csv")
    else:
        df = pd.read_csv(data_path, index_col="date", parse_dates=True)
        target = df["npl_diff"]
        exog_cols = [c for c in df.columns if c not in ["npl_diff", "npl_level"]]
        exog = df[exog_cols]

        print("\n" + "=" * 65)
        print("1. AJUSTE Y DIAGNÓSTICOS REGULATORIOS SARIMAX:")
        sarimax = SARIMAXModelWrapper(order=(1, 0, 1), seasonal_order=(1, 0, 0, 12))
        sarimax.fit(target, exog=exog)
        diag = sarimax.run_diagnostics()
        print(diag.summary_df.to_string(index=False))

        print("\n" + "=" * 65)
        print("TABLA DE COEFICIENTES ESTIMADOS (SARIMAX):")
        print(sarimax.get_coefficients_table())

        print("\n" + "=" * 65)
        print("2. AJUSTE Y DIAGNÓSTICOS DEL MODELO VAR:")
        var_data = df[["gdp_growth_yoy_lag3", "policy_rate_diff_lag8", "inflation_diff_lag9", "fx_depreciation_mom_lag12", "npl_diff"]]
        var_model = VARModelWrapper(target_col="npl_diff", maxlags=4)
        var_model.fit(var_data, ic="aic")
        is_stable, max_root = var_model.check_dynamic_stability()
        print(f"Estabilidad dinámica: {is_stable} (Max Eigenvalue Modulus: {max_root:.4f})")
        gc = var_model.test_granger_causality()
        print("Causalidad de Granger hacia Morosidad (p-values):", gc)

        print("\n" + "=" * 65)
        print("3. BACKTESTING TEMPORAL EXPANSIVO (HORIZONTE h=12 MESES):")
        backtester = TimeDimensionBacktester(min_train_periods=72, forecast_horizon=12, step_size=6)
        comp_df = backtester.evaluate_models(target, exog, var_data)
        print(comp_df.to_string(index=False))
