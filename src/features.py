"""
Módulo de Ingeniería Econométrica y Preprocesamiento (src/features.py)
====================================================================
Este módulo implementa el pipeline de preparación de datos bajo rigor econométrico:
1. StationarityTester: Pruebas conjuntas de Raíz Unitaria ADF (Dickey-Fuller Aumentado)
   y KPSS (Kwiatkowski-Phillips-Schmidt-Shin) con diagnóstico analítico I(d).
2. FinancialTransformations: Diferencias regulares (1 - B), estacionales (1 - B^12),
   variaciones logarítmicas (MoM / YoY) y transformaciones logit.
3. CCFLagSelector: Selección de rezagos óptimos mediante Función de Correlación Cruzada (CCF)
   con intervalos de confianza de Bartlett al 95% y validación de sentido económico.
4. MulticollinearityController: Cálculo y depuración de multicolinealidad vía Variance
   Inflation Factor (VIF < 5.0).
5. FeatureEngineeringPipeline: Orquestador integral de procesamiento y exportación.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, ccf, kpss

# Suprimir advertencias menores de statsmodels para outputs limpios
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =====================================================================
# 1. PRUEBAS DE RAÍZ UNITARIA (ADF + KPSS)
# =====================================================================

@dataclass
class StationarityResult:
    variable: str
    order_of_diff: int
    adf_stat: float
    adf_pvalue: float
    adf_lags: int
    adf_critical_5pct: float
    kpss_stat: float
    kpss_pvalue: float
    kpss_lags: int
    kpss_critical_5pct: float
    conclusion: str
    is_stationary: bool


class StationarityTester:
    """
    Evaluador conjunto de estacionariedad para series de tiempo financieras.
    Combina:
    - Augmented Dickey-Fuller (ADF):
        H0: La serie posee raíz unitaria (No Estacionaria).
        H1: La serie es estacionaria.
    - KPSS Test:
        H0: La serie es estacionaria en torno a nivel o tendencia.
        H1: La serie posee raíz unitaria (No Estacionaria).
    """

    @staticmethod
    def test_single_series(
        series: pd.Series,
        regression: str = "c",
        alpha: float = 0.05,
        d: int = 0
    ) -> StationarityResult:
        """
        Ejecuta ADF y KPSS sobre una serie unidimensional.
        """
        clean_s = series.dropna()
        if len(clean_s) < 15:
            raise ValueError(f"Longitud de serie insuficiente ({len(clean_s)}) para tests econométricos.")

        # 1. ADF Test
        adf_res = adfuller(clean_s, regression=regression, autolag="AIC")
        adf_stat = float(adf_res[0])
        adf_pval = float(adf_res[1])
        adf_usedlag = int(adf_res[2])
        adf_crit_5 = float(adf_res[4]["5%"])

        # 2. KPSS Test
        kpss_res = kpss(clean_s, regression=regression, nlags="auto")
        kpss_stat = float(kpss_res[0])
        kpss_pval = float(kpss_res[1])
        kpss_usedlag = int(kpss_res[2])
        kpss_crit_5 = float(kpss_res[3]["5%"])

        # 3. Diagnóstico conjunto de la matriz econométrica
        adf_rejects_h0 = adf_pval < alpha          # Rechaza raíz unitaria -> Estacionaria según ADF
        kpss_rejects_h0 = kpss_pval < alpha        # Rechaza estacionariedad -> No estacionaria según KPSS

        if adf_rejects_h0 and not kpss_rejects_h0:
            conclusion = "Estacionaria I(0) [ADF rechaza H0, KPSS no rechaza H0]"
            is_stat = True
        elif not adf_rejects_h0 and kpss_rejects_h0:
            conclusion = f"No estacionaria / Raíz Unitaria I({d + 1}) [ADF no rechaza H0, KPSS rechaza H0]"
            is_stat = False
        elif adf_rejects_h0 and kpss_rejects_h0:
            conclusion = "Estacionaria con quiebre estructural / tendencia determinística"
            is_stat = True  # Para fines prácticos de modelado con constante
        else:
            conclusion = "Inconcluso / Muestra pequeña (ambos fallan en rechazar H0)"
            is_stat = False

        return StationarityResult(
            variable=series.name or "series",
            order_of_diff=d,
            adf_stat=round(adf_stat, 4),
            adf_pvalue=round(adf_pval, 4),
            adf_lags=adf_usedlag,
            adf_critical_5pct=round(adf_crit_5, 4),
            kpss_stat=round(kpss_stat, 4),
            kpss_pvalue=round(kpss_pval, 4),
            kpss_lags=kpss_usedlag,
            kpss_critical_5pct=round(kpss_crit_5, 4),
            conclusion=conclusion,
            is_stationary=is_stat
        )

    @classmethod
    def determine_integration_order(
        cls,
        series: pd.Series,
        max_d: int = 2,
        alpha: float = 0.05
    ) -> Tuple[int, pd.Series, List[StationarityResult]]:
        """
        Determina de forma iterativa el orden exacto de integración I(d) de la serie.
        Diferencia sucesivamente (1 - B)^d hasta lograr estacionariedad.
        """
        current_s = series.copy()
        history: List[StationarityResult] = []

        for d in range(max_d + 1):
            res = cls.test_single_series(current_s, alpha=alpha, d=d)
            history.append(res)
            if res.is_stationary:
                return d, current_s, history
            # Aplicar primera diferencia regular
            current_s = current_s.diff().dropna()
            current_s.name = f"{series.name}_d{d + 1}"

        # Si tras max_d no es estacionaria, retornar último orden
        return max_d, current_s, history

    @classmethod
    def analyze_panel(cls, df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
        """
        Ejecuta el análisis de estacionariedad para todas las columnas de un DataFrame.
        """
        records = []
        for col in df.columns:
            res = cls.test_single_series(df[col], alpha=alpha, d=0)
            records.append({
                "Variable": col,
                "ADF Stat": res.adf_stat,
                "ADF p-value": res.adf_pvalue,
                "KPSS Stat": res.kpss_stat,
                "KPSS p-value": res.kpss_pvalue,
                "Estacionaria": "Sí" if res.is_stationary else "No",
                "Diagnóstico": res.conclusion
            })
        return pd.DataFrame(records)


# =====================================================================
# 2. TRANSFORMACIONES FINANCIERAS Y ECONOMÉTRICAS
# =====================================================================

class FinancialTransformations:
    """
    Catálogo de transformaciones estandarizadas para variables macrofinancieras.
    """

    @staticmethod
    def diff(series: pd.Series, order: int = 1) -> pd.Series:
        """Diferencia regular: (1 - B)^order * Y_t"""
        s = series.copy()
        for _ in range(order):
            s = s.diff()
        return s

    @staticmethod
    def seasonal_diff(series: pd.Series, s: int = 12) -> pd.Series:
        """Diferencia estacional anual: (1 - B^s) * Y_t"""
        return series.diff(periods=s)

    @staticmethod
    def log_diff_mom(series: pd.Series) -> pd.Series:
        """Variación porcentual intermensual logarítmica: Delta ln(X_t) * 100"""
        return (np.log(series) - np.log(series.shift(1))) * 100.0

    @staticmethod
    def log_diff_yoy(series: pd.Series, s: int = 12) -> pd.Series:
        """Variación porcentual interanual logarítmica: Delta_12 ln(X_t) * 100"""
        return (np.log(series) - np.log(series.shift(s))) * 100.0

    @staticmethod
    def logit_transform(series: pd.Series, max_val: float = 100.0) -> pd.Series:
        """
        Transformación Logit para tasas acotadas en [0, 1] o [0, 100]:
        logit(p) = ln(p / (1 - p))
        Evita proyecciones de morosidad negativas o superiores al 100%.
        """
        p = np.clip(series / max_val, 1e-4, 1.0 - 1e-4)
        return np.log(p / (1.0 - p))

    @staticmethod
    def inverse_logit(series: pd.Series, max_val: float = 100.0) -> pd.Series:
        """Transformación inversa de Logit: p = 1 / (1 + exp(-x))"""
        return (1.0 / (1.0 + np.exp(-series))) * max_val

    @classmethod
    def apply_standard_macro_transformations(cls, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica las transformaciones financieramente óptimas para el sistema bancario peruano:
        - npl_diff: Delta NPL_t (cambio mensual en puntos porcentuales de la morosidad).
        - gdp_growth_yoy: Delta_12 ln(PBI_t) * 100 (crecimiento económico interanual %).
        - policy_rate_diff: Delta TasaRef_t (shock mensual de política monetaria en p.p.).
        - inflation_diff: Delta Inflacion_12m_t (aceleración/desaceleración de la inflación).
        - fx_depreciation_mom: Delta ln(TC_t) * 100 (depreciación mensual del Sol %).
        """
        df = pd.DataFrame(index=raw_df.index)

        # 1. Target: Morosidad de Consumo
        if "npl_consumer" in raw_df.columns:
            df["npl_diff"] = raw_df["npl_consumer"].diff()
            df["npl_level"] = raw_df["npl_consumer"]  # Mantenemos nivel para referencia
            df["npl_logit"] = cls.logit_transform(raw_df["npl_consumer"])
            df["d_npl_logit"] = df["npl_logit"].diff()

        # 2. PBI Global (Índice de producción)
        if "gdp_index" in raw_df.columns:
            df["gdp_growth_yoy"] = cls.log_diff_yoy(raw_df["gdp_index"], s=12)
            df["gdp_growth_mom"] = cls.log_diff_mom(raw_df["gdp_index"])

        # 3. Tasa de Política Monetaria
        if "policy_rate" in raw_df.columns:
            df["policy_rate_diff"] = raw_df["policy_rate"].diff()
            df["policy_rate_level"] = raw_df["policy_rate"]

        # 4. Inflación 12 meses
        if "inflation_12m" in raw_df.columns:
            df["inflation_12m"] = raw_df["inflation_12m"]
            df["inflation_diff"] = raw_df["inflation_12m"].diff()

        # 5. Tipo de Cambio
        if "fx_rate" in raw_df.columns:
            df["fx_depreciation_mom"] = cls.log_diff_mom(raw_df["fx_rate"])
            df["fx_depreciation_yoy"] = cls.log_diff_yoy(raw_df["fx_rate"], s=12)

        return df.dropna()


# =====================================================================
# 3. SELECCIÓN DE REZAGOS VÍA CORRELACIÓN CRUZADA (CCF)
# =====================================================================

@dataclass
class CCFSelectionResult:
    variable: str
    optimal_lag: int
    correlation: float
    is_significant: bool
    expected_sign_satisfied: bool
    summary_df: pd.DataFrame


class CCFLagSelector:
    """
    Selector econométrico de rezagos óptimos basado en la Función de Correlación Cruzada (CCF).
    Calcula Corr(Y_t, X_{t-k}) para k in {1, 2, ..., max_lags}.
    Aplica intervalos de confianza de Bartlett: +- 1.96 / sqrt(T) al 95% de significancia.
    """

    # Signos económicos esperados sobre el incremento de Morosidad (Delta NPL_t)
    EXPECTED_SIGNS = {
        "policy_rate_diff": "+",    # Alza de tasa eleva cuotas de crédito -> mayor morosidad
        "policy_rate_level": "+",   # Niveles altos de tasa asfixian capacidad de pago
        "gdp_growth_yoy": "-",      # Crecimiento económico mejora ingresos -> menor morosidad
        "gdp_growth_mom": "-",      # Crecimiento mensual positivo reduce morosidad
        "inflation_12m": "+",       # Mayor inflación deteriora poder adquisitivo -> mayor morosidad
        "inflation_diff": "+",      # Aceleración inflacionaria eleva morosidad
        "fx_depreciation_mom": "+", # Depreciación encarece deudas en US$ e importaciones
        "fx_depreciation_yoy": "+"  # Depreciación acumulada eleva defaults
    }

    @classmethod
    def compute_ccf_profile(
        cls,
        target_series: pd.Series,
        exog_series: pd.Series,
        max_lags: int = 12
    ) -> pd.DataFrame:
        """
        Calcula la correlación cruzada entre target Y_t y exog X_{t-k} para k in 1..max_lags.
        """
        aligned = pd.concat([target_series, exog_series], axis=1).dropna()
        y = aligned.iloc[:, 0].values
        x = aligned.iloc[:, 1].values
        t_len = len(y)
        bartlett_bound = 1.96 / np.sqrt(t_len)

        # En statsmodels ccf(y, x)[k] evalúa Corr(y_t, x_{t-k}) para k >= 0
        raw_ccf = ccf(y, x, adjusted=False)

        records = []
        for k in range(1, min(max_lags + 1, len(raw_ccf))):
            corr = float(raw_ccf[k])
            sig = abs(corr) >= bartlett_bound
            records.append({
                "Lag_k": k,
                "Correlation": round(corr, 4),
                "Bartlett_CI": round(bartlett_bound, 4),
                "Significant_95pct": sig
            })

        return pd.DataFrame(records)

    @classmethod
    def select_optimal_lag(
        cls,
        target_series: pd.Series,
        exog_series: pd.Series,
        expected_sign: Optional[str] = None,
        max_lags: int = 12
    ) -> CCFSelectionResult:
        """
        Selecciona el rezago óptimo k* que maximice la correlación absoluta respetando
        la significancia estadística y el sentido económico esperado.
        """
        var_name = exog_series.name or "exog"
        sign_rule = expected_sign or cls.EXPECTED_SIGNS.get(var_name, None)

        ccf_df = cls.compute_ccf_profile(target_series, exog_series, max_lags=max_lags)

        # Filtrar candidatos
        candidates = ccf_df.copy()
        if sign_rule == "+":
            candidates["sign_ok"] = candidates["Correlation"] > 0
        elif sign_rule == "-":
            candidates["sign_ok"] = candidates["Correlation"] < 0
        else:
            candidates["sign_ok"] = True

        valid_pool = candidates[candidates["Significant_95pct"] & candidates["sign_ok"]]

        if not valid_pool.empty:
            # Seleccionar rezago con máxima correlación absoluta dentro de los válidos
            best_idx = valid_pool["Correlation"].abs().idxmax()
            best_row = valid_pool.loc[best_idx]
            opt_lag = int(best_row["Lag_k"])
            opt_corr = float(best_row["Correlation"])
            is_sig = True
            sign_ok = True
        else:
            # Fallback: Mejor rezago empírico según signo (aunque esté al límite de significancia)
            sign_pool = candidates[candidates["sign_ok"]]
            if not sign_pool.empty:
                best_idx = sign_pool["Correlation"].abs().idxmax()
                best_row = sign_pool.loc[best_idx]
                opt_lag = int(best_row["Lag_k"])
                opt_corr = float(best_row["Correlation"])
                is_sig = bool(best_row["Significant_95pct"])
                sign_ok = True
            else:
                # Fallback absoluto: Máxima correlación general
                best_idx = candidates["Correlation"].abs().idxmax()
                best_row = candidates.loc[best_idx]
                opt_lag = int(best_row["Lag_k"])
                opt_corr = float(best_row["Correlation"])
                is_sig = bool(best_row["Significant_95pct"])
                sign_ok = False

        return CCFSelectionResult(
            variable=var_name,
            optimal_lag=opt_lag,
            correlation=opt_corr,
            is_significant=is_sig,
            expected_sign_satisfied=sign_ok,
            summary_df=ccf_df
        )

    @classmethod
    def build_optimal_lag_matrix(
        cls,
        target_series: pd.Series,
        exog_df: pd.DataFrame,
        max_lags: int = 12
    ) -> Tuple[pd.DataFrame, Dict[str, CCFSelectionResult]]:
        """
        Genera la matriz de diseño X_{t-k*} desplazando cada variable macroeconómica
        según su rezago óptimo seleccionado.
        """
        lagged_cols = {}
        results = {}

        for col in exog_df.columns:
            sel = cls.select_optimal_lag(target_series, exog_df[col], max_lags=max_lags)
            results[col] = sel
            k = sel.optimal_lag
            col_lag_name = f"{col}_lag{k}"
            lagged_cols[col_lag_name] = exog_df[col].shift(k)

        lag_df = pd.DataFrame(lagged_cols, index=exog_df.index)
        return lag_df, results


# =====================================================================
# 4. CONTROL DE MULTICOLINEALIDAD (VIF)
# =====================================================================

class MulticollinearityController:
    """
    Controlador y diagnosticador de multicolinealidad vía Variance Inflation Factor (VIF).
    Regla estándar bancaria / regulatoria:
    - VIF < 5.0: Multicolinealidad baja/tolerable.
    - 5.0 <= VIF < 10.0: Multicolinealidad moderada (requiere atención).
    - VIF >= 10.0: Multicolinealidad severa (debe eliminarse o regularizarse).
    """

    @staticmethod
    def calculate_vif(X: pd.DataFrame, add_constant: bool = True) -> pd.Series:
        """
        Calcula el VIF de cada columna en la matriz de diseño X.
        """
        clean_X = X.dropna()
        if add_constant:
            clean_X = sm.add_constant(clean_X)

        vif_data = []
        for i in range(clean_X.shape[1]):
            v = variance_inflation_factor(clean_X.values, i)
            vif_data.append(float(v))

        s = pd.Series(vif_data, index=clean_X.columns, name="VIF")
        if add_constant and "const" in s:
            s = s.drop("const")
        return s.round(3)

    @classmethod
    def prune_features_by_vif(
        cls,
        X: pd.DataFrame,
        threshold: float = 5.0,
        protected_features: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Elimina secuencialmente la variable con mayor VIF que exceda el umbral,
        recalculando iterativamente hasta que todas cumplan VIF < threshold.
        """
        protected = protected_features or []
        current_X = X.dropna().copy()

        while True:
            if current_X.shape[1] <= 1:
                break
            vifs = cls.calculate_vif(current_X, add_constant=True)
            max_vif = vifs.max()
            if max_vif <= threshold:
                break

            # Buscar la variable no protegida con mayor VIF
            droppable = [c for c in vifs.sort_values(ascending=False).index if c not in protected]
            if not droppable:
                logger.warning("Todas las variables restantes con VIF alto están en protected_features. Deteniendo poda.")
                break

            drop_var = droppable[0]
            logger.info(f"Podando variable '{drop_var}' con VIF={vifs[drop_var]:.2f} (umbral={threshold})")
            current_X = current_X.drop(columns=[drop_var])

        final_vifs = cls.calculate_vif(current_X, add_constant=True)
        return current_X, final_vifs


# =====================================================================
# 5. PIPELINE INTEGRAL DE INGENIERÍA ECONOMÉTRICA
# =====================================================================

class FeatureEngineeringPipeline:
    """
    Pipeline orquestador para transformación de datos crudos en variables estacionarias,
    alineadas con rezagos óptimos y validadas contra multicolinealidad.
    """

    def __init__(
        self,
        raw_path: Union[str, Path] = "data/raw/macro_npl_raw.csv",
        processed_dir: Union[str, Path] = "data/processed"
    ):
        self.raw_path = Path(raw_path)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        vif_threshold: float = 5.0,
        max_lags: int = 12,
        save_artifacts: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Ejecuta el pipeline econométrico integral.

        Returns:
            processed_df: DataFrame con series estacionarias y rezagos óptimos.
            metadata: Diccionario con tests de estacionariedad, rezagos y VIF.
        """
        logger.info(">>> Iniciando Pipeline de Ingeniería Econométrica (src/features.py)...")

        if not self.raw_path.exists():
            raise FileNotFoundError(f"No se encontró dataset crudo en: {self.raw_path}. Ejecuta src/ingestion.py primero.")

        raw_df = pd.read_csv(self.raw_path, index_col="date", parse_dates=True)
        logger.info(f"Cargado dataset crudo: {raw_df.shape[0]} observaciones x {raw_df.shape[1]} columnas.")

        # 1. Pruebas de raíz unitaria en series crudas
        logger.info("Paso 1: Evaluando tests conjuntos ADF + KPSS en series crudas...")
        raw_stationarity_df = StationarityTester.analyze_panel(raw_df)

        # 2. Aplicar transformaciones financieras estandarizadas
        logger.info("Paso 2: Aplicando transformaciones financieras (diferencias y variaciones logarítmicas)...")
        transformed_df = FinancialTransformations.apply_standard_macro_transformations(raw_df)

        # 3. Pruebas de raíz unitaria en series transformadas
        logger.info("Paso 3: Verificando estacionariedad post-transformación...")
        stat_eval_df = StationarityTester.analyze_panel(transformed_df)

        # 4. Selección de variables clave y correlación cruzada con el target
        target_var = "npl_diff"
        candidate_exog_cols = [
            "gdp_growth_yoy",
            "policy_rate_diff",
            "inflation_diff",
            "fx_depreciation_mom"
        ]
        available_exog = [c for c in candidate_exog_cols if c in transformed_df.columns]

        logger.info(f"Paso 4: Calculando Correlación Cruzada (CCF) de {available_exog} contra {target_var}...")
        lag_matrix, ccf_results = CCFLagSelector.build_optimal_lag_matrix(
            target_series=transformed_df[target_var],
            exog_df=transformed_df[available_exog],
            max_lags=max_lags
        )

        # 5. Control de Multicolinealidad (VIF)
        logger.info(f"Paso 5: Evaluando Variance Inflation Factor (VIF < {vif_threshold})...")
        clean_lag_matrix = lag_matrix.dropna()
        final_X, final_vifs = MulticollinearityController.prune_features_by_vif(
            clean_lag_matrix,
            threshold=vif_threshold,
            protected_features=[]
        )

        # 6. Consolidación de dataset procesado
        # Incluimos: Target (npl_diff, npl_level), exógenas rezagadas seleccionadas
        processed_df = pd.concat([
            transformed_df[["npl_diff", "npl_level"]],
            final_X
        ], axis=1).dropna()
        processed_df.index.name = "date"

        # 7. Metadata de auditoría econométrica
        metadata = {
            "sample_start": str(processed_df.index.min().date()),
            "sample_end": str(processed_df.index.max().date()),
            "total_observations": int(len(processed_df)),
            "target_variable": target_var,
            "raw_stationarity_tests": raw_stationarity_df.to_dict(orient="records"),
            "transformed_stationarity_tests": stat_eval_df.to_dict(orient="records"),
            "selected_lags": {
                k: {
                    "optimal_lag": v.optimal_lag,
                    "correlation": v.correlation,
                    "is_significant": v.is_significant,
                    "sign_ok": v.expected_sign_satisfied
                }
                for k, v in ccf_results.items()
            },
            "final_vifs": final_vifs.to_dict(),
            "final_features": list(final_X.columns)
        }

        if save_artifacts:
            processed_file = self.processed_dir / "macro_npl_processed.csv"
            processed_df.to_csv(processed_file)
            meta_file = self.processed_dir / "feature_metadata.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Dataset procesado exportado a: {processed_file}")
            logger.info(f"Metadata econométrica exportada a: {meta_file}")

        return processed_df, metadata


if __name__ == "__main__":
    pipeline = FeatureEngineeringPipeline()
    df_proc, meta = pipeline.run(vif_threshold=5.0, max_lags=12)
    print("\n" + "=" * 60)
    print("RESUMEN DE REZAGOS ÓPTIMOS (CCF):")
    for var, info in meta["selected_lags"].items():
        print(f"- {var}: Lag k* = {info['optimal_lag']} meses | Corr = {info['correlation']:+.4f} | Sig: {info['is_significant']} | Signo: {info['sign_ok']}")
    print("\n" + "=" * 60)
    print("VIF DE MATRIZ DE DISEÑO:")
    for col, v in meta["final_vifs"].items():
        print(f"- {col}: VIF = {v:.3f}")
    print("\nDataset Procesado Final (head):")
    print(df_proc.head())
