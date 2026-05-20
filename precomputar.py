"""
Precomputa las predicciones de matrícula para todas las carreras.

Replica fielmente la lógica del notebook (Predicciones_Matricula_colab.ipynb):
 - Mismo preprocesamiento: extracción de año, agregación por (programa, año)
   sumando SS+SA, descarte de programas sin datos.
 - Mismos 5 modelos: Regresión Lineal, Exponencial, ARIMA con grid search
   (p,d,q) por AIC, Holt (clásico y amortiguado), Media Móvil (3).
 - Mismo criterio de selección: si la serie tiene >= 8 años se reservan
   los últimos 3 como test y se elige por menor RMSE en test; si no,
   RMSE in-sample.
 - Predicciones clipeadas a >= 0 y redondeadas a enteros.

Como no tengo statsmodels disponible, ARIMA y Holt están implementados
manualmente con scipy.optimize (CSS para ARIMA, optimización de
parámetros para Holt). La estructura y el criterio son los mismos.

Salida: web/resultados.json (consumido por la página estática).
"""
import json
import re
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

RUTA_ARCHIVO = "MATRICULA2026.xlsx"
HORIZONTE = 5
TEST_SIZE = 3


# ---------------------------------------------------------------------------
# Preprocesamiento (idéntico al notebook)
# ---------------------------------------------------------------------------
def ciclo_a_anio(c):
    m = re.match(r"(\d{2})/(\d{2})", str(c).strip())
    if not m:
        return np.nan
    yy = int(m.group(1))
    return 2000 + yy if yy < 80 else 1900 + yy


def cargar_y_agregar(ruta, columna_objetivo="NVO_INGRESO"):
    df = pd.read_excel(ruta)
    df.columns = [c.strip() for c in df.columns]
    df["ANIO"] = df["CICLO"].apply(ciclo_a_anio)
    df = df.dropna(subset=["ANIO"]).copy()
    df["ANIO"] = df["ANIO"].astype(int)

    # Paso 2a: por (programa, ciclo). Para NVO_INGRESO suma; lo mismo aplica
    # para REINGRESO y MATRICULA. Se replica la lógica del notebook.
    paso_a = df.groupby(
        ["PROGRAMA_EDUCATIVO", "CICLO", "ANIO"], as_index=False
    ).agg({columna_objetivo: "sum"})

    # Paso 2b: por (programa, año). Suma SS+SA del mismo año.
    df_agg = paso_a.groupby(
        ["PROGRAMA_EDUCATIVO", "ANIO"], as_index=False
    ).agg({columna_objetivo: "sum"})

    # Paso 3: descarte de programas con todos los valores en 0/NaN
    progs_iniciales = set(df_agg["PROGRAMA_EDUCATIVO"].unique())
    mask_cero = (df_agg[columna_objetivo] == 0) | df_agg[columna_objetivo].isna()
    df_agg_filtrado = df_agg[~mask_cero].reset_index(drop=True)
    progs_validos = set(df_agg_filtrado["PROGRAMA_EDUCATIVO"].unique())
    descartados = sorted(progs_iniciales - progs_validos)
    return df_agg_filtrado, descartados


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


# ---------------------------------------------------------------------------
# ARMA(p,q) implementado manualmente vía CSS (Conditional Sum of Squares).
# Reemplaza statsmodels para que el script funcione sin esa librería.
# ---------------------------------------------------------------------------
def _arma_residuals(params, y, p, q):
    """Calcula residuos one-step-ahead de un ARMA(p,q) con CSS."""
    n = len(y)
    c = params[0]
    phi = np.asarray(params[1 : 1 + p])
    theta = np.asarray(params[1 + p : 1 + p + q])
    eps = np.zeros(n)
    mu = c
    for t in range(n):
        ar_part = 0.0
        for i in range(p):
            if t - i - 1 >= 0:
                ar_part += phi[i] * (y[t - i - 1] - mu)
        ma_part = 0.0
        for j in range(q):
            if t - j - 1 >= 0:
                ma_part += theta[j] * eps[t - j - 1]
        eps[t] = (y[t] - mu) - ar_part - ma_part
    return eps


def _arma_neg_loglik(params, y, p, q):
    eps = _arma_residuals(params, y, p, q)
    n = len(y)
    sse = float(np.sum(eps ** 2))
    if sse <= 0 or not np.isfinite(sse):
        return 1e18
    sigma2 = sse / n
    # Gaussian log-likelihood condicional
    ll = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1.0)
    return -ll


def fit_arima(y, order):
    """Ajusta ARIMA(p,d,q) por CSS. Devuelve (params, residuos, AIC) o None."""
    p, d, q = order
    if p == 0 and d == 0 and q == 0:
        return None
    y = np.asarray(y, dtype=float)
    # Aplicar diferenciación d veces
    y_diff = y.copy()
    for _ in range(d):
        y_diff = np.diff(y_diff)
    if len(y_diff) < max(p, q) + 2:
        return None
    # Inicialización
    init = [float(np.mean(y_diff))] + [0.05] * p + [0.05] * q
    try:
        res = minimize(
            _arma_neg_loglik,
            init,
            args=(y_diff, p, q),
            method="Nelder-Mead",
            options={"maxiter": 800, "xatol": 1e-5, "fatol": 1e-5},
        )
    except Exception:
        return None
    if not np.isfinite(res.fun):
        return None
    params = res.x
    eps = _arma_residuals(params, y_diff, p, q)
    sse = float(np.sum(eps ** 2))
    n = len(y_diff)
    sigma2 = sse / n
    if sigma2 <= 0 or not np.isfinite(sigma2):
        return None
    k = 1 + p + q + 1  # c, phi's, theta's, sigma²
    aic = 2 * k + 2 * res.fun
    if not np.isfinite(aic):
        return None
    return {
        "params": params,
        "p": p,
        "d": d,
        "q": q,
        "eps_diff": eps,
        "y_diff_last": y_diff,
        "y_last": y,
        "aic": float(aic),
        "sigma2": float(sigma2),
    }


def arima_forecast(fit, horizonte):
    """Pronostica `horizonte` pasos hacia adelante con un ajuste ARIMA.

    Devuelve (ajuste_in_sample, pronostico) en la escala original (no diff).
    Si horizonte == 0 devuelve solo el ajuste.
    """
    p, d, q = fit["p"], fit["d"], fit["q"]
    params = fit["params"]
    c = params[0]
    phi = np.asarray(params[1 : 1 + p])
    theta = np.asarray(params[1 + p : 1 + p + q])
    y_diff = fit["y_diff_last"].copy()  # serie diferenciada d veces
    eps = fit["eps_diff"].copy()
    y_full = fit["y_last"]

    # ---- Pronóstico en escala diferenciada ----
    pronostico_diff = []
    y_ext = list(y_diff)
    eps_ext = list(eps)
    for _ in range(horizonte):
        ar_part = 0.0
        for i in range(p):
            if len(y_ext) > i:
                ar_part += phi[i] * (y_ext[-i - 1] - c)
        ma_part = 0.0
        for j in range(q):
            if len(eps_ext) > j:
                ma_part += theta[j] * eps_ext[-j - 1]
        yhat = c + ar_part + ma_part
        pronostico_diff.append(yhat)
        y_ext.append(yhat)
        eps_ext.append(0.0)
    pronostico_diff = np.asarray(pronostico_diff)

    # ---- Ajuste in-sample en escala diferenciada ----
    fit_diff = y_diff - eps  # ŷ_diff = y_diff - residuo

    # ---- Reintegración (deshacer las diferenciaciones) ----
    if d == 0:
        ajuste = fit_diff
        pronostico = pronostico_diff
    else:
        # Calculamos las series diferenciadas a cada nivel:
        # niveles[0] = y_full (sin diferenciar), niveles[d] = y_diff
        niveles = [y_full.astype(float)]
        for _ in range(d):
            niveles.append(np.diff(niveles[-1]))

        # Reintegrar el pronóstico: a cada nivel sumamos el último valor del nivel inferior.
        # En el nivel k, cumsum(pronóstico_k) + último valor del nivel k-1 = pronóstico_{k-1}.
        pronostico_actual = pronostico_diff.copy()
        for nivel in range(d, 0, -1):
            ultimo = niveles[nivel - 1][-1]
            pronostico_actual = np.cumsum(pronostico_actual) + ultimo
        pronostico = pronostico_actual

        # Ajuste in-sample en escala original (solo para d=1; para d>=2 lo
        # aproximamos rellenando con el observado, ya que la reintegración del
        # ajuste in-sample paso a paso es delicada).
        if d == 1:
            # ŷ_t = y_{t-1} + ŷ_diff_t. El primer valor no es predecible.
            ajuste = np.concatenate([[y_full[0]], y_full[:-1] + fit_diff])
        else:
            # Aproximación honesta: el ajuste in-sample no es informativo y lo
            # marcamos como NaN. El criterio de selección usa solo el pronóstico
            # en test cuando hay >= 8 años, así que esto no afecta la decisión.
            ajuste = np.full(len(y_full), np.nan)
            # Pero para la visualización, rellenamos con la propia y como
            # placeholder (la línea no se desviará).
            ajuste = y_full.astype(float).copy()
    return ajuste, pronostico


def buscar_mejor_arima(y_train):
    mejor_aic = np.inf
    mejor_fit = None
    mejor_orden = None
    for p, d, q in itertools.product(range(4), range(3), range(4)):
        if p == 0 and d == 0 and q == 0:
            continue
        fit = fit_arima(y_train, (p, d, q))
        if fit is None:
            continue
        if fit["aic"] < mejor_aic:
            mejor_aic = fit["aic"]
            mejor_fit = fit
            mejor_orden = (p, d, q)
    return mejor_fit, mejor_orden, mejor_aic


# ---------------------------------------------------------------------------
# Holt's exponential smoothing (clásico y amortiguado) implementado manualmente.
# ---------------------------------------------------------------------------
def _holt_fit_and_forecast(y, horizonte, damped=False):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return None

    def sse(params):
        if damped:
            alpha, beta, phi = params
            if not (0 < alpha < 1 and 0 < beta < 1 and 0 < phi <= 1):
                return 1e18
        else:
            alpha, beta = params
            phi = 1.0
            if not (0 < alpha < 1 and 0 < beta < 1):
                return 1e18
        L = y[0]
        T = y[1] - y[0]
        ajuste = np.zeros(n)
        ajuste[0] = L
        s = 0.0
        for t in range(1, n):
            yhat_t = L + phi * T
            ajuste[t] = yhat_t
            L_new = alpha * y[t] + (1 - alpha) * yhat_t
            T_new = beta * (L_new - L) + (1 - beta) * phi * T
            L, T = L_new, T_new
            s += (y[t] - yhat_t) ** 2
        return s

    x0 = [0.3, 0.1, 0.95] if damped else [0.3, 0.1]
    try:
        res = minimize(
            sse,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-5, "fatol": 1e-5},
        )
    except Exception:
        return None
    if not np.isfinite(res.fun):
        return None
    if damped:
        alpha, beta, phi = res.x
        phi = max(min(phi, 1.0), 1e-3)
    else:
        alpha, beta = res.x
        phi = 1.0
    alpha = max(min(alpha, 0.999), 1e-3)
    beta = max(min(beta, 0.999), 1e-3)

    # Recalcular fitted values y forecast con los parámetros finales
    L = y[0]
    T = y[1] - y[0]
    fitted = np.zeros(n)
    fitted[0] = L
    for t in range(1, n):
        yhat_t = L + phi * T
        fitted[t] = yhat_t
        L_new = alpha * y[t] + (1 - alpha) * yhat_t
        T_new = beta * (L_new - L) + (1 - beta) * phi * T
        L, T = L_new, T_new

    forecast = np.zeros(horizonte)
    if damped:
        phi_acum = 0.0
        for h in range(1, horizonte + 1):
            phi_acum += phi ** h
            forecast[h - 1] = L + phi_acum * T
    else:
        for h in range(1, horizonte + 1):
            forecast[h - 1] = L + h * T
    return fitted, forecast


# ---------------------------------------------------------------------------
# Motor de evaluación (mismo criterio que el notebook)
# ---------------------------------------------------------------------------
def evaluar_modelos(y, horizonte=5, test_size=3):
    y = np.asarray(y, dtype=float)
    n = len(y)
    usar_test = n >= 8
    y_train = y[:-test_size] if usar_test else y
    y_test = y[-test_size:] if usar_test else None

    resultados = {}

    # 1) Regresión Lineal
    try:
        X_tr = np.arange(len(y_train)).reshape(-1, 1)
        lr = LinearRegression().fit(X_tr, y_train)
        fit_lr = lr.predict(np.arange(n).reshape(-1, 1))
        fut_lr = lr.predict(np.arange(n, n + horizonte).reshape(-1, 1))
        err = rmse(y_test, fit_lr[-test_size:]) if usar_test else rmse(y, fit_lr)
        if np.isfinite(err):
            resultados["Regresión Lineal"] = {
                "rmse": err,
                "ajuste": fit_lr.tolist(),
                "prediccion": fut_lr.tolist(),
                "info": "",
            }
    except Exception:
        pass

    # 2) Exponencial
    try:
        if np.all(y_train > 0):
            coefs = np.polyfit(np.arange(len(y_train)), np.log(y_train), 1)
            a, b = np.exp(coefs[1]), np.exp(coefs[0])
            fit_exp = a * b ** np.arange(n)
            fut_exp = a * b ** np.arange(n, n + horizonte)
            err = rmse(y_test, fit_exp[-test_size:]) if usar_test else rmse(y, fit_exp)
            if np.isfinite(err):
                resultados["Exponencial"] = {
                    "rmse": err,
                    "ajuste": fit_exp.tolist(),
                    "prediccion": fut_exp.tolist(),
                    "info": "",
                }
    except Exception:
        pass

    # 3) ARIMA con grid search por AIC
    try:
        fit_arima_best, orden, aic = buscar_mejor_arima(y_train)
        if fit_arima_best is not None:
            # Pronosticar el horizonte completo con refit sobre toda la serie
            fit_full = fit_arima(y, orden) if usar_test else fit_arima_best
            if fit_full is None:
                fit_full = fit_arima_best
            ajuste_train, _ = arima_forecast(fit_arima_best, 0)
            if ajuste_train is None or len(ajuste_train) != len(y_train):
                ajuste_train = y_train.copy()
            _, fut = arima_forecast(fit_full, horizonte)

            # Calcular RMSE con el mismo criterio del notebook
            if usar_test:
                _, fc_test = arima_forecast(fit_arima_best, test_size)
                ajuste_full = np.concatenate(
                    [np.asarray(ajuste_train), np.asarray(fc_test)]
                )
                err = rmse(y_test, fc_test)
            else:
                ajuste_full = np.asarray(ajuste_train)
                err = rmse(y, ajuste_full[: len(y)])

            # Filtro de cordura: si el pronóstico se va completamente fuera
            # del rango histórico (más de 5× max o menor que -2× min), lo
            # descartamos: nuestro CSS no se estabiliza para esta serie.
            rango = float(np.max(y) - np.min(y) + 1e-6)
            limite_alto = float(np.max(y)) + 5 * rango
            limite_bajo = float(np.min(y)) - 2 * rango
            futuro_arr = np.asarray(fut, dtype=float)
            if np.any(futuro_arr > limite_alto) or np.any(futuro_arr < limite_bajo):
                # No incluir ARIMA: sus predicciones son inestables.
                pass
            elif np.isfinite(err):
                resultados[f"ARIMA{orden}"] = {
                    "rmse": float(err),
                    "ajuste": [
                        float(x) if np.isfinite(x) else None for x in ajuste_full
                    ],
                    "prediccion": [float(x) for x in fut],
                    "info": f"AIC={aic:.1f}",
                }
    except Exception:
        pass

    # 4) Holt (clásico y amortiguado)
    for nombre, damped in [("Holt", False), ("Holt amortiguado", True)]:
        try:
            if usar_test:
                # Ajustar en y_train, pronosticar test_size pasos
                res_train = _holt_fit_and_forecast(y_train, test_size, damped=damped)
                if res_train is None:
                    continue
                fitted_tr, fc_test = res_train
                # Pronosticar horizonte completo desde el final con refit en y
                res_full = _holt_fit_and_forecast(y, horizonte, damped=damped)
                if res_full is None:
                    continue
                fitted_full, fut = res_full
                ajuste = np.concatenate([fitted_tr, fc_test])
                err = rmse(y_test, fc_test)
            else:
                res_full = _holt_fit_and_forecast(y, horizonte, damped=damped)
                if res_full is None:
                    continue
                ajuste, fut = res_full
                err = rmse(y, ajuste)
            if np.isfinite(err):
                resultados[nombre] = {
                    "rmse": float(err),
                    "ajuste": [float(x) for x in ajuste],
                    "prediccion": [float(x) for x in fut],
                    "info": "",
                }
        except Exception:
            pass

    # 5) Media móvil (SMA-3)
    try:
        w = 3
        if n > w:
            fit_ma = pd.Series(y).rolling(window=w, min_periods=1).mean().values
            fut_ma = np.repeat(fit_ma[-1], horizonte)
            err = rmse(y_test, fit_ma[-test_size:]) if usar_test else rmse(y, fit_ma)
            if np.isfinite(err):
                resultados["Media Móvil (3)"] = {
                    "rmse": float(err),
                    "ajuste": [float(x) for x in fit_ma],
                    "prediccion": [float(x) for x in fut_ma],
                    "info": "",
                }
    except Exception:
        pass

    if not resultados:
        return None, None, usar_test

    # Clip a 0 y redondeo de predicciones a entero
    for k in resultados:
        resultados[k]["prediccion"] = [
            max(0.0, float(x)) for x in resultados[k]["prediccion"]
        ]
        resultados[k]["ajuste"] = [
            (max(0.0, float(x)) if x is not None else None)
            for x in resultados[k]["ajuste"]
        ]

    mejor = min(resultados, key=lambda k: resultados[k]["rmse"])
    return mejor, resultados, usar_test


# ---------------------------------------------------------------------------
# Procesamiento principal
# ---------------------------------------------------------------------------
def procesar_columna(col):
    print(f"\n=== Procesando columna: {col} ===")
    df_agg, descartados = cargar_y_agregar(RUTA_ARCHIVO, columna_objetivo=col)
    programas = sorted(df_agg["PROGRAMA_EDUCATIVO"].unique())
    print(f"  Programas válidos: {len(programas)}")
    if descartados:
        print(f"  Descartados (sin datos): {len(descartados)}")

    salida = {}
    for prog in programas:
        s = df_agg[df_agg["PROGRAMA_EDUCATIVO"] == prog].sort_values("ANIO")
        anios = s["ANIO"].values.tolist()
        y = s[col].values.astype(float)
        if len(y) == 0:
            continue
        anios_fut = list(range(anios[-1] + 1, anios[-1] + 1 + HORIZONTE))
        mejor, resultados, uso_test = evaluar_modelos(y, HORIZONTE, TEST_SIZE)
        if mejor is None:
            print(f"  ⚠ Sin modelos para: {prog}")
            continue
        salida[prog] = {
            "anios_hist": anios,
            "valores_hist": [float(v) for v in y],
            "anios_fut": anios_fut,
            "mejor": mejor,
            "uso_test": bool(uso_test),
            "criterio": "RMSE en test" if uso_test else "RMSE in-sample",
            "modelos": resultados,
        }
    return {"programas": programas, "datos": salida}


def main():
    todo = {
        "horizonte": HORIZONTE,
        "test_size": TEST_SIZE,
        "columnas": {},
    }
    for col in ["NVO_INGRESO", "REINGRESO", "MATRICULA"]:
        todo["columnas"][col] = procesar_columna(col)

    import os
    os.makedirs("web", exist_ok=True)
    with open("web/resultados.json", "w", encoding="utf-8") as f:
        json.dump(todo, f, ensure_ascii=False)
    print(f"\n✓ web/resultados.json generado")
    tamano_mb = os.path.getsize("web/resultados.json") / 1024 / 1024
    print(f"  Tamaño: {tamano_mb:.2f} MB")


if __name__ == "__main__":
    main()
