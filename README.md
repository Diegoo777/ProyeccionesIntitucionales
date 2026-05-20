# Predicción de Matrícula Universitaria

Página web estática que muestra, de forma interactiva, las proyecciones de matrícula a 5 años por licenciatura de la UMSNH. Replica fielmente la lógica del notebook `Predicciones_Matricula_colab.ipynb`: los mismos cinco modelos, el mismo criterio de selección, los mismos resultados.

## ¿Qué hay aquí?

- **`index.html`** — la página, con selectores de indicador, carrera y modelo, tarjetas de resumen, gráfico interactivo (Plotly), ranking de modelos y predicciones.
- **`styles.css`** — estilos.
- **`app.js`** — lógica del frontend.
- **`resultados.json`** — predicciones precomputadas para las 52+ carreras y los 3 indicadores (Nuevo Ingreso, Reingreso, Matrícula total).
- **`precomputar.py`** — script que genera `resultados.json` a partir de `MATRICULA2026.xlsx`.

## Cómo subirla a GitHub Pages

1. Crea un repositorio nuevo en GitHub (público).
2. Sube todos estos archivos a la raíz del repo:
   ```
   index.html
   styles.css
   app.js
   resultados.json
   README.md
   ```
3. Ve a **Settings → Pages**.
4. En **Source**, elige **Deploy from a branch**.
5. Selecciona la rama `main` y la carpeta `/ (root)`. Guarda.
6. Espera ~30 segundos. Tu página estará en:
   `https://<tu-usuario>.github.io/<nombre-del-repo>/`

No necesita servidor, base de datos, ni configuración adicional. Todo corre en el navegador.

## Cómo actualizar los datos

Si tienes una versión nueva del archivo `MATRICULA2026.xlsx`, regenera `resultados.json`:

```bash
pip install pandas numpy scipy scikit-learn openpyxl
python precomputar.py
```

El script vuelve a aplicar todo el flujo del notebook (extracción de año desde `CICLO`, agregación SS+SA por año, descarte de programas sin datos) y ajusta los 5 modelos a cada carrera. Sube el `resultados.json` actualizado al repo y la página recargada mostrará los nuevos números.

## Cómo se elige el "mejor" modelo

Mismo criterio que el notebook:

- Si la serie tiene **≥ 8 años**, se reservan los últimos 3 como conjunto de prueba y el modelo ganador es el de **menor RMSE en test**.
- Si la serie es más corta, se usa **RMSE in-sample** sobre toda la serie.

Los 5 modelos en competencia: **Regresión Lineal**, **Exponencial**, **ARIMA(p,d,q)** con grid search por AIC (`p,q ∈ {0..3}`, `d ∈ {0,1,2}`), **Holt** (clásico y amortiguado) y **Media Móvil (3)**.

## Diferencias con el notebook original

El notebook usa `statsmodels` para ARIMA y Holt. Como `statsmodels` no es necesario en una página estática, los modelos se implementan en `precomputar.py` con `scipy.optimize`:

- **ARIMA**: estimado por **Conditional Sum of Squares** (CSS), Gaussian likelihood, AIC para el grid search. Filtro de cordura: si el pronóstico se sale del rango histórico × 5, el modelo se descarta para esa carrera.
- **Holt** (clásico y amortiguado): optimización numérica de α, β (y φ para el amortiguado) minimizando SSE.

Los criterios de **decisión** son idénticos; los **valores numéricos** pueden diferir en una o dos unidades respecto a `statsmodels` por las distintas implementaciones internas.

## Licencia

Código de la página: libre, úsalo como quieras. Los datos vienen de `MATRICULA2026.xlsx`.
