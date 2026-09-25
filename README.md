# ¿Cuánto vale un diamante? — Dashboard de valoración

Proyecto final de **Programación para Ciencia de Datos II** (Fundación Universitaria Compensar, 2026-2).
Autor: Brandow Brusly León Rodríguez.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BrandowBruslyXD/dashboard-diamantes-ucompensar/HEAD?urlpath=proxy/8050/)

Modelo de valoración automática para 53.920 diamantes (dataset `diamonds`) y un dashboard en Dash que
cuenta cómo se llegó a él: de una recta con 38 % de error típico que predecía precios negativos, a un
modelo de gradient boosting con **R² 0,982 y 6,5 % de error típico** en datos no vistos.

## Ver el dashboard

**En Binder (sin instalar nada):** abre el enlace de arriba. La primera construcción tarda unos minutos;
el dashboard aparece directamente en `…/proxy/8050/`.

**En tu equipo:**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                      # abre http://127.0.0.1:8050
```

Para regenerar todos los resultados desde los datos crudos (≈ 3 minutos): `python modelo.py`.

## Estructura

| Ruta | Qué es |
|---|---|
| `modelo.py` | Todos los experimentos: iteraciones M0–M5, barridos de hiperparámetros, calidad y cantidad de datos. Determinista (semilla 42, partición 70-30). |
| `app.py` | El dashboard. Solo lee `resultados/`; no entrena al arrancar. |
| `assets/estilo.css` | Estilos del dashboard. |
| `resultados/` | Salidas de `modelo.py`: métricas, barridos, predicciones de prueba y modelos entrenados. |
| `datos/diamonds.csv` | 53.940 referencias, 10 columnas. |
| `verificar.py` | Recalcula y comprueba cada cifra citada en el informe y el dashboard. |
| `binder/` | Configuración de Binder (`requirements.txt`, `runtime.txt`, `start`). |

## Pestañas del dashboard

1. **La historia**: el problema, los indicadores y la evolución del modelo en cada iteración.
2. **Explorar el catálogo**: dispersión y cajas con filtros de corte y peso, color por atributo y escala lineal o logarítmica.
3. **La paradoja del color**: los promedios frente al modelo, con una comparación interactiva por banda de peso.
4. **Experimentos del modelo**: profundidad del boosting, regularización Ridge/Lasso, curva de aprendizaje, calidad de datos y el gráfico de predicho frente a real.
5. **Tasador y oportunidades**: precio estimado con intervalo del 80 % y análisis de sensibilidad. Incluye un detector de diamantes con precio de lista por debajo de su valor estimado.
