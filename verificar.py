# Verificador de afirmaciones: recalcula desde los datos crudos y los modelos guardados
# cada cifra citada en el informe y el dashboard, y la compara con el valor escrito.
# Uso: python verificar.py   → sale con código 1 si alguna afirmación no cuadra.
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent
RES = BASE / 'resultados'
ok = mal = 0


def afirma(texto, escrito, real, tol):
    global ok, mal
    bien = abs(float(escrito) - float(real)) <= tol
    ok += bien
    mal += not bien
    print(f'{"OK " if bien else "MAL"}  {texto}: escrito {escrito} · medido {real:.6g}')


crudo = pd.read_csv(BASE / 'datos' / 'diamonds.csv')
ceros = (crudo[['x', 'y', 'z']] == 0).any(axis=1)
df = crudo[~ceros].copy()
df['log_price'], df['log_carat'] = np.log(df.price), np.log(df.carat)
entreno, prueba = train_test_split(df, test_size=0.3, random_state=42)
mod = joblib.load(RES / 'modelos.joblib')
met = json.loads((RES / 'metricas.json').read_text())
it = pd.read_csv(RES / 'iteraciones.csv').set_index('nombre')

# --- datos
afirma('referencias en el catálogo', 53940, len(crudo), 0)
afirma('registros con alguna dimensión en cero', 20, ceros.sum(), 0)
afirma('diamantes analizados', 53920, len(df), 0)
afirma('filas de entrenamiento', 37744, len(entreno), 0)
afirma('filas de prueba', 16176, len(prueba), 0)

# --- modelo final recalculado desde el modelo guardado (no desde el CSV de métricas)
p5 = np.exp(mod['final'].predict(prueba[mod['cols_final']]))
afirma('M5 R² prueba', 0.9816, r2_score(prueba.price, p5), 5e-5)
afirma('M5 RMSE prueba (USD)', 535, np.sqrt(mean_squared_error(prueba.price, p5)), 0.5)
afirma('M5 MAPE prueba (%)', 6.5, mean_absolute_percentage_error(prueba.price, p5) * 100, 0.05)
p3 = np.exp(mod['m3'].predict(prueba[mod['cols_m3']]))
afirma('M3 R² prueba', 0.9612, r2_score(prueba.price, p3), 5e-5)
afirma('M3 MAPE prueba (%)', 10.4, mean_absolute_percentage_error(prueba.price, p3) * 100, 0.05)
afirma('M3 predicciones negativas', 0, (p3 < 0).sum(), 0)

# --- M0/M1 recalculados con numpy puro (independiente de sklearn)
X = np.column_stack([np.ones(len(entreno)), entreno[['carat', 'depth', 'table']]])
beta = np.linalg.lstsq(X, entreno.price, rcond=None)[0]
p1 = np.column_stack([np.ones(len(prueba)), prueba[['carat', 'depth', 'table']]]) @ beta
afirma('M1 R² prueba', 0.8587, r2_score(prueba.price, p1), 5e-5)
afirma('M1 MAPE prueba (%)', 38.0, mean_absolute_percentage_error(prueba.price, p1) * 100, 0.05)
afirma('M1 RMSE prueba (USD)', 1484, np.sqrt(mean_squared_error(prueba.price, p1)), 0.5)
afirma('M1 predicciones negativas en prueba', 894, (p1 < 0).sum(), 0)
b0 = np.polyfit(entreno.carat, entreno.price, 1)
afirma('M0 R² prueba', 0.8541, r2_score(prueba.price, np.polyval(b0, prueba.carat)), 5e-5)
afirma('M2 predicciones negativas', 1452, it.loc['M2', 'negativos'], 0)
afirma('M2 MAPE (%)', 44.0, it.loc['M2', 'mape'] * 100, 0.05)
afirma('M4 R² prueba', 0.9655, it.loc['M4', 'r2'], 5e-5)
afirma('M5 brecha R² entreno-prueba', 0.0047, it.loc['M5', 'brecha_r2'], 5e-5)

# --- intervalo: cobertura medida en prueba con cuantiles calibrados en entrenamiento
r = np.log(prueba.price) - np.log(p5)
cob = ((r >= mod['q10']) & (r <= mod['q90'])).mean()
afirma('cobertura del intervalo del 80 % en prueba (%)', 80.1, cob * 100, 0.05)
afirma('semiancho inferior del intervalo (%)', -9.9, (np.exp(mod['q10']) - 1) * 100, 0.05)
afirma('semiancho superior del intervalo (%)', 10.6, (np.exp(mod['q90']) - 1) * 100, 0.05)

# --- paradoja de Simpson, desde los datos crudos
media = df.groupby('color').price.mean()
afirma('precio medio color D (USD)', 3168, media['D'], 0.5)
afirma('precio medio color J (USD)', 5324, media['J'], 0.5)
peso = df.groupby('color').carat.mean()
afirma('peso medio color J (q)', 1.16, peso['J'], 0.005)
afirma('peso medio color D (q)', 0.66, peso['D'], 0.005)
banda = df[df.carat.between(0.3, 0.5)]
afirma('D - J en banda 0,3-0,5 q (USD)', 309, banda[banda.color == 'D'].price.mean() - banda[banda.color == 'J'].price.mean(), 0.5)
coef = pd.read_csv(RES / 'coeficientes_m3.csv').set_index('termino')
afirma('prima de D sobre J a igual peso (%)', 67, coef.loc['cat__color_D', 'efecto_pct'], 0.5)
afirma('prima de IF sobre I1 (%)', 208, coef.loc['cat__clarity_IF', 'efecto_pct'], 0.5)
afirma('prima de Ideal sobre Fair (%)', 18, coef.loc['cat__cut_Ideal', 'efecto_pct'], 0.5)

# --- elasticidad del peso, recalculada con numpy (MCO con dummies)
Xd = pd.get_dummies(entreno[['cut', 'color', 'clarity']], drop_first=True).astype(float)
Xe = np.column_stack([np.ones(len(entreno)), entreno.log_carat, Xd])
el = np.linalg.lstsq(Xe, entreno.log_price, rcond=None)[0][1]
afirma('elasticidad precio-peso', 1.88, el, 0.005)

# --- hiperparámetros y experimentos
afirma('max_depth elegido', 6, met['mejor_max_depth'], 0)
afirma('α Ridge elegido', 3.16, met['mejor_alfa_ridge'], 0.005)
a = pd.read_csv(RES / 'barrido_alfa.csv')
las = a[(a.tipo == 'lasso') & np.isclose(a.alfa, 1e-4)].iloc[0]
afirma('Lasso α=1e-4: términos activos', 108, las.coef_no_nulos, 0)
cal = pd.read_csv(RES / 'calidad_datos.csv')
afirma('R² con volumen y ceros imputados', 0.9592, cal.r2[1], 5e-5)
afirma('R² con volumen y ceros eliminados', 0.9613, cal.r2[2], 5e-5)
cu = pd.read_csv(RES / 'curva_aprendizaje.csv')
afirma('M3 con 755 filas: MAPE (%)', 10.7, cu[(cu.modelo == 'M3') & (cu.n_entreno == 755)].mape.iloc[0] * 100, 0.05)
afirma('M5 con 18.872 filas: MAPE (%)', 6.61, cu[(cu.modelo == 'M5') & (cu.n_entreno == 18872)].mape.iloc[0] * 100, 0.005)
pr = pd.read_csv(RES / 'predicciones_prueba.csv')
afirma('diamantes de prueba ≥20 % bajo su valor', 255, (pr.desvio_pct <= -20).sum(), 0)

# --- pies de figura del informe (predicho frente a real)
pr['tercil'] = pd.qcut(pr.price, 3, labels=['bajo', 'medio', 'alto'])
for col, escrito in (('pred_m3', 12.2), ('pred_final', 5.7)):
    e = (pr.price / pr[col] - 1).abs()
    afirma(f'{col}: MAPE en el tercio de menor precio (%)', escrito, e[pr.tercil == 'bajo'].mean() * 100, 0.05)
for col, lo, hi in (('pred_m3', -6.1, 4.1), ('pred_final', -1.0, 1.0)):
    med = np.log(pr.price / pr[col]).groupby(pd.qcut(pr[col], 10), observed=True).median()
    rango = (np.exp(med) - 1) * 100
    if col == 'pred_m3':
        afirma('M3: residuo mediano mínimo por decil (%)', lo, rango.min(), 0.05)
        afirma('M3: residuo mediano máximo por decil (%)', hi, rango.max(), 0.05)
    else:
        afirma('M5: |residuo mediano| máximo por decil < 1 % (1 = sí)', 1, float(rango.abs().max() < 1), 0)

print(f'\n{ok} OK, {mal} incorrectas')
sys.exit(1 if mal else 0)
