# Iteraciones del modelo de valoración de diamantes (Actividad 6 — Transferencia).
# Corre todos los experimentos y deja los resultados en resultados/.
# El dashboard (app.py) solo lee estos archivos: no entrena nada al arrancar.
# Uso: python modelo.py        (determinista: semilla 42, partición 70-30)
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

SEMILLA = 42
BASE = Path(__file__).parent
SALIDA = BASE / 'resultados'
SALIDA.mkdir(exist_ok=True)
t0 = time.time()

ORDEN = {
    'cut': ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
    'color': ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
    'clarity': ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'],
}
CATS = ['cut', 'color', 'clarity']

# ---------------------------------------------------------------- datos
crudo = pd.read_csv(BASE / 'datos' / 'diamonds.csv')
ceros = (crudo[['x', 'y', 'z']] == 0).any(axis=1)
df = crudo[~ceros].copy()
df['log_price'] = np.log(df['price'])
df['log_carat'] = np.log(df['carat'])
df['volumen'] = df['x'] * df['y'] * df['z']

entreno, prueba = train_test_split(df, test_size=0.3, random_state=SEMILLA)


def metricas(y_real, y_pred, y_real_tr=None, y_pred_tr=None):
    m = {
        'r2': r2_score(y_real, y_pred),
        'rmse': float(np.sqrt(mean_squared_error(y_real, y_pred))),
        'mape': mean_absolute_percentage_error(y_real, y_pred),
    }
    if y_real_tr is not None:
        m['r2_entreno'] = r2_score(y_real_tr, y_pred_tr)
        m['brecha_r2'] = m['r2_entreno'] - m['r2']
    return {k: round(float(v), 4) for k, v in m.items()}


def onehot(num):
    return ColumnTransformer([
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(categories=[ORDEN[c] for c in CATS], drop='first'), CATS),
    ])


def ajustar(nombre, modelo, cols, log=False, descripcion=''):
    """Entrena, evalúa en prueba (siempre en USD) y devuelve métricas."""
    objetivo = 'log_price' if log else 'price'
    modelo.fit(entreno[cols], entreno[objetivo])
    p_te = modelo.predict(prueba[cols])
    p_tr = modelo.predict(entreno[cols])
    if log:
        p_te, p_tr = np.exp(p_te), np.exp(p_tr)
    m = metricas(prueba['price'], p_te, entreno['price'], p_tr)
    m.update(nombre=nombre, descripcion=descripcion,
             negativos=int((p_te < 0).sum()))
    return m, modelo


iteraciones = []

# M0 — la recta de la práctica 2
m, _ = ajustar('M0', LinearRegression(), ['carat'],
               descripcion='Recta precio ~ peso (Práctica 2)')
iteraciones.append(m)

# M1 — regresión múltiple de la actividad 4
m, _ = ajustar('M1', LinearRegression(), ['carat', 'depth', 'table'],
               descripcion='Múltiple con continuas: peso, profundidad, tabla (Actividad 4)')
iteraciones.append(m)

# M2 — se agregan las categóricas, sin transformar
m, _ = ajustar('M2', Pipeline([('prep', onehot(['carat'])), ('reg', LinearRegression())]),
               ['carat'] + CATS,
               descripcion='Peso + corte + color + pureza (one-hot), escala USD')
iteraciones.append(m)

# M3 — ingeniería de características: log-log
cols_m3 = ['log_carat'] + CATS
m, modelo_m3 = ajustar('M3', Pipeline([('prep', onehot(['log_carat'])), ('reg', LinearRegression())]),
                       cols_m3, log=True,
                       descripcion='log(precio) ~ log(peso) + categóricas')
iteraciones.append(m)

# M4 — interacciones de grado 2 + Ridge (regularización), alfa por validación cruzada
kf = KFold(5, shuffle=True, random_state=SEMILLA)


def pipe_poli(alfa, tipo='ridge'):
    reg = Ridge(alpha=alfa) if tipo == 'ridge' else Lasso(alpha=alfa, max_iter=5000)
    return Pipeline([
        ('prep', onehot(['log_carat'])),
        ('poli', PolynomialFeatures(2, include_bias=False)),
        ('reg', reg),
    ])


barrido_alfa = []
for tipo, alfas in [('ridge', np.logspace(-3, 4, 15)), ('lasso', np.logspace(-5, -1, 9))]:
    for a in alfas:
        modelo = pipe_poli(a, tipo)
        cv = cross_val_score(modelo, entreno[cols_m3], entreno['log_price'], cv=kf, scoring='r2')
        modelo.fit(entreno[cols_m3], entreno['log_price'])
        r2_tr = r2_score(entreno['log_price'], modelo.predict(entreno[cols_m3]))
        coefs = modelo.named_steps['reg'].coef_
        barrido_alfa.append({'tipo': tipo, 'alfa': float(a), 'r2_cv': float(cv.mean()),
                             'r2_cv_std': float(cv.std()), 'r2_entreno': float(r2_tr),
                             'coef_no_nulos': int((np.abs(coefs) > 1e-8).sum()),
                             'coef_total': int(coefs.size)})
barrido_alfa = pd.DataFrame(barrido_alfa)
barrido_alfa.to_csv(SALIDA / 'barrido_alfa.csv', index=False)
# regla de un error estándar: el modelo más simple (α más grande) a 1 EE del mejor
def regla_1ee(tabla, col_simple, mas_simple_es_mayor):
    mejor = tabla.loc[tabla.r2_cv.idxmax()]
    ee = mejor.r2_cv_std / np.sqrt(5)
    aptos = tabla[tabla.r2_cv >= mejor.r2_cv - ee]
    return aptos.sort_values(col_simple, ascending=not mas_simple_es_mayor).iloc[0]


mejor_ridge = regla_1ee(barrido_alfa[barrido_alfa.tipo == 'ridge'], 'alfa', True)
m, modelo_m4 = ajustar('M4', pipe_poli(mejor_ridge.alfa), cols_m3, log=True,
                       descripcion=f'M3 + interacciones de grado 2 + Ridge (α={mejor_ridge.alfa:.3g} por CV 5-fold y regla 1-EE)')
m['alfa'] = float(mejor_ridge.alfa)
iteraciones.append(m)

# M5 — gradient boosting: barrido de profundidad para ver sobreajuste
cols_gb = ['carat', 'depth', 'table', 'x', 'y', 'z'] + CATS


def pipe_gb(prof, iters=300):
    return Pipeline([
        ('prep', ColumnTransformer([
            ('num', 'passthrough', cols_gb[:6]),
            ('cat', OneHotEncoder(categories=[ORDEN[c] for c in CATS]), CATS),
        ])),
        ('reg', HistGradientBoostingRegressor(max_depth=prof, max_iter=iters, learning_rate=0.1,
                                              min_samples_leaf=5, early_stopping=False,
                                              random_state=SEMILLA)),
    ])


barrido_prof = []
for prof in [1, 2, 3, 4, 6, 8, 12]:
    modelo = pipe_gb(prof, iters=300)
    cv = cross_val_score(modelo, entreno[cols_gb], entreno['log_price'], cv=kf, scoring='r2')
    modelo.fit(entreno[cols_gb], entreno['log_price'])
    r2_tr = r2_score(entreno['log_price'], modelo.predict(entreno[cols_gb]))
    barrido_prof.append({'max_depth': prof, 'r2_cv': float(cv.mean()), 'r2_cv_std': float(cv.std()),
                         'r2_entreno': float(r2_tr)})
barrido_prof = pd.DataFrame(barrido_prof)
barrido_prof['brecha'] = barrido_prof.r2_entreno - barrido_prof.r2_cv
barrido_prof.to_csv(SALIDA / 'barrido_profundidad.csv', index=False)
mejor_prof = int(regla_1ee(barrido_prof, 'max_depth', False).max_depth)
m, modelo_m5 = ajustar('M5', pipe_gb(mejor_prof, iters=300), cols_gb, log=True,
                       descripcion=f'Gradient boosting sobre log(precio), max_depth={mejor_prof} por CV y regla 1-EE')
m['max_depth'] = mejor_prof
iteraciones.append(m)

pd.DataFrame(iteraciones).to_csv(SALIDA / 'iteraciones.csv', index=False)

# ------------------------------------------------ calidad y cantidad de datos (sobre M3)
# (a) nueva característica log(volumen), con y sin limpiar los 20 registros con dimensiones en cero
tmp_tr, tmp_te = entreno.copy(), prueba.copy()
tmp_tr['log_vol'] = np.log(tmp_tr['volumen'])
tmp_te['log_vol'] = np.log(tmp_te['volumen'])
mod_vol = Pipeline([('prep', onehot(['log_carat', 'log_vol'])), ('reg', LinearRegression())])
mod_vol.fit(tmp_tr[['log_carat', 'log_vol'] + CATS], tmp_tr['log_price'])
m_vol = metricas(tmp_te['price'], np.exp(mod_vol.predict(tmp_te[['log_carat', 'log_vol'] + CATS])))

# con los datos crudos (ceros sustituidos por la mediana para poder tomar log, práctica común ingenua)
crudo_tr, crudo_te = train_test_split(crudo.assign(
    log_price=np.log(crudo.price), log_carat=np.log(crudo.carat),
    volumen=(crudo.x * crudo.y * crudo.z).replace(0, np.nan)), test_size=0.3, random_state=SEMILLA)
med = crudo_tr['volumen'].median()
for d in (crudo_tr, crudo_te):
    d['log_vol'] = np.log(d['volumen'].fillna(med))
mod_crudo = Pipeline([('prep', onehot(['log_carat', 'log_vol'])), ('reg', LinearRegression())])
mod_crudo.fit(crudo_tr[['log_carat', 'log_vol'] + CATS], crudo_tr['log_price'])
m_crudo = metricas(crudo_te['price'], np.exp(mod_crudo.predict(crudo_te[['log_carat', 'log_vol'] + CATS])))

calidad = pd.DataFrame([
    {'experimento': 'M3 (sin volumen)', **{k: v for k, v in iteraciones[3].items() if k in ('r2', 'rmse', 'mape')}},
    {'experimento': 'M3 + log(volumen), ceros imputados con mediana', **m_crudo},
    {'experimento': 'M3 + log(volumen), 20 registros con ceros eliminados', **m_vol},
])
calidad.to_csv(SALIDA / 'calidad_datos.csv', index=False)

# (b) curva de aprendizaje: ¿más datos mejorarían M3 y M5?
curva = []
for frac in [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]:
    sub = entreno.sample(frac=frac, random_state=SEMILLA)
    for nombre, modelo, cols in [
        ('M3', Pipeline([('prep', onehot(['log_carat'])), ('reg', LinearRegression())]), cols_m3),
        ('M5', pipe_gb(mejor_prof, iters=300), cols_gb),
    ]:
        modelo.fit(sub[cols], sub['log_price'])
        p = np.exp(modelo.predict(prueba[cols]))
        curva.append({'modelo': nombre, 'fraccion': frac, 'n_entreno': len(sub),
                      **metricas(prueba['price'], p)})
pd.DataFrame(curva).to_csv(SALIDA / 'curva_aprendizaje.csv', index=False)

# ------------------------------------------------ interpretación: coeficientes de M3
reg = modelo_m3.named_steps['reg']
nombres = modelo_m3.named_steps['prep'].get_feature_names_out()
esc = modelo_m3.named_steps['prep'].named_transformers_['num']
coef = pd.DataFrame({'termino': nombres, 'coef_log': reg.coef_})
coef['efecto_pct'] = (np.exp(coef.coef_log) - 1) * 100
elasticidad = float(reg.coef_[0] / esc.scale_[0])  # coeficiente de log(carat) en escala original
coef.to_csv(SALIDA / 'coeficientes_m3.csv', index=False)

# ------------------------------------------------ paradoja de Simpson: bruto vs ajustado
bruto = df.groupby('color', observed=True)['price'].mean().reindex(ORDEN['color'])
ajustado = coef[coef.termino.str.startswith('cat__color_')].copy()
ajustado['color'] = ajustado.termino.str.replace('cat__color_', '')
simpson = pd.DataFrame({'color': ORDEN['color'], 'precio_medio_bruto': bruto.values,
                        'peso_medio': df.groupby('color')['carat'].mean().reindex(ORDEN['color']).values})
simpson = simpson.merge(ajustado[['color', 'efecto_pct']], on='color', how='left').fillna({'efecto_pct': 0.0})
simpson.to_csv(SALIDA / 'simpson_color.csv', index=False)

# ------------------------------------------------ predicciones de prueba para el dashboard
modelo_final = modelo_m5
pred = prueba[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z', 'price']].copy()
pred['pred_m3'] = np.exp(modelo_m3.predict(prueba[cols_m3])).round(2)
pred['pred_final'] = np.exp(modelo_final.predict(prueba[cols_gb])).round(2)
# Intervalo del 80 %: se calibra con residuos fuera de pliegue del ENTRENAMIENTO
# y su cobertura se mide en PRUEBA (si se calibrara en prueba, la cobertura sería 0,80 por construcción).
oof = cross_val_predict(pipe_gb(mejor_prof, iters=300), entreno[cols_gb], entreno['log_price'], cv=kf)
q10, q90 = np.quantile(entreno['log_price'] - oof, [0.1, 0.9])
resid_log = np.log(pred['price']) - np.log(pred['pred_final'])
pred['desvio_pct'] = ((pred['price'] / pred['pred_final'] - 1) * 100).round(2)
pred.to_csv(SALIDA / 'predicciones_prueba.csv', index=False)

joblib.dump({'m3': modelo_m3, 'final': modelo_final, 'cols_m3': cols_m3, 'cols_final': cols_gb,
             'q10': float(q10), 'q90': float(q90)}, SALIDA / 'modelos.joblib')

# ------------------------------------------------ resumen
resumen = {
    'n_crudo': int(len(crudo)), 'n_ceros': int(ceros.sum()), 'n_limpio': int(len(df)),
    'n_entreno': int(len(entreno)), 'n_prueba': int(len(prueba)),
    'iteraciones': iteraciones,
    'mejor_alfa_ridge': float(mejor_ridge.alfa),
    'mejor_max_depth': mejor_prof,
    'elasticidad_peso': round(elasticidad, 4),
    'intervalo_80_log': [round(float(q10), 4), round(float(q90), 4)],
    'cobertura_80_prueba': round(float(((resid_log >= q10) & (resid_log <= q90)).mean()), 4),
    'calidad_datos': calidad.round(4).to_dict('records'),
    'segundos': round(time.time() - t0, 1),
}
(SALIDA / 'metricas.json').write_text(json.dumps(resumen, ensure_ascii=False, indent=2))
print(pd.DataFrame(iteraciones)[['nombre', 'r2', 'rmse', 'mape', 'r2_entreno', 'brecha_r2', 'negativos']])
print(calidad)
print('mejor alfa', mejor_ridge.alfa, 'mejor prof', mejor_prof, 'elasticidad', elasticidad)
print(barrido_prof)
print('segundos', resumen['segundos'])
