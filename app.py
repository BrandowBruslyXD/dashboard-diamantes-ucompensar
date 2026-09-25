# Dashboard — ¿Cuánto vale un diamante? (Actividad 6 — Transferencia)
# Lee los resultados que deja modelo.py; no entrena nada al arrancar.
# Uso local:  python app.py   → http://127.0.0.1:8050
# En Binder:  binder/start lo lanza detrás de jupyter-server-proxy (DASH_PREFIX).
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

BASE = Path(__file__).parent
RES = BASE / 'resultados'

# ---------------------------------------------------------------- datos
df = pd.read_csv(BASE / 'datos' / 'diamonds.csv')
df = df[(df[['x', 'y', 'z']] > 0).all(axis=1)].copy()
pred = pd.read_csv(RES / 'predicciones_prueba.csv')
iters = pd.read_csv(RES / 'iteraciones.csv')
alfa = pd.read_csv(RES / 'barrido_alfa.csv')
prof = pd.read_csv(RES / 'barrido_profundidad.csv')
curva = pd.read_csv(RES / 'curva_aprendizaje.csv')
calidad = pd.read_csv(RES / 'calidad_datos.csv')
simpson = pd.read_csv(RES / 'simpson_color.csv')
coef = pd.read_csv(RES / 'coeficientes_m3.csv')
met = json.loads((RES / 'metricas.json').read_text())
mod = joblib.load(RES / 'modelos.joblib')

ORDEN = {
    'cut': ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
    'color': ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
    'clarity': ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'],
}
NOMBRE = {'cut': 'Corte', 'color': 'Color', 'clarity': 'Pureza'}
for c, o in ORDEN.items():
    df[c] = pd.Categorical(df[c], o, ordered=True)

# Dimensiones típicas por peso (ley de potencia ajustada a los datos) para el tasador:
# el modelo final usa x, y, z y el usuario solo conoce el peso.
POT = {d: np.polyfit(np.log(df.carat), np.log(df[d]), 1) for d in ('x', 'y', 'z')}

final = iters.iloc[-1]
m3 = iters[iters.nombre == 'M3'].iloc[0]
m1 = iters[iters.nombre == 'M1'].iloc[0]

# ---------------------------------------------------------------- estilo (paleta validada, modo claro)
C = {
    'fondo': '#f4f3f0', 'superficie': '#fcfcfb', 'borde': '#e4e2dc',
    'texto': '#0b0b0b', 'texto2': '#52514e', 'tenue': '#8a887f',
    'azul': '#2a78d6', 'naranja': '#eb6834', 'aqua': '#1baf7a', 'gris': '#c9c7c0',
}
RAMPA = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#104281']


def rampa(n):
    """n pasos ordinales de la rampa azul (claro = menor calidad)."""
    idx = np.linspace(0, len(RAMPA) - 1, n).round().astype(int)
    return [RAMPA[i] for i in idx]


FUENTE = 'Inter, "Segoe UI", system-ui, sans-serif'


def estilo(fig, alto=380, leyenda=True):
    fig.update_layout(
        template='simple_white', height=alto, separators=',.', font=dict(family=FUENTE, size=13, color=C['texto2']),
        paper_bgcolor=C['superficie'], plot_bgcolor=C['superficie'],
        margin=dict(l=60, r=20, t=40, b=50), showlegend=leyenda,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, title=None),
        hoverlabel=dict(bgcolor='white', font=dict(family=FUENTE, color=C['texto'])),
    )
    fig.update_traces(cliponaxis=False, selector=dict(type='bar'))
    fig.update_xaxes(gridcolor=C['borde'], linecolor=C['gris'], zeroline=False, showgrid=False)
    fig.update_yaxes(gridcolor=C['borde'], linecolor=C['gris'], zeroline=False, showgrid=True)
    return fig


def usd(v):
    return f'{v:,.0f}'.replace(',', '.') + ' USD'


def num(v, d=4):
    return f'{v:.{d}f}'.replace('.', ',')


def pct(v, d=1):
    return f'{v * 100:.{d}f}'.replace('.', ',') + ' %'


def tarjeta(*hijos, **kw):
    return html.Div(hijos, className='tarjeta', **kw)


def kpi(valor, etiqueta, nota=''):
    return html.Div([html.Div(valor, className='kpi-valor'), html.Div(etiqueta, className='kpi-etiqueta'),
                     html.Div(nota, className='kpi-nota')], className='kpi')


def parrafo(t):
    return dcc.Markdown(t, className='narrativa')


# ---------------------------------------------------------------- figuras estáticas
def fig_evolucion():
    colores = [C['gris']] * (len(iters) - 1) + [C['azul']]
    fig = go.Figure(go.Bar(
        x=iters.nombre, y=iters.mape * 100, marker_color=colores, marker_line_width=0,
        text=[f'{v * 100:.1f} %'.replace('.', ',') for v in iters.mape], textposition='outside',
        customdata=np.stack([iters.descripcion, iters.r2, iters.rmse], axis=1),
        hovertemplate='<b>%{x}</b> — %{customdata[0]}<br>Error porcentual medio: %{y:.1f} %'
                      '<br>R²: %{customdata[1]:.4f}<br>RMSE: %{customdata[2]:,.0f} USD<extra></extra>'))
    fig.update_yaxes(title='Error porcentual medio (MAPE, %)', rangemode='tozero')
    fig.update_xaxes(title='Iteración del modelo')
    return estilo(fig, 340, leyenda=False)


def fig_simpson_bruto():
    fig = go.Figure(go.Bar(
        x=simpson.color, y=simpson.precio_medio_bruto, marker_color=rampa(7), marker_line_width=0,
        customdata=simpson.peso_medio,
        hovertemplate='Color %{x}<br>Precio medio: %{y:,.0f} USD<br>Peso medio: %{customdata:.2f} q<extra></extra>'))
    fig.update_yaxes(title='Precio medio (USD)')
    fig.update_xaxes(title='Color (J = peor → D = mejor)')
    return estilo(fig, 320, leyenda=False)


def fig_simpson_ajustado():
    fig = go.Figure(go.Bar(
        x=simpson.color, y=simpson.efecto_pct, marker_color=rampa(7), marker_line_width=0,
        text=[f'+{v:.0f} %' if v > 0 else 'base' for v in simpson.efecto_pct], textposition='outside',
        hovertemplate='Color %{x}: %{y:.1f} % más caro que J<br>a igual peso, corte y pureza<extra></extra>'))
    fig.update_yaxes(title='Prima sobre color J (%)', rangemode='tozero')
    fig.update_xaxes(title='Color (J = peor → D = mejor)')
    return estilo(fig, 320, leyenda=False)


def fig_efectos():
    filas = coef[coef.termino.str.startswith('cat__')].copy()
    filas['var'] = filas.termino.str.extract(r'cat__(\w+)_')[0]
    filas['nivel'] = filas.termino.str.replace(r'cat__\w+?_', '', regex=True)
    filas['etq'] = filas['var'].map(NOMBRE) + ': ' + filas['nivel']
    filas = filas.sort_values('efecto_pct')
    fig = go.Figure(go.Bar(
        y=filas.etq, x=filas.efecto_pct, orientation='h', marker_color=C['azul'], marker_line_width=0,
        hovertemplate='%{y}<br>%{x:.1f} % sobre el nivel base<extra></extra>'))
    fig.update_xaxes(title='Prima sobre el peor nivel de su atributo (%), a igual peso', showgrid=True)
    fig.update_yaxes(showgrid=False)
    return estilo(fig, 520, leyenda=False)


# ---------------------------------------------------------------- layout
def pestaña_historia():
    return html.Div([
        html.Div([
            kpi(num(final.r2, 3).replace('0,', '0,'), 'R² en datos no vistos', f'M1 de la actividad 4: {num(m1.r2, 3)}'),
            kpi(pct(final.mape), 'error típico por diamante', f'antes: {pct(m1.mape)}'),
            kpi(usd(final.rmse), 'RMSE en prueba', f'antes: {usd(m1.rmse)}'),
            kpi(f'{met["n_limpio"]:,}'.replace(',', '.'), 'diamantes analizados', f'{met["n_ceros"]} descartados por medidas en cero'),
        ], className='fila-kpi'),
        html.Div([
            tarjeta(
                html.H3('El problema'),
                parrafo(f"""Una joyería que compra y vende diamantes necesita **fijar un precio justo a partir de
los atributos del catálogo** —peso, corte, color y pureza— sin depender del olfato de un tasador.
En la práctica 2 una recta con el peso explicaba el 85 % del precio pero **predecía precios negativos**
para las piedras pequeñas; en la actividad 4 la regresión múltiple apenas mejoró (R² {num(m1.r2, 4)}) y el
contraste de hipótesis destapó una **paradoja de Simpson** con el color.

La pregunta de esta etapa: **¿qué modelo valora un diamante con un error que un comprador aceptaría, y qué
aprendemos del mercado al construirlo?**"""),
            ),
            tarjeta(
                html.H3('Qué cambió el modelo, iteración por iteración'),
                dcc.Graph(figure=fig_evolucion(), config={'displayModeBar': False}),
                parrafo(f"""El salto grande **no vino de un algoritmo más sofisticado sino de mirar bien el problema**:
el precio crece de forma multiplicativa, así que modelar **log(precio) ~ log(peso)** más las categóricas (M3)
bajó el error típico de {pct(m1.mape)} a {pct(m3.mape)} y eliminó los {int(m1.negativos)} precios negativos de M1.
El gradient boosting (M5) captura las interacciones restantes y deja el error en **{pct(final.mape)}**."""),
            ),
        ], className='rejilla-2'),
        tarjeta(
            html.H3('Tres hallazgos para el negocio'),
            html.Div([
                html.Div([html.Div('1', className='num'), parrafo(
                    f"""**El peso manda, y no linealmente.** Un 1 % más de peso sube el precio
≈ **{num(met['elasticidad_peso'], 2)} %** (elasticidad de M3). Duplicar el peso multiplica el precio por ≈ {num(2 ** met['elasticidad_peso'], 1)}.""")]),
                html.Div([html.Div('2', className='num'), parrafo(
                    """**El color sí se paga, aunque el promedio diga lo contrario.** En bruto un diamante D
vale menos que uno J, porque los J son más grandes. A igual peso, corte y pureza, **D vale ≈ 67 % más que J**.""")]),
                html.Div([html.Div('3', className='num'), parrafo(
                    f"""**La pureza pesa más que el corte.** Pasar de I1 a IF triplica el precio (+208 %);
el mejor corte frente al peor solo suma ≈ 18 %. Para el comprador, el corte es donde se ahorra.""")]),
            ], className='hallazgos'),
        ),
    ])


def pestaña_explorar():
    return html.Div([
        tarjeta(
            html.Div([
                html.Div([html.Label('Colorear por'), dcc.RadioItems(
                    id='ex-color', options=[{'label': NOMBRE[k], 'value': k} for k in ORDEN],
                    value='clarity', inline=True, className='radio')], className='control'),
                html.Div([html.Label('Escala'), dcc.RadioItems(
                    id='ex-escala', options=[{'label': 'Lineal', 'value': 'linear'},
                                             {'label': 'Logarítmica', 'value': 'log'}],
                    value='log', inline=True, className='radio')], className='control'),
                html.Div([html.Label('Corte'), dcc.Dropdown(
                    id='ex-cut', options=ORDEN['cut'], value=ORDEN['cut'], multi=True)], className='control ancho'),
            ], className='filtros'),
            html.Div([
                html.Div([html.Label('Rango de peso (quilates)'), dcc.RangeSlider(
                    id='ex-peso', min=0.2, max=5.1, step=0.05, value=[0.2, 5.1],
                    marks={v: f'{v:g}'.replace('.', ',') for v in [0.2, 0.5, 1, 2, 3, 4, 5]},
                    tooltip={'placement': 'bottom'})], className='control ancho'),
            ], className='filtros'),
            html.Div(id='ex-resumen', className='resumen'),
        ),
        html.Div([
            tarjeta(html.H3('Precio frente a peso'), dcc.Graph(id='ex-disp', config={'displayModeBar': False})),
            tarjeta(html.H3(id='ex-caja-titulo'), dcc.Graph(id='ex-caja', config={'displayModeBar': False})),
        ], className='rejilla-2'),
        tarjeta(parrafo("""**Cómo leerlo.** En escala logarítmica la nube se vuelve una banda recta: esa es la evidencia
que justificó la transformación log-log de M3. Dentro de la banda, las capas por pureza o color están ordenadas —a igual peso,
mejor calidad es más cara—. Pero mira la caja: **sin controlar el peso las medianas se invierten**, porque las piedras de
peor calidad tienden a ser más grandes. Esa es la paradoja que explica la pestaña siguiente.""")),
    ])


def pestaña_paradoja():
    return html.Div([
        html.Div([
            tarjeta(html.H3('Lo que dicen los promedios'), dcc.Graph(figure=fig_simpson_bruto(), config={'displayModeBar': False}),
                    parrafo('El mejor color (D) aparenta ser **más barato** que el peor (J). Tomar esto literalmente llevaría a sobrevalorar los J.')),
            tarjeta(html.H3('Lo que dice el modelo (a igual peso, corte y pureza)'),
                    dcc.Graph(figure=fig_simpson_ajustado(), config={'displayModeBar': False}),
                    parrafo('Al controlar los confusores, el orden se endereza y es monótono: cada escalón de color suma prima.')),
        ], className='rejilla-2'),
        tarjeta(
            html.H3('Compruébalo tú: compara dos colores dentro de una misma banda de peso'),
            html.Div([
                html.Div([html.Label('Color A'), dcc.Dropdown(id='sp-a', options=ORDEN['color'], value='D', clearable=False)], className='control'),
                html.Div([html.Label('Color B'), dcc.Dropdown(id='sp-b', options=ORDEN['color'], value='J', clearable=False)], className='control'),
                html.Div([html.Label('Banda de peso (quilates)'), dcc.RangeSlider(
                    id='sp-banda', min=0.2, max=2.5, step=0.05, value=[0.3, 0.5],
                    marks={v: f'{v:g}'.replace('.', ',') for v in [0.2, 0.5, 1, 1.5, 2, 2.5]}, tooltip={'placement': 'bottom'})], className='control ancho'),
            ], className='filtros'),
            html.Div(id='sp-texto', className='resumen'),
            dcc.Graph(id='sp-fig', config={'displayModeBar': False}),
        ),
        tarjeta(
            html.H3('Cuánto paga el mercado por cada atributo (modelo M3, a igual peso)'),
            dcc.Graph(figure=fig_efectos(), config={'displayModeBar': False}),
            parrafo('La pureza domina: pasar de I1 a IF triplica el precio. El corte es el atributo que menos mueve el precio.'),
        ),
    ])


def pestaña_experimentos():
    return html.Div([
        tarjeta(
            html.Div([
                html.Div([html.Label('Experimento'), dcc.RadioItems(
                    id='xp-sel', value='prof', className='radio', inline=True, options=[
                        {'label': 'Profundidad del boosting', 'value': 'prof'},
                        {'label': 'Regularización (α)', 'value': 'alfa'},
                        {'label': 'Cantidad de datos', 'value': 'curva'},
                        {'label': 'Calidad de datos', 'value': 'calidad'},
                    ])], className='control ancho'),
            ], className='filtros'),
            html.Div([
                html.Div(dcc.Graph(id='xp-fig', config={'displayModeBar': False}), className='xp-grafico'),
                html.Div(id='xp-texto', className='xp-texto'),
            ], className='xp'),
        ),
        tarjeta(
            html.H3('Predicho frente a real en el conjunto de prueba'),
            html.Div([html.Div([html.Label('Modelo'), dcc.RadioItems(
                id='rs-mod', value='pred_final', inline=True, className='radio',
                options=[{'label': 'M3 (log-lineal)', 'value': 'pred_m3'},
                         {'label': 'M5 (final)', 'value': 'pred_final'}])], className='control')], className='filtros'),
            dcc.Graph(id='rs-fig', config={'displayModeBar': False}),
        ),
    ])


def selector(id_, var):
    return html.Div([html.Label(NOMBRE[var]), dcc.Dropdown(
        id=id_, options=ORDEN[var], value=ORDEN[var][len(ORDEN[var]) // 2], clearable=False)], className='control')


def pestaña_tasador():
    return html.Div([
        html.Div([
            tarjeta(
                html.H3('Tasador'),
                parrafo('Describe la piedra y el modelo final estima su precio con un intervalo del 80 %.'),
                html.Label('Peso (quilates)'),
                dcc.Slider(id='ts-peso', min=0.2, max=3.0, step=0.01, value=1.0,
                           marks={v: f'{v:g}'.replace('.', ',') for v in [0.2, 0.5, 1, 1.5, 2, 2.5, 3]}, tooltip={'placement': 'bottom'}),
                html.Div([selector('ts-cut', 'cut'), selector('ts-color', 'color'), selector('ts-clarity', 'clarity')],
                         className='filtros'),
                html.Div(id='ts-resultado'),
            ),
            tarjeta(
                html.H3('¿Qué pasa si cambio un atributo?'),
                dcc.Graph(id='ts-fig', config={'displayModeBar': False}),
                parrafo('Cada barra es el precio estimado moviendo **solo** ese atributo al nivel indicado; lo demás queda como lo elegiste.'),
            ),
        ], className='rejilla-2'),
        tarjeta(
            html.H3('Detector de oportunidades'),
            parrafo(f"""Diamantes del conjunto de prueba (que el modelo **nunca vio**) cuyo precio de lista está muy por debajo
de lo que su combinación de atributos vale en el mercado. Para una joyería compradora es una lista de revisión; para una
vendedora, una alerta de precios mal fijados. El intervalo del 80 % del modelo es ±10 %, así que desvíos mayores son inusuales. Los desvíos extremos (más del 50 %)
merecen revisión manual: pueden ser gangas o errores de captura en el catálogo."""),
            html.Div([
                html.Div([html.Label('Presupuesto máximo (USD)'), dcc.Slider(
                    id='op-presu', min=500, max=19000, step=500, value=5000,
                    marks={v: f'{v // 1000}k' for v in [1000, 5000, 10000, 15000, 19000]}, tooltip={'placement': 'bottom'})],
                    className='control ancho'),
                html.Div([html.Label('Descuento mínimo frente al modelo'), dcc.Slider(
                    id='op-desc', min=10, max=50, step=5, value=20, marks={v: f'{v} %' for v in range(10, 51, 10)})],
                    className='control ancho'),
            ], className='filtros'),
            html.Div(id='op-resumen', className='resumen'),
            dash_table.DataTable(
                id='op-tabla', page_size=10, sort_action='native',
                style_as_list_view=True, locale_format={'decimal': ',', 'group': '.'},
                style_header={'fontWeight': '600', 'backgroundColor': C['fondo'], 'color': C['texto'], 'border': 'none'},
                style_cell={'fontFamily': FUENTE, 'fontSize': 13, 'padding': '8px 10px', 'color': C['texto'],
                            'backgroundColor': C['superficie'], 'textAlign': 'right'},
                style_cell_conditional=[{'if': {'column_id': c}, 'textAlign': 'left'} for c in ('cut', 'color', 'clarity')],
            ),
        ),
    ])


PREFIJO = os.environ.get('DASH_PREFIX', '/')
app = Dash(__name__, title='¿Cuánto vale un diamante?', requests_pathname_prefix=PREFIJO, routes_pathname_prefix='/')
server = app.server

app.layout = html.Div([
    html.Header([
        html.Div([
            html.Div('Programación para Ciencia de Datos II · Proyecto final', className='sobretitulo'),
            html.H1('¿Cuánto vale un diamante?'),
            html.P('Un modelo de valoración para 53.920 piedras: de una recta que predecía precios negativos a un '
                   'tasador con 6,5 % de error típico.', className='bajada'),
        ], className='contenedor'),
    ], className='cabecera'),
    html.Main([
        dcc.Tabs(id='pestanas', value='historia', className='pestanas', children=[
            dcc.Tab(label='1 · La historia', value='historia', children=pestaña_historia()),
            dcc.Tab(label='2 · Explorar el catálogo', value='explorar', children=pestaña_explorar()),
            dcc.Tab(label='3 · La paradoja del color', value='paradoja', children=pestaña_paradoja()),
            dcc.Tab(label='4 · Experimentos del modelo', value='experimentos', children=pestaña_experimentos()),
            dcc.Tab(label='5 · Tasador y oportunidades', value='tasador', children=pestaña_tasador()),
        ]),
    ], className='contenedor'),
    html.Footer(html.Div('Brandow Brusly León Rodríguez · Fundación Universitaria Compensar · Datos: diamonds (ggplot2), '
                         '53.940 referencias · Partición 70-30, semilla 42', className='contenedor'), className='pie'),
])


# ---------------------------------------------------------------- callbacks
def filtrar(cortes, peso):
    d = df[df.cut.isin(cortes or []) & df.carat.between(*peso)]
    return d


@app.callback(Output('ex-disp', 'figure'), Output('ex-caja', 'figure'), Output('ex-caja-titulo', 'children'),
              Output('ex-resumen', 'children'),
              Input('ex-color', 'value'), Input('ex-escala', 'value'), Input('ex-cut', 'value'), Input('ex-peso', 'value'))
def explorar(var, escala, cortes, peso):
    d = filtrar(cortes, peso)
    niveles = ORDEN[var]
    colores = dict(zip(niveles, rampa(len(niveles))))
    muestra = d.sample(min(len(d), 12000), random_state=1) if len(d) else d
    disp = go.Figure()
    for nv in niveles:
        s = muestra[muestra[var] == nv]
        disp.add_trace(go.Scattergl(
            x=s.carat, y=s.price, mode='markers', name=nv,
            marker=dict(size=5, color=colores[nv], opacity=0.55, line=dict(width=0)),
            hovertemplate=f'{NOMBRE[var]} {nv}<br>%{{x:.2f}} q · %{{y:,.0f}} USD<extra></extra>'))
    disp.update_xaxes(title='Peso (quilates)', type=escala)
    disp.update_yaxes(title='Precio (USD)', type=escala)
    if escala == 'log':
        disp.update_xaxes(tickvals=[0.2, 0.3, 0.5, 1, 2, 3, 5], ticktext=['0,2', '0,3', '0,5', '1', '2', '3', '5'])
        disp.update_yaxes(tickvals=[300, 1000, 3000, 10000, 18000], ticktext=['300', '1.000', '3.000', '10.000', '18.000'])
    caja = go.Figure()
    for nv in niveles:
        s = d[d[var] == nv]
        caja.add_trace(go.Box(y=s.price, name=nv, marker_color=colores[nv], boxpoints=False, line=dict(width=1.5)))
    caja.update_yaxes(title='Precio (USD)', type=escala)
    if escala == 'log':
        caja.update_yaxes(tickvals=[300, 1000, 3000, 10000, 18000], ticktext=['300', '1.000', '3.000', '10.000', '18.000'])
    caja.update_xaxes(title=f'{NOMBRE[var]} (peor → mejor)')
    if len(d):
        resumen = [html.B(f'{len(d):,}'.replace(',', '.')), ' diamantes en el filtro · precio mediano ',
                   html.B(usd(d.price.median())), ' · peso mediano ', html.B(f'{d.carat.median():.2f} q'.replace('.', ','))]
        if len(d) > len(muestra):
            resumen.append(f' · el gráfico de dispersión muestra {len(muestra):,} al azar para mantenerse fluido'.replace(',', '.'))
    else:
        resumen = 'Ningún diamante cumple el filtro: amplía el rango de peso o agrega cortes.'
    return (estilo(disp, 420), estilo(caja, 420, leyenda=False),
            f'Distribución del precio por {NOMBRE[var].lower()}, sin controlar el peso', resumen)


@app.callback(Output('sp-fig', 'figure'), Output('sp-texto', 'children'),
              Input('sp-a', 'value'), Input('sp-b', 'value'), Input('sp-banda', 'value'))
def paradoja(a, b, banda):
    d = df[df.carat.between(*banda) & df.color.isin([a, b])]
    fig = go.Figure()
    for col, color in ((a, C['azul']), (b, C['naranja'])):
        s = d[d.color == col]
        fig.add_trace(go.Histogram(x=s.price, name=f'Color {col} (n={len(s):,})'.replace(',', '.'),
                                   marker_color=color, opacity=0.6, nbinsx=40, histnorm='percent'))
    fig.update_layout(barmode='overlay')
    fig.update_xaxes(title='Precio (USD)')
    fig.update_yaxes(title='% de diamantes del color')
    pa, pb = d[d.color == a].price, d[d.color == b].price
    if len(pa) < 5 or len(pb) < 5:
        texto = 'Hay muy pocos diamantes de alguno de los dos colores en esta banda para comparar.'
    else:
        dif = pa.mean() - pb.mean()
        texto = [f'En {num(banda[0], 2)}–{num(banda[1], 2)} q, el color {a} cuesta en promedio ', html.B(usd(abs(dif))),
                 ' más' if dif >= 0 else ' menos', f' que el {b} (medias {usd(pa.mean())} frente a {usd(pb.mean())}; '
                 f'pesos medios {num(d.loc[pa.index, "carat"].mean(), 2)} q y {num(d.loc[pb.index, "carat"].mean(), 2)} q). ',
                 'Si la banda es estrecha, el peso ya no confunde y aparece la prima real del color.']
    return estilo(fig, 320), texto


@app.callback(Output('xp-fig', 'figure'), Output('xp-texto', 'children'), Input('xp-sel', 'value'))
def experimentos(sel):
    fig = go.Figure()
    if sel == 'prof':
        fig.add_trace(go.Scatter(x=prof.max_depth, y=prof.r2_entreno, name='Entrenamiento', mode='lines+markers',
                                 line=dict(color=C['naranja'], width=2), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=prof.max_depth, y=prof.r2_cv, name='Validación cruzada (5 pliegues)', mode='lines+markers',
                                 line=dict(color=C['azul'], width=2), marker=dict(size=8),
                                 error_y=dict(type='data', array=prof.r2_cv_std, color=C['azul'], thickness=1)))
        fig.add_vline(x=met['mejor_max_depth'], line_dash='dot', line_color=C['tenue'],
                      annotation_text=f'elegida: {met["mejor_max_depth"]}', annotation_position='top left')
        fig.update_xaxes(title='max_depth (profundidad de cada árbol)')
        fig.update_yaxes(title='R² sobre log(precio)')
        f = prof.set_index('max_depth')
        texto = f"""**Hiperparámetro: profundidad de los árboles.** La validación mejora rápido hasta 4 y después se aplana
(de {num(f.r2_cv[6])} en 6 a {num(f.r2_cv[12])} en 12), mientras la brecha entrenamiento–validación sigue
creciendo ({num(f.brecha[1])} → {num(f.brecha[12])}): más profundidad ya solo memoriza.

Con la **regla de un error estándar** (el modelo más simple cuyo R² de validación está a menos de 1 EE del mejor) se eligió
**max_depth = {met['mejor_max_depth']}**. El resto de hiperparámetros quedó fijo (300 árboles, tasa 0,1) para que el experimento
sea controlado: una variable a la vez."""
    elif sel == 'alfa':
        for tipo, color, nombre in (('ridge', C['azul'], 'Ridge (L2)'), ('lasso', C['naranja'], 'Lasso (L1)')):
            s = alfa[alfa.tipo == tipo]
            fig.add_trace(go.Scatter(x=s.alfa, y=s.r2_cv, name=nombre, mode='lines+markers',
                                     line=dict(color=color, width=2), marker=dict(size=8),
                                     customdata=np.stack([s.coef_no_nulos, s.coef_total], axis=1),
                                     hovertemplate='α=%{x:.2g}<br>R² CV=%{y:.4f}<br>coeficientes activos: %{customdata[0]}/%{customdata[1]}<extra></extra>'))
        fig.update_xaxes(title='α (fuerza de la penalización, escala log)', type='log')
        fig.update_yaxes(title='R² de validación cruzada (log-precio)')
        l = alfa[(alfa.tipo == 'lasso')].set_index('alfa')
        texto = f"""**Regularización sobre M3 con interacciones de grado 2** (189 términos). Ridge es plano hasta α≈10: con 37.744
filas y la brecha entrenamiento–prueba en ≈ 0,002 **no hay sobreajuste que corregir**, y la penalización solo empieza a costar
cuando es grande. Se eligió α = {num(met['mejor_alfa_ridge'], 2)} con la regla de 1 EE.

Lasso aporta otra lectura: con α = 10⁻⁴ conserva **{int(l.coef_no_nulos.iloc[2])} de 189** términos perdiendo apenas
{num(l.r2_cv.iloc[0] - l.r2_cv.iloc[2])} de R². La mayoría de interacciones sobran: la conclusión fue que regularizar
no era la palanca, y que el siguiente paso debía ser un modelo no lineal (M5)."""
    elif sel == 'curva':
        for m, color in (('M3', C['naranja']), ('M5', C['azul'])):
            s = curva[curva.modelo == m]
            fig.add_trace(go.Scatter(x=s.n_entreno, y=s.mape * 100, name=f'{m} ({"log-lineal" if m == "M3" else "boosting"})',
                                     mode='lines+markers', line=dict(color=color, width=2), marker=dict(size=8),
                                     hovertemplate='%{x:,} filas<br>MAPE %{y:.2f} %<extra></extra>'))
        fig.update_xaxes(title='Filas de entrenamiento (escala log)', type='log')
        fig.update_yaxes(title='Error porcentual en prueba (%)')
        a, b = curva[curva.modelo == 'M3'], curva[curva.modelo == 'M5']
        texto = f"""**¿Conseguir más datos ayudaría?** El modelo log-lineal (M3) ya rinde lo mismo con 755 filas
({pct(a.mape.iloc[0])}) que con 37.744 ({pct(a.mape.iloc[-1])}): **su límite es la forma del modelo, no los datos**.

El boosting sí aprovecha los datos hasta ≈ 19 mil filas ({pct(b.mape.iloc[4], 2)}) y luego se estabiliza
({pct(b.mape.iloc[-1], 2)} con todo). Recolectar más diamantes del mismo tipo aportaría poco; lo valioso serían
**variables nuevas** (certificado, fluorescencia, proporciones de la talla) que hoy no están en el catálogo."""
    else:
        fig.add_trace(go.Scatter(x=calidad.r2, y=['M3 base', 'M3 + log(volumen)<br>ceros imputados', 'M3 + log(volumen)<br>ceros eliminados'],
                                 mode='markers+text', marker=dict(size=14, color=[C['tenue'], C['naranja'], C['azul']]),
                                 text=[num(v) for v in calidad.r2], textposition='top center',
                                 customdata=calidad.mape * 100,
                                 hovertemplate='R² %{x:.4f}<br>MAPE %{customdata:.2f} %<extra></extra>'))
        fig.update_xaxes(title='R² en prueba (diferencias en la cuarta cifra decimal)', range=[0.958, 0.9625], showgrid=True)
        fig = estilo(fig, 360, leyenda=False)
        texto = f"""**Procesamiento adicional: una característica nueva y la limpieza de 20 registros.** El catálogo trae 20 piedras
con alguna dimensión en cero (físicamente imposible). Se probó agregar **log(volumen) = log(x·y·z)**:

- imputando los ceros con la mediana (lo "rápido"), el R² **baja** a {num(calidad.r2[1])};
- eliminando esos 20 registros, sube apenas a {num(calidad.r2[2])}.

Lección: 20 filas malas de 53.940 (0,04 %) bastan para anular una característica. El volumen es casi redundante con
el peso (densidad constante), por eso M3 no la incluye; el boosting sí usa x, y, z con los datos limpios."""
        return fig, parrafo(texto)
    return estilo(fig, 360), parrafo(texto)


@app.callback(Output('rs-fig', 'figure'), Input('rs-mod', 'value'))
def residuos(col):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=pred[col], y=pred.price, mode='markers', name='Diamantes de prueba',
                               marker=dict(size=4, color=C['azul'], opacity=0.3),
                               hovertemplate='predicho %{x:,.0f} · real %{y:,.0f} USD<extra></extra>'))
    lim = [300, 19000]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode='lines', name='Predicción perfecta', line=dict(color=C['texto'], width=1.5, dash='dash')))
    fila = iters.iloc[-1] if col == 'pred_final' else m3
    ticks = dict(tickvals=[300, 1000, 3000, 10000, 18000], ticktext=['300', '1.000', '3.000', '10.000', '18.000'])
    fig.update_xaxes(title='Precio predicho (USD, escala log)', type='log', **ticks)
    fig.update_yaxes(title='Precio real (USD, escala log)', type='log', **ticks)
    fig.add_annotation(x=0.02, y=0.98, xref='paper', yref='paper', showarrow=False, align='left', xanchor='left',
                       text=f'R² {num(fila.r2)} · RMSE {usd(fila.rmse)} · MAPE {pct(fila.mape)}',
                       font=dict(color=C['texto'], size=13), bgcolor=C['superficie'])
    return estilo(fig, 420)


def fila_tasador(peso, cut, color, clarity):
    fila = {'carat': peso, 'cut': cut, 'color': color, 'clarity': clarity,
            'depth': float(df.depth.median()), 'table': float(df.table.median()), 'log_carat': np.log(peso)}
    for d_, (b, a) in POT.items():
        fila[d_] = float(np.exp(a + b * np.log(peso)))
    return fila


def tasar(filas):
    X = pd.DataFrame(filas)
    return np.exp(mod['final'].predict(X[mod['cols_final']]))


@app.callback(Output('ts-resultado', 'children'), Output('ts-fig', 'figure'),
              Input('ts-peso', 'value'), Input('ts-cut', 'value'), Input('ts-color', 'value'), Input('ts-clarity', 'value'))
def tasador(peso, cut, color, clarity):
    base = fila_tasador(peso, cut, color, clarity)
    p = tasar([base])[0]
    lo, hi = p * np.exp(mod['q10']), p * np.exp(mod['q90'])
    p3 = float(np.exp(mod['m3'].predict(pd.DataFrame([base])[mod['cols_m3']]))[0])
    resultado = html.Div([
        html.Div(usd(p), className='precio'),
        html.Div(f'Intervalo del 80 %: {usd(lo)} – {usd(hi)}', className='intervalo'),
        html.Div(f'Modelo interpretable M3: {usd(p3)} · medidas x, y, z estimadas a partir del peso '
                 f'({num(base["x"], 2)} × {num(base["y"], 2)} × {num(base["z"], 2)} mm)', className='kpi-nota'),
    ], className='caja-precio')
    # sensibilidad: mover un atributo a la vez
    filas, etiquetas, grupos = [], [], []
    for var, actual in (('cut', cut), ('color', color), ('clarity', clarity)):
        for nv in ORDEN[var]:
            f = dict(base)
            f[var] = nv
            filas.append(f)
            etiquetas.append(nv)
            grupos.append(NOMBRE[var])
    precios = tasar(filas)
    fig = go.Figure()
    i = 0
    for var, actual in (('cut', cut), ('color', color), ('clarity', clarity)):
        n = len(ORDEN[var])
        cols = [C['azul'] if nv == actual else C['gris'] for nv in ORDEN[var]]
        fig.add_trace(go.Bar(x=[[NOMBRE[var]] * n, ORDEN[var]], y=precios[i:i + n], marker_color=cols, marker_line_width=0,
                             name=NOMBRE[var], hovertemplate='%{x}<br>%{y:,.0f} USD<extra></extra>'))
        i += n
    fig.update_yaxes(title='Precio estimado (USD)')
    fig.update_xaxes(tickangle=-45)
    return resultado, estilo(fig, 420, leyenda=False)


@app.callback(Output('op-tabla', 'data'), Output('op-tabla', 'columns'), Output('op-resumen', 'children'),
              Input('op-presu', 'value'), Input('op-desc', 'value'))
def oportunidades(presu, desc):
    d = pred[(pred.price <= presu) & (pred.desvio_pct <= -desc)].sort_values('desvio_pct')
    cols = [('carat', 'Peso (q)'), ('cut', 'Corte'), ('color', 'Color'), ('clarity', 'Pureza'),
            ('price', 'Precio lista (USD)'), ('pred_final', 'Valor modelo (USD)'), ('desvio_pct', 'Desvío (%)')]
    total = (pred.price <= presu).sum()
    resumen = [html.B(f'{len(d):,}'.replace(',', '.')), f' de {total:,} diamantes dentro del presupuesto'.replace(',', '.'),
               f' ({len(d) / max(total, 1) * 100:.1f} %) están al menos {desc} % por debajo de su valor estimado.'.replace('.', ',', 1)]
    return (d[[c for c, _ in cols]].round(2).to_dict('records'),
            [{'name': n, 'id': c, 'type': 'numeric' if c not in ('cut', 'color', 'clarity') else 'text',
              'format': {'specifier': '.2f' if c in ('carat', 'desvio_pct') else ',.0f', 'locale': {'decimal': ',', 'group': '.'}}} for c, n in cols], resumen)


if __name__ == '__main__':
    app.run(host=os.environ.get('HOST', '127.0.0.1'), port=int(os.environ.get('PORT', 8050)), debug=False)
