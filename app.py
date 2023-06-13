import matplotlib.pyplot as plt
import plotly.express as px
import scipy.optimize as sci_opt
from Funciones.Funciones import *
import time

def calculos_(weights: list):
  ret = np.sum(df_retornos.mean() * weights) * 252
  vol = np.sqrt(np.dot(weights.T, np.dot(df_retornos.cov() * 252, weights)))
  sr = ret / vol
  return np.array([ret, vol, sr])

def neg_s(weights: list) -> np.array:
  return calculos_(weights)[2] - 1

def get_vol(weights: list)-> np.array:
  return calculos_(weights)[1]

st.set_page_config(
    page_title = "Portfolio",
    page_icon = ":open_file_folder:",
)

    
st.header('Crea tu propio portafolio!')
st.markdown("Cree su portafolio basandose en la frontera eficiente de Markowitz, primero seleccione la cantidad de activos a considerar, luego escriba su ticker. los datos son obtenidos de Yahoo Finance: https://finance.yahoo.com/lookup/")
st.markdown('Por default los datos se obtienen a partir del 2015-01-01.')

st.subheader('Seleccion:')
number = st.number_input('Ingrese la cantidad de activos: ', min_value = 1, max_value = 5)
st.write('El total de activos a considerar es: ', number)

# Crear una instancia del objeto
mi_objeto = Mi_Asset(number)
ticker_l = []
n = 0
for elemento in mi_objeto:
    rdo = mi_objeto.__createWrite__(key=n)
    ticker_l.append(rdo.upper())
    n += 1

with st.spinner('Cargando'):
    df = descarga_procesado(ticker_l, '2015-01-01')
    
    sp = descarga_procesado(['^GSPC'], '2015-01-01')
    df_retornos = np.log(df / df.shift())
    df_retornos.dropna(inplace=True)
    time.sleep(12)
    st.success('¡Listo!')    
            
st.write('-' * 200)

option = st.selectbox(
    'Como le gustaria obener su cartera?',
    ('Simulacion de Monte Carlo', 'Optimizacion por Sharpe ratio', 'Optimizacion por volatibilidad'))

if option == 'Simulacion de Monte Carlo':
    df_portafolio = montecarlo(5000, number, df_retornos)
    st.markdown('Dataframe de simulaciones con sus retornos, volatibilidad, Sharpe y pesos por orden de activo.')
    st.write(df_portafolio)
    
    max_sh = df_portafolio.iloc[df_portafolio.Sharpe.idxmax()]
    min_vol = df_portafolio.iloc[df_portafolio.Volatibilidad.idxmin()]
    max_ret = df_portafolio.iloc[df_portafolio.Retornos.idxmax()]
    
    
    fig = px.scatter(df_portafolio, x='Volatibilidad', y='Retornos', color='Sharpe',
                     labels={'Volatibilidad': 'Volatibilidad esperada', 'Retornos': 'Retornos esperados'},
                     title='Optimizacion de Portfolio')
    
    fig.add_scatter(x=[min_vol[1]], y=[min_vol[0]], mode='markers', marker=dict(color='black', symbol='star', size=20), name='Maxima volatibilidad esperada')
    fig.add_scatter(x=[max_ret[1]], y=[max_ret[0]], mode='markers', marker=dict(color='black', symbol='star', size=20), name='Maximos rendimientos esperados')
    fig.add_scatter(x=[max_sh[1]], y=[max_sh[0]], mode='markers', marker=dict(color='black', symbol='star', size=20), name='Maximo Sharpe Ratio')
    
    fig.update_layout(
        coloraxis=dict(colorscale='plasma', colorbar=dict(title='Sharpe Ratio')),
        legend=dict(
            title=None,
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='right',
            x=1
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, title_font=dict(size=12)),
        yaxis=dict(showgrid=True, title_font=dict(size=12)),
    )
    
    st.subheader("Simulacion de Monte carlo")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Benchmark")
    lista_pesos = st.selectbox('Seleccione los pesos para cada activos:', df_portafolio.Pesos)
    bench = benchmkark(df, sp, pesos=lista_pesos)
    st.write(bench)
    st.subheader('Benchmark, Valor del portafolio vs SP500')
    st.line_chart(bench[['Value', 'SP500']])
  
elif option=='Optimizacion por Sharpe ratio':   
    
    st.subheader('Optimizacion del portafolio sobre el Sharpe ratio')

    bounds = tuple((0, 1) for symbol in range(number))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    init_guess = number * [1 / number]

    optimized_sharpe = sci_opt.minimize(
        neg_s, 
        init_guess, 
        method='SLSQP',
        bounds=bounds, 
        constraints=constraints
    )
        
    optimized_metrics = calculos_(weights=optimized_sharpe.x)
    st.write('Resultados: ')
    result = pd.DataFrame({
        'Retornos': optimized_metrics[0],
        'Volatibilidad': optimized_metrics[1],
        'Sharpe': optimized_metrics[2],
        'Pesos': [optimized_sharpe.x]
        })
    
    st.write(result)
    bench = benchmkark(df, sp, pesos=optimized_sharpe.x)
    st.write(bench)
    st.subheader('Benchmark, Valor del portafolio vs SP500')
    st.line_chart(bench[['Value', 'SP500']])
    
else:
    st.subheader('Optimizacion del portafolio sobre la volatibilidad')
    bounds = tuple((0, 1) for symbol in range(number))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    init_guess = number * [1 / number]
    
    optimized_vol = sci_opt.minimize(
        get_vol, 
        init_guess, 
        method='SLSQP',
        bounds=bounds, 
        constraints=constraints
    )
        
    optimized_metrics = calculos_(weights=optimized_vol.x)
    st.write('Resultados: ')
    result = pd.DataFrame({
        'Retornos': optimized_metrics[0],
        'Volatibilidad': optimized_metrics[1],
        'Sharpe': optimized_metrics[2],
        'Pesos': [np.round(optimized_vol.x, 3)]
    })
    
    st.write(result)
    bench = benchmkark(df, sp, pesos=optimized_vol.x)
    st.write(bench)
    st.subheader('Benchmark, Valor del portafolio vs SP500')
    st.line_chart(bench[['Value', 'SP500']])