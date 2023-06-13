import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np


class Mi_Asset:
    def __init__(self, limit):
        self.limit = limit
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.limit:
            self.current += 1
            return self.current
        else:
            raise StopIteration

    def __createWrite__(self, key):
        return st.text_input('Ingrese el ticker:', key)

@st.cache_data
def descarga_procesado(lista, fecha = str):
  assets_df = []
  try:
      for i in lista:
        df = yf.download(i, start = fecha)['Adj Close']
        assets_df.append(df)
  except:
      st.write("Error!!")
      st.write("Compruebe que los tickers sean correctos.")
  df = pd.concat(assets_df, axis= 1)
  df.columns = lista
  df.dropna(inplace=True)
  return df


@st.cache_data
def montecarlo(n_iter, n_stocks, df_retornos):
  portfolio_returns, portfolio_volatilities, portfolio_sharpe  = [list() for _ in range(3)]
  all_weights = np.zeros((n_iter, n_stocks))

  for i in range(n_iter):
  
    weights = np.random.random(n_stocks)
    weights = weights / np.sum(weights)

    all_weights[i, :] = weights
    ret = np.sum(df_retornos.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(df_retornos.cov() * 252, weights)))
    sr = ret / vol
    
    portfolio_returns.append(ret)
    portfolio_volatilities.append(vol)
    portfolio_sharpe.append(sr)

  return pd.DataFrame({
      'Retornos': portfolio_returns,
      'Volatibilidad': portfolio_volatilities,
      'Sharpe': portfolio_sharpe,
      'Pesos': np.round(all_weights, 3).tolist()
      })

def benchmkark(df, sp, pesos=list):    
    data = df / df.iloc[0]
    dff = pd.DataFrame()
    indice = -1
    for i in data:
        indice += 1
        dff[i] = (data[i] * pesos[indice]) * 100000
    
    dff['Value'] = dff.sum(axis=1)
    dff['SP500'] = (sp / sp.iloc[0]) * 100000
    dff.dropna(inplace=True)
    return dff
    