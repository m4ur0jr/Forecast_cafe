# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 08:50:47 2019

@author: mauro
"""

#ANÁLISE PREDITIVA DOS PREÇOS DIÁRIOS EM DÓLAR DO CAFÉ ARÁBICA
#PROBLEMA: como se comporta o preço do café no futuro?
#OBJETIVO: fazer previsão do preço do café
#METODOLOGIA: abrodagem univariada, com algoritmo preditivo Arima
#DATASET: histórico de preços em dólar do café arábica, baixado do site Cepea

#IMPORTANDO OS PACOTES NECESSÁRIOS
import pandas as pd
import numpy as np
import statsmodels as sm
from matplotlib import pyplot as plt
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf

#Importando o conjunto de dados
base = pd.read_excel('café_price_dia_09-96_12-19.xlsx')
print(base.index) #exibe as configurações do dataset
print(base.head(5)) #exibe as cinco primeiras linhas do dataset
print(base.tail(5)) #exibe as cinco últimmas linhas do dataset
print(base.dtypes) #exibe o formato dos dados do conjunto

#Configurar as datas do conjunto
dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y')
base = pd.read_excel('café_price_dia_09-96_12-19.xlsx', parse_dates=['Data'],
                     index_col = 'Data', date_parser = dateparse)

print(base.index)

#Convertendo para série temporal
ts = base['Preço']

#Visualizando a série
plt.plot(ts)

#Agrupando os dados por ano e mês, respenctivamente, e fazendo plotagem
ts_ano = ts.resample('A').sum()
plt.plot(ts_ano)
ts_mes = ts.resample('M').sum()
plt.plot(ts_mes)

#Fazendo decomposição da série temporal. 
#Os dados estão espaçados de maneira desigual, apesar de ser uma base diária. 
#Não existe preço para sábado, domingo e feriado. Neste caso a saída é completar
#estes dados ausentes como sendo a média dos últimos setes dias ou valor igual 
#ao dia anterior. Os componentes da série são não sistemáticos
#decomposicao = seasonal_decompose(ts, model="multiplicative", freq=None)

ts.hist()
#a maioria dos preços está entre 100 e 150 dólares

#Fazendo testes de raiz unitária - Dickey fuller
result = adfuller(ts)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
#o valor-p 0,318478 é maior que 0,05 logo, rejeitamos H0 e concluímos que os 
#dados possuem raiz unitária e não obedecem aos padrões de estacionariedade

#Criando conjuntos de treino e teste
treino = ts[:int(0.8 * (len(ts)))]
teste = ts[int(0.8 * (len(ts))):]

#checando o percentual de cada grupo
print(treino.count()/ts.count())
print(teste.count()/ts.count())

#visualizando os grupos
plt.plot(treino)
plt.plot(teste)

#Construindo o modelo preditivo
modelo = auto_arima(treino, trace = True, error_action='ignore', 
                    surpress_warnings=True)
modelo.fit(treino)
forecast = modelo.predict(n_periods=len(teste + 7))
forecast = pd.DataFrame(forecast, index = teste.index, columns=['Prediction'])

#Plotando os gráficos
plt.plot(treino, label='Treino')
plt.plot(teste, label='Válido')
plt.plot(forecast, label='Previsão')
plt.show()

#Calculando erro
from math import sqrt
from sklearn.metrics import mean_squared_error
rms = sqrt(mean_squared_error(teste, forecast))
print(rms)

forecast
plt.plot(forecast)

treino.describe()
teste.describe()
modelo.summary()


















































