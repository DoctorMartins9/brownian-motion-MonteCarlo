import quandl
import numpy as np
import matplotlib.pyplot as plt
import math
from statistics import mean,variance,stdev
from math import exp, log, sqrt, pi
from wallstreet import Stock, Call, Put
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# Funzione che, dato il ticker, ritorna media, varianza e s0
def getData(title_name):
    quandl.ApiConfig.api_key = 'NxTUTAQswbKs5ybBbwfK'
    data = quandl.get('WIKI/' + title_name)
    close = data['2016-10':'2017-10']['Adj. Close']
    annual_return = (close[-1]/close[1])** (365.0/len(close)) - 1
    annual_vol = (close/close.shift(1)-1)[1:].std()*np.sqrt(252)
    mu = annual_return 
    sigma = annual_vol
    s0 = close[-1] # 903.5
    return [mu,sigma,s0]

# Funzione che calcola la normale
def norm_cdf(x):
    k = 1.0/(1.0+0.2316419*x)
    k_sum = k * (0.319381530 + k * (-0.356563782 + \
        k * (1.781477937 + k * (-1.821255978 + 1.330274429 * k))))

    if x >= 0.0:
        return (1.0 - (1.0 / ((2 * pi)**0.5)) * exp(-0.5 * x * x) * k_sum)
    else:
        return 1.0 - norm_cdf(-x)

# Funzione che calcola il delta
def delta_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta_call = norm_cdf(d1)
    return delta_call

# Funzione che calcola il gamma
def gamma_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
    gamma = prob_density / (S * sigma * np.sqrt(T))
    return gamma

# Funzione che calcola vega
def vega_call(S, S0, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
    vega = S0 * prob_density * np.sqrt(T)    
    return vega

# Plots delta, gamma and vega
def plot_greek(s0, strike_call, T, mu, sigma):

    # Creo i range di maturità e s per fare i grafici
    s = np.array([range(int(strike_call-100),int(strike_call+100),1) for i in range(23)])
    I = np.ones((np.shape(s)))
    time = np.arange(1,12.5,0.5)/12
    t = np.array([ele for ele in time for i in range(np.shape(s)[1])]).reshape(np.shape(s))

    # Inizializzo a zero i valori di delta, gamma e vega
    delta = np.zeros(np.shape(s))
    gamma = np.zeros(np.shape(s))
    vega = np.zeros(np.shape(s))

    # Calcolo delta, gamma e vega
    for i in range(np.shape(s)[0]):
        for j in range(np.shape(s)[1]):
            delta[i][j] = delta_call(s0, s[i][j], t[i][j], mu, sigma)
            gamma[i][j] = gamma_call(s0, s[i][j], t[i][j], mu, sigma)
            vega[i][j] = vega_call(s0,s0, s[i][j], t[i][j], mu, sigma)

    # Sistemo le dimensioni di delta, gamma e vega per poter fare il grafico
    delta = np.array(delta).reshape(np.shape(s))
    gamma = np.array(gamma).reshape(np.shape(s))
    vega = np.array(vega).reshape(np.shape(s))

    fig = plt.figure(figsize=(20,11))

    # Preparo il grafico di DELTA
    z = delta
    ax = fig.add_subplot(221, projection='3d')
    ax.view_init(40,290)
    ax.plot_wireframe(s, t, z, rstride=1, cstride=1)
    ax.plot_surface(s, t, z, facecolors=cm.jet(delta),linewidth=0.001, rstride=1, cstride=1, alpha = 0.75)
    ax.set_zlim3d(0, z.max())
    ax.set_xlabel('stock price')
    ax.set_ylabel('Time to Expiration')
    ax.set_zlabel('delta')
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(z)
    cbar = plt.colorbar(m)

    # Preparo il grafico di GAMMA
    z = gamma
    ax = fig.add_subplot(222, projection='3d')
    ax.view_init(40,290)
    ax.plot_wireframe(s, t, z, rstride=1, cstride=1)
    ax.plot_surface(s, t, z, facecolors=cm.jet(delta),linewidth=0.001, rstride=1, cstride=1, alpha = 0.75)
    ax.set_zlim3d(0, z.max())
    ax.set_xlabel('stock price')
    ax.set_ylabel('Time to Expiration')
    ax.set_zlabel('gamma')
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(z)
    cbar = plt.colorbar(m)

    # Preparo il grafico di VEGA
    z = vega
    ax = fig.add_subplot(223, projection='3d')
    ax.view_init(40,290)
    ax.plot_wireframe(s, t, z, rstride=1, cstride=1)
    ax.plot_surface(s, t, z, facecolors=cm.jet(delta),linewidth=0.001, rstride=1, cstride=1, alpha = 0.75)
    ax.set_zlim3d(0, z.max())
    ax.set_xlabel('stock price')
    ax.set_ylabel('Time to Expiration')
    ax.set_zlabel('vega')
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(z)
    cbar = plt.colorbar(m)

    # Mostro il grafico finale
    plt.show()

if __name__ == '__main__':

    # Ticker
    print('Inserisci il titolo da analizzare :')
    nome = input()
    
    # Maturità
    T = 3.0/12
    print('Inserisci la maturità :')
    T = float(input())

    # Prendo i dati da yahoo finance
    mu,sigma,s0 = getData(nome)                                       
    delta_t = 0.001                     

    # Ricavo la strike call con il pacchetto wallstreet
    c = Call(nome, d=13, m=12, y=2019)  
    strike_call = c.strikes[len(c.strikes)-1]   
    
    # Mostro i valori di delta, vega e gamma con un grafico
    plot_greek(s0, strike_call, T, mu, sigma)