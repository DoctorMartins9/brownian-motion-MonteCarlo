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

def getData(title_name):
    quandl.ApiConfig.api_key = 'NxTUTAQswbKs5ybBbwfK'
    data = quandl.get('WIKI/' + title_name)
    close = data['2016-10':'2017-10']['Adj. Close']
    annual_return = (close[-1]/close[1])** (365.0/len(close)) - 1
    annual_vol = (close/close.shift(1)-1)[1:].std()*np.sqrt(252)
    mu = annual_return 
    sigma = annual_vol
    s0 = close[-1]
    return [mu,sigma,s0]

def norm_cdf(x):
    k = 1.0/(1.0+0.2316419*x)
    k_sum = k * (0.319381530 + k * (-0.356563782 + \
        k * (1.781477937 + k * (-1.821255978 + 1.330274429 * k))))

    if x >= 0.0:
        return (1.0 - (1.0 / ((2 * pi)**0.5)) * exp(-0.5 * x * x) * k_sum)
    else:
        return 1.0 - norm_cdf(-x)

def d_j(j, S, K, r, v, T):
    return (log(S/K) + (r + ((-1)**(j-1))*0.5*v*v)*T)/(v*(T**0.5))

def vanilla_call_price(S, K, r, v, T):
    return S * norm_cdf(d_j(1, S, K, r, v, T)) - \
        K*exp(-r*T) * norm_cdf(d_j(2, S, K, r, v, T))


def mc_euro_options(option_type,s0,strike,maturity,r,sigma,num_reps):
    payoff_sum = 0
    for j in range(num_reps):
        st = s0
        st = st*exp(((r-0.5*sigma**2)*maturity + sigma*np.sqrt(maturity)*np.random.normal(0, 1)))
        if option_type == 'c':
            payoff = max(0,st-strike)
        elif option_type == 'p':
            payoff = max(0,strike-st)
        payoff_sum += payoff
    premium = (payoff_sum/float(num_reps))*exp((-r*maturity))
    return premium


if __name__ == '__main__':
    
    # Ticker
    print('Inserisci il titolo da analizzare :')
    nome = input()
    
    # Maturità
    T = 3.0/12
    print('Inserisci la maturità :')
    T = float(input())
    
    # Numero di simulazioni
    num_reps = 100
    print('Inserisci il numero di simulazioni:')
    num_reps = int(input())

    mu,sigma,s0 = getData(nome)                                       
    delta_t = 0.001                     

    c = Call(nome, d=13, m=12, y=2019)  
    strike_call = c.strikes[len(c.strikes)-1]   

    # Price Call with MonteCarlo and GMB
    mc_call = mc_euro_options('c',s0,strike_call,T,mu,sigma,num_reps)
    call = vanilla_call_price(s0,strike_call,mu,sigma,T)
    print("Montecarlo price call : " + str(mc_call))
    print("Price call standard   : " + str(call))