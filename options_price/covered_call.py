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
    close = data['2017-10':'2019-12']['Adj. Close']
    annual_return = (close[-1]/close[1])** (365.0/len(close)) - 1
    annual_vol = (close/close.shift(1)-1)[1:].std()*np.sqrt(252)
    mu = annual_return 
    sigma = annual_vol
    s0 = close[-1] # 903.5
    return [mu,sigma,s0]

def covered_call(s0,k,strike_call):
    price = k # the stock price at expiration date
    strike = s0 # the strike price
    premium = 7.5 # the option premium

    #premium = (strike_call - s0)/100

    print((strike_call - s0)/100)

    # the payoff of short call position
    payoff_short_call = [min(premium, -(i - strike-premium)) for i in price]
    # the payoff of long stock postion
    payoff_long_stock = [i-strike for i in price]
    # the payoff of covered call
    payoff_covered_call = np.sum([payoff_short_call, payoff_long_stock], axis=0)
    plt.figure(figsize=(20,11))
    plt.plot(price, payoff_short_call, label = 'short call')
    plt.plot(price, payoff_long_stock, label = 'underlying stock')
    plt.plot(price, payoff_covered_call, label = 'covered call')
    plt.legend(fontsize = 20)
    plt.xlabel('Stock Price at Expiry',fontsize = 15)
    plt.ylabel('payoff',fontsize = 15)
    plt.title('Covered Call Strategy Payoff at Expiration',fontsize = 20)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':

    # Ticker
    print('Inserisci il titolo da analizzare :')
    nome = input()
    
    mu,sigma,s0 = getData(nome)                                       
    delta_t = 0.001                     

    c = Call(nome, d=13, m=12, y=2019)  
    strike_call = c.strikes[len(c.strikes)-1]   
    covered_call(s0,c.strikes,strike_call)