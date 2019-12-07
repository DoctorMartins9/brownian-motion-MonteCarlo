import quandl
import numpy as np
import matplotlib.pyplot as plt
import math
from statistics import mean,variance,stdev
from math import exp, log, sqrt, pi
from wallstreet import Stock, Call, Put

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

# An approximation to the cumulative distribution function for the standard normal distribution:
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
    """
    Price of a European call option struck at K, with
    spot S, constant rate r, constant vol v (over the
    life of the option) and time to maturity T
    """
    return S * norm_cdf(d_j(1, S, K, r, v, T)) - \
        K*exp(-r*T) * norm_cdf(d_j(2, S, K, r, v, T))

def vanilla_put_price(S, K, r, v, T):
    """
    Price of a European put option struck at K, with
    spot S, constant rate r, constant vol v (over the
    life of the option) and time to maturity T
    """
    return -S * norm_cdf(-d_j(1, S, K, r, v, T)) + \
        K*exp(-r*T) * norm_cdf(-d_j(2, S, K, r, v, T))

def GMB_plot(mu,sigma,s0,T,delta_t,num_reps):
    steps = T/delta_t
    plt.figure(figsize=(15,10))
    for j in range(num_reps):
        price_path = [s0]
        st = s0
        for i in range(int(steps)):
            st = st*exp(((mu-0.5*sigma**2)*delta_t + sigma*np.sqrt(delta_t)*np.random.normal(0, 1)))
            price_path.append(st)
        plt.plot(price_path)
    plt.ylabel('stock price',fontsize=15)
    plt.xlabel('steps',fontsize=15)
    plt.show()

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

def delta_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta_call = norm_cdf(d1)
    return delta_call

def gamma_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
    gamma = prob_density / (S * sigma * np.sqrt(T))
    return gamma

def vega_call(S, S0, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
    vega = S0 * prob_density * np.sqrt(T)    
    return vega


if __name__ == '__main__':

    # DATA COLLECTION
    nome = 'GOOG'
    c = Call(nome, d=13, m=12, y=2019)      
    mu,sigma,s0 = getData(nome)    # mu,sigma,s0
    T = 3.0/12                       # maturita'
    delta_t = 0.001     
    num_reps = 500
    strike_call = mean(c.strikes)

    # Plot the data
    #GMB_plot(mu,sigma,s0,T,delta_t,num_reps)
    
    # Price Call
    mc_call = mc_euro_options('c',s0,strike_call,T,mu,sigma,num_reps)
    call = vanilla_call_price(s0,strike_call,mu,sigma,T)

    # Greek
    delta = delta_call(s0, strike_call, T, mu, sigma)
    gamma = gamma_call(s0, strike_call, T, mu, sigma)
    vega = vega_call(s0,s0, strike_call, T, mu, sigma)


    print("MC CALL : " + str(mc_call))
    print("CALL_STD: " + str(call))
    print("delta   : " + str(delta))
    print("gamma   : " + str(gamma))
    print("vega    : " + str(vega))