# Library for handling csv files
import csv
import math
from statistics import mean,variance,stdev
import matplotlib.pyplot as plt
import numpy as np
from math import exp, log, sqrt, pi

# File csv da salvare nella cartella data per poter essere analizzato
def analyze(file_name):
    close_adj = []
    with open('../data/' + file_name) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        for row in csv_reader:
            close_adj.append(row[5])
    
    # Calcolo rendimento giornaliero
    rendimento = []
    for i in range(2,len(close_adj)):
        #rendimento.append(math.log(float(close_adj[i])) /  math.log(float(close_adj[i-1]))-1)
        rendimento.append(float(close_adj[i]) / float(close_adj[i-1]) -1)

    # Calcolo del mu annuo
    rendimento_annuo = []
    for r in rendimento:
        rendimento_annuo.append(r * 252)
    # Calcolo di media e varianza
    media = mean(rendimento_annuo)
    deviation = stdev(rendimento_annuo)
    var = variance(rendimento_annuo)

    return [media,deviation,var]

def display_GMB(mu,sigma):
 
    S0 = 100 #initial stock price
    K = 100 #strike price
    r = mu  # 0.05 #risk-free interest rate
    #sigma = 0.50 #volatility in market
    T = 1 #time in years
    N = 100 #number of steps within each simulation
    deltat = T/N #time step
    i = 1000 #number of simulations
    discount_factor = np.exp(-r*T) #discount factor

    S = np.zeros([i,N])
    t = range(0,N,1)

    for y in range(0,i-1):
        S[y,0]=S0
        for x in range(0,N-1):
            S[y,x+1] = S[y,x]*(1 + r*deltat + sigma*np.random.normal(0,deltat))
        plt.plot(t,S[y])

    # Display Geometric Method Brownian Simulation
    plt.title('Simulations %d Steps %d Sigma %.2f r %.2f S0 %.2f' % (i, N, sigma, r, S0))
    plt.xlabel('Steps')
    plt.ylabel('Stock Price')
    plt.show()

def GMB(S0, K, r, v, T):

    N = 100 #number of steps within each simulation
    deltat = T/N #time step
    
    i = 1000 #number of simulations
    discount_factor = np.exp(-r*T) #discount factor

    S = np.zeros([i,N])

    for y in range(0,i-1):
        S[y,0]=S0
        for x in range(0,N-1):
            S[y,x+1] = S[y,x]*(1 + r*deltat + v*np.random.normal(0,deltat))

    C = np.zeros((i-1,1), dtype=np.float16)
    for y in range(0,i-1):
        C[y]=np.maximum(S[y,N-1]-K,0)

    CallPayoffAverage = np.average(C)
    CallPayoff = discount_factor*CallPayoffAverage
    return CallPayoff

# Standard normal probability density function
def norm_pdf(x):
    return (1.0/((2*pi)**0.5))*exp(-0.5*x*x)

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
