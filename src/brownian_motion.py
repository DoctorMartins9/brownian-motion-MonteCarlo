# Library for handling csv files
import csv
import math
from statistics import mean,variance,stdev
import matplotlib.pyplot as plt
import numpy as np


# File csv da salvare nella cartella data per poter essere analizzato
file_name = 'AAPL.csv'

def getRendimento():
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
    mu_annuo = []
    for r in rendimento:
        mu_annuo.append(r * 252)

    # Calcolo di media e varianza
    media = mean(mu_annuo)
    deviation = stdev(mu_annuo)

    return [media,deviation]

def moto_browniano_geometrico():
    T = 2
    mu = 0.1
    sigma = 0.01
    S0 = 20
    dt = 0.01
    N = round(T/dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N) 
    W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
    X = (mu-0.5*sigma**2)*t + sigma*W 
    S = S0*np.exp(X) ### geometric brownian motion ###
    plt.plot(t, S)
    plt.show()