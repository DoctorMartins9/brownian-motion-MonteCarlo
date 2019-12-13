import quandl
import numpy as np
import matplotlib.pyplot as plt
from math import exp, sqrt

# Funzione che, dato il ticker, ritorna media, varianza e s0
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

# Funzione che simula con moto browniano geometrico e mostra il risultato in un grafico
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

    # Ricavo i dati da yahoo finance
    mu,sigma,s0 = getData(nome)                                       
    delta_t = 0.001                     
                        
    # Eseguo la simulazione
    GMB_plot(mu,sigma,s0,T,delta_t,num_reps)