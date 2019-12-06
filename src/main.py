from brownian_motion import analyze, GMB, display_GMB,vanilla_call_price,vanilla_put_price

# Main function
if __name__ == "__main__":
    
    # Get mean and stdev
    data = analyze('AAPL.csv')
    mean = data[0]
    stdev = data[1]
    variance = data[2]

    # Simulation variables
    S = 100         # initial stock price
    K = 100         # strike price
    r = mean        # risk-free interest rate
    v = stdev       # volatility in market
    T = 1           # time in years
    N = 100         # number of steps within each simulation
    deltat = T/N    # time step
    

    # Draw Geometric Method Brownian Simulation
    #display_GMB(mean,stdev)


    CallPayoff = GMB(S,K,r,v,T)

    call = vanilla_call_price(S,K,r,v,deltat)
    put = vanilla_put_price(S,K,r,v,deltat)
    
    print('CallPayoff: ' + str(CallPayoff) + '\n'\
          'Call:       ' + str(call) + '\n'\
          'Put:        ' + str(put) )
