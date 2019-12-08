from brownian import *
from tkinter import *

def w_GMB():
    # Window setting
    window = Tk()
    window.title("Simulazione moto Browniano geometrico")

    lbl_nome = Label(window, text="Ticker: ")
    lbl_nome.grid(column=0, row=0)
    txt_nome = Entry(window,width=15)
    txt_nome.grid(column=1, row=0)

    
    lbl_T = Label(window, text="Maturità: ")
    lbl_T.grid(column=0, row=1)
    txt_T = Entry(window,width=15)
    txt_T.grid(column=1, row=1)
    T = 3.0/12
    txt_T.insert(0,str(T))
    

    lbl_delta = Label(window, text="Delta t: ")
    lbl_delta.grid(column=0, row=2)
    txt_delta = Entry(window,width=15)
    txt_delta.grid(column=1, row=2)
    txt_delta.insert(0,'0.001')

    lbl_reps = Label(window, text="Delta t: ")
    lbl_reps.grid(column=0, row=3)
    txt_reps = Entry(window,width=15)
    txt_reps.grid(column=1, row=3)
    txt_reps.insert(0,'100')

    def clk_simula():  
        mu,sigma,s0 = getData(txt_nome.get())
        GMB_plot(mu,sigma,s0,float(txt_T.get()),float(txt_delta.get()),int(txt_reps.get()))

    btn = Button(window, text="Calcola", bg="orange", fg="red",command=clk_simula)
    btn.grid(column=1, row=4)

    window.mainloop()



def w_info():
    # Window setting
    window = Tk()
    window.title("Dati")
    
    lbl = Label(window, text="Inserisci ticker: ")
    lbl.grid(column=0, row=0)

    txt = Entry(window,width=15)
    txt.grid(column=1, row=0)

    lbl_name0 = Label(window, text="Nome:")
    lbl_name0.grid(column=0, row=1)
    lbl_name = Label(window, text="")
    lbl_name.grid(column=1, row=1)


    lbl_mu = Label(window, text="Media:")
    lbl_mu.grid(column=0, row=2)
    lbl_mu = Label(window, text="")
    lbl_mu.grid(column=1, row=2)

    lbl_sigma = Label(window, text="Varianza:")
    lbl_sigma.grid(column=0, row=3)
    lbl_sigma = Label(window, text="")
    lbl_sigma.grid(column=1, row=3)

    lbl_s0 = Label(window, text="S0:")
    lbl_s0.grid(column=0, row=4)
    lbl_s0 = Label(window, text="")
    lbl_s0.grid(column=1, row=4)

    lbl_strike = Label(window, text="Strike:")
    lbl_strike.grid(column=0, row=4)
    lbl_strike = Label(window, text="")
    lbl_strike.grid(column=1, row=4)


    def clk_name():
        c = Call(txt.get(), d=13, m=12, y=2019)
        strike_call = c.strikes[len(c.strikes)-1]  
        mu,sigma,s0 = getData(txt.get())

        lbl_name.configure(text = txt.get())
        lbl_mu.configure(text = str(mu))
        lbl_sigma.configure(text = str(sigma) )
        lbl_s0.configure(text = str(s0) )
        lbl_strike.configure(text = str(strike_call) )

    btn = Button(window, text="Calcola", bg="orange", fg="red",command=clk_name)
    btn.grid(column=2, row=0)

    window.mainloop()


if __name__ == "__main__":
    # Window setting
    window = Tk()
    window.title("Analisi di opzioni")
    btn_dati = Button(window, text="Dati statistici", bg="orange", fg="red",command=w_info)
    btn_dati.grid(column=0, row=0)

    btn_gmb = Button(window, text="Simulazione moto Browniano geometrico", bg="orange", fg="red",command=w_GMB)
    btn_gmb.grid(column=0, row=1)

    window.mainloop()