import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import random
from deap import base, creator, tools, algorithms

stocks = [
    "AGRO.BA", "ALUA.BA", "AUSO.BA", "BBAR.BA", "BHIP.BA", "BMA.BA", "BPAT.BA",
    "BRIO.BA", "SUPV.BA", "BOLT.BA", "BYMA.BA", "CVH.BA", "CGPA2.BA", "CAPX.BA",
    "CADO.BA", "CELU.BA", "CECO2.BA", "CEPU.BA", "COME.BA", "INTR.BA", "CTIO.BA",
    "CRES.BA", "DOME.BA", "DYCA.BA", "EDN.BA", "FERR.BA", "FIPL.BA", "GARO.BA",
    "DGCU2.BA", "GBAN.BA", "GGAL.BA", "OEST.BA", "GRIM.BA", "VALO.BA", "HAVA.BA",
    "HARG.BA", "INAG.BA", "INVJ.BA", "IRSA.BA", "SEMI.BA", "LEDE.BA", "LOMA.BA",
    "LONG.BA", "METR.BA", "MIRG.BA", "MOLI.BA", "MORI.BA", "PAMP.BA", "PATA.BA",
    "POLL.BA", "RIGO.BA", "ROSE.BA", "SAMI.BA", "TECO2.BA", "TXAR.BA",
    "TRAN.BA", "TGNO4.BA", "YPFD.BA",
    "BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD", "XLM-USD", "ADA-USD",
    "DOGE-USD", "LINK-USD", "DOT-USD", "UNI-USD", "SOL-USD", "USDT-USD"
]

start_date = "2011-01-02"
end_date = datetime.now().strftime('%Y-%m-%d')

# Fetch historical stock data
data = yf.download(stocks, start=start_date, end=end_date)

data.interpolate(method='linear', inplace=True)

returns = data['Adj Close'].pct_change()

nStocks = len(stocks)
R = returns.mean()
S = returns.cov()
s = returns.std()

def weights(w):
    w = np.clip(w, 0, 1)
    return w / sum(w)

def portfolio_return(w):
    return sum(w * R)

def portfolio_volatility(w):
    return np.dot(w.T, np.dot(S, w))

portfolio_return(np.ones(nStocks))
portfolio_volatility(np.ones(nStocks))

def fitness(w):
    exp_return = portfolio_return(w)
    penalty = 0
    if exp_return < 0:
        penalty = (exp_return*100)**2
    return -(portfolio_volatility(w) + penalty)

# Create the necessary DEAP components
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator for individuals
toolbox.register("attr_float", random.uniform, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=nStocks)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the evaluation function
toolbox.register("evaluate", fitness)

# Register the genetic operators
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Ensure the weights sum to 1
def evaluate_and_correct(individual):
    
    return fitness(weights(np.float64(individual))),

toolbox.register("evaluate", evaluate_and_correct)

def run():
    # Seed for reproducibility
    random.seed(123)
    
    # Create the population
    pop = toolbox.population(n=300)
    
    # Number of generations
    ngen = 50
    
    # Probability of crossover
    cxpb = 0.7
    
    # Probability of mutation
    mutpb = 0.2
    
    # Run the genetic algorithm
    algorithms.eaSimple(pop, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=True)
    
    # Get the best individual
    best_ind = tools.selBest(pop, 1)[0]
    return best_ind


result = run()

w0 = weights(result)

fitness(weights(result))

print("Optimal weights:", w0)
print("Expected return:", portfolio_return(w0))
print("Expected volatility:", portfolio_volatility(w0))