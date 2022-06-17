import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import random 
from collections import defaultdict



class QTrades(object):
    def __init__(self):
        self.returns = pd.read_csv('returns.csv',parse_dates=['Date'],index_col='Date')
        

    def buy_and_sell(self,dates):
        return pd.Series(1,index=dates)
    
    def buy_tbills(self,dates):
        return pd.Series(0,index=dates)

    def random(self,dates):
        return pd.Series(np.random.randint(-1,2, size=len(dates)),index=dates)

    def evaluate(self,holdings):
        return pd.Series(self.returns.TBills + holdings * (self.returns.risk_adjusted) + 1, index=holdings.index).cumprod()

    def graph_portfolio(self):
        midpoint = int(len(self.returns.index)/2)
        training_indexes = self.returns.index[:midpoint]
        testing_indexes = self.returns.index[midpoint:]

        portfolios = pd.DataFrame({
            'buy_and_sell':self.evaluate(self.buy_and_sell(training_indexes)),
            'buy_tbills':self.evaluate(self.buy_tbills(training_indexes)),
            'random':self.evaluate(self.random(training_indexes)),
            'qtrader': self.evaluate(self.q_holdings(training_indexes,testing_indexes))
        },index=training_indexes)

        portfolios.plot()
        plt.show()

if __name__ == '__main__':
    bot = QTrades()
    bot.graph_portfolio()



    
