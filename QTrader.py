from turtle import update
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import random 
from collections import defaultdict



class QTrades(object):
    def __init__(self):
        '''
        get 12 data points to create bolinger bands 
        What is high average
        What is low average
        If returns is greater than high average  then state--> 1
        If returns is lower than low average  then state --> -1
         '''

        self.returns = pd.read_csv('returns.csv',parse_dates=['Date'],index_col='Date')
        self.returns['risk_adjusted_moving'] = self.returns.risk_adjusted.rolling(window=12).apply(lambda x:x.mean())
        self.returns['risk_adjusted_stdev'] = self.returns.risk_adjusted.rolling(window=12).apply(lambda x:x.std())
        self.returns['risk_adjusted_high'] = self.returns.risk_adjusted + 1.5 * self.returns['risk_adjusted_stdev']
        self.returns['risk_adjusted_low'] = self.returns.risk_adjusted - 1.5 * self.returns['risk_adjusted_stdev']
        self.returns['state'] = (self.returns.risk_adjusted > self.returns.risk_adjusted_high).astype(int) - \
                                (self.returns.risk_adjusted < self.returns.risk_adjusted_low).astype(int)

    def sharpe(self,expected_returns):
        return (( expected_returns - self.returns.loc[expected_returns.index,'risk_adjusted_moving'] ) / self.returns.loc[expected_returns.index,'risk_adjusted_stdev']).dropna()

    def buy_and_hold(self,dates):
        return pd.Series(1,index=dates)
    
    def buy_tbills(self,dates):
        return pd.Series(0,index=dates)

    def random(self,dates):
        return pd.Series(np.random.randint(-1,2, size=len(dates)),index=dates)

    def evaluate(self,holdings):
        return pd.Series(self.returns.TBills + holdings * (self.returns.risk_adjusted) + 1, index=holdings.index).cumprod()

    def q_holdings(self,training_indexes,testing_indexes):
        factors = pd.DataFrame({'action':0, 'reward':0, 'state':0}, index=training_indexes)
        
        # Value iteration
        # iterate over state ,action ,reward and new state/ state prime 
        # 0 - Buy, 1 - Short -1 - Do nothing
        # Q learning will learn the Q table
        q = {0: { 1:0, 0:0, -1:0}}
        
        # iterate over training set for 100 episodes
        for i in range(100):
            last_row, last_date = None,None

            for date, row in factors.iterrows():
                return_data = self.returns.loc[date]

                if return_data.state not in q:
                    q[return_data.state] = {1:0, 0:0, -1:0}
                
                if last_row is None or np.isnan(return_data.state):
                    state =0
                    reward=0
                    action=0
                else:
                    state = int(return_data.state)
                    if random.random() > 0.001:
                        # {1:2, 0:0} --> 1
                        action = max(q[state],key=q[state].get)
                    else:
                        action = random.randint(-1,1)

                    reward = last_row.action + (return_data.Stocks - return_data.TBills)

                    factors.loc[date, 'reward'] = reward
                    factors.loc[date, 'action'] = action
                    factors.loc[date, 'state'] = return_data.state

                    # perform Q learning
                    alpha = 1
                    discount = 0.9

                    update = alpha * (factors.loc[date, 'reward'] + discount * max(q[row.state].values()) - q[state][action])

                    if not np.isnan(update):
                        q[state][action] += update 

                last_date,last_row = date, factors.loc[date]
        
            # converge the traing based on shap ratios
            sharpe = self.sharpe(factors.action)
            
            if sharpe.any() > 0.2:
                break

            print('For episode {} we get an internal shap ratio of {}'.format(i, sharpe))
        
        testing = pd.DataFrame({'action':0,'state':0},index=training_indexes)
        testing['state'] = self.returns.loc[training_indexes,'state']
        testing['action'] = testing['state'].apply(lambda state: max(q[state],key=q[state].get('value')))

        return testing.action


    def graph_portfolio(self):
        midpoint = int(len(self.returns.index)/2)
        training_indexes = self.returns.index[:midpoint]
        testing_indexes = self.returns.index[midpoint:]

        portfolios = pd.DataFrame({
            'buy_and_hold':self.buy_and_hold(training_indexes),
            'buy_tbills':self.buy_tbills(training_indexes),
            'random':self.random(training_indexes),
            'qtrader': self.q_holdings(training_indexes,testing_indexes)
        },index=training_indexes)

        portfolios_values = pd.DataFrame({
            'buy_and_hold':self.evaluate(portfolios.buy_and_hold),
            'buy_tbills':self.evaluate(portfolios.buy_tbills),
            'random':self.evaluate(portfolios.random),
            'qtrader': self.evaluate(portfolios.qtrader)
        },index=training_indexes)

        portfolios_values.plot()
        plt.annotate("Buy and hold sharpe ratio : {} \n Qtrader {}".format(self.sharpe(portfolios.buy_and_hold), 
        self.sharpe(portfolios.qtrader)),xy=(0.25,0.95),xycoords='axes fraction')
        plt.show()

if __name__ == '__main__':
    bot = QTrades()
    bot.graph_portfolio()



    
