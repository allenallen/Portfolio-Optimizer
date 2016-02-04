import pandas as pd
import scipy

import utils

symbols = ['IBM','AAPL','GLD','XOM']
L = len(symbols)
addressable_dates = pd.date_range('2010-02-01', '2010-12-31')
df = utils.get_data(symbols, addressable_dates)
df = df.dropna()

daily_ret = utils.daily_returns(df)
daily_ret_mean = scipy.array(daily_ret.mean())

std = utils.std_daily_return(daily_ret)

stds = scipy.array(std)
print stds

print symbols
print daily_ret_mean

corr = scipy.array(daily_ret.corr())



# cor = scipy.corrcoef(x)
# print cor



def main():

    return
# def sharpe_ratio(weights):
#     var = portfolio_variance(weights)
#     returns = scipy.array(exp_returns)
#     return (scipy.dot(weights,return) - r) / math.sqrt(var)


def portfolio_variance(a):
    '''Returns the variance of the portfolio with weights a.'''
    var = 0.0
    # to speed up sum covariance over i < j and variance over i
    for i in xrange(L):
        for j in xrange(L):
            var += a[i]*a[j]*std_dev[i]*std_dev[j]*cor[i, j]
    if var <= 0: # floating point errors for very low variance
        return 10**(-6)
    return var


def sharpe_ratio(weights):
    '''Returns the sharpe ratio of the portfolio with weights.'''
    var = portfolio_variance(weights)
    returns = scipy.array(daily_returns)
    return (scipy.dot(weights, returns) - r)/sqrt(var)

def sharpe_optimizer(weights):
    # for optimization - computes last weight and returns negative of sharpe
    # ratio (for minimizer to work)
    weights = scipy.append(weights, 1 - sum(weights)) # sum of weights = 1
    return - sharpe_ratio(weights)



main()

