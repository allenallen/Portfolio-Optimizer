import math

import numpy as np
import scipy.optimize as op

import utils


def __get_sharpe_ratio(allocation,  df, symbol, start_value):
    df_normed = df / df.ix[0, :]

    df_alloced = df_normed * allocation
    pos_vals = df_alloced * start_value
    # portfolio_value = pos_vals.sum(axis=1)
    #
    port_daily_ret = utils.daily_returns(pos_vals)

    portfolio_daily_return_mean = port_daily_ret.mean()
    portfolio_standard_deviation = port_daily_ret.std()

    # plt.scatter(portfolio_daily_return_mean,portfolio_standard_deviation)
    # plt.show()

    # print portfolio_daily_return_mean,portfolio_standard_deviation
    sharpe = ((math.sqrt(252) * (portfolio_daily_return_mean / portfolio_standard_deviation)) * -1)
    print "Allocations: ", np.round(allocation, 2)
    # print "Portfolio Value: ", portfolio_value.tail(5)
    print "Sharpe Ratio: ", sharpe * -1
    print "symbols: ", symbol
    return sharpe


def optimize_for_sharpe_ratio(allocation, start_value, df, symbols):
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
    bnds = tuple((0, 1) for item in allocation)
    result = op.minimize(__get_sharpe_ratio, allocation, args=(df, symbols, start_value), method='SLSQP', bounds=bnds,
                         constraints=cons)
    return np.round(result.x, 2)
