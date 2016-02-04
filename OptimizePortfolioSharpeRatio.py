import numpy as np
import scipy.optimize as op

import utils


def variance(L,allocation,sd,corr):
    var = 0.0
    # print sd
    for i in xrange(L):
        for j in xrange(L):
            # print "sd[{}][{}]: {}{}".format(i,j,sd.ix[i],sd.ix[j])
            var += (allocation[i] * allocation[j] * sd.ix[i] * sd.ix[j]) ** 2 #* corr.ix[i][j]) ** 2
    if var <= 0: # floating point errors for very low variance
        return 10**(-6)
    return var


def __get_sharpe_ratio(allocation,  df, symbol, start_value):
    dr = utils.daily_returns(df)
    sd = dr.std()

    dr = dr.mean()

    corr = df.corr()

    var = variance(len(symbol),allocation,sd,corr)

    df_normed = df / df.ix[0, :]

    df_alloced = df_normed * allocation
    pos_vals = df_alloced * start_value
    portfolio_value = pos_vals.sum(axis=1)
    # print "portfolio value: ", portfolio_value
    print (dr * allocation)/np.sqrt(var)
    return (np.sum(dr * allocation)/np.sqrt(var)) * -1

    # print corr

    # df_normed = df / df.ix[0, :]
    #
    # df_alloced = df_normed * allocation
    # pos_vals = df_alloced * start_value
    # portfolio_value = pos_vals.sum(axis=1)
    #
    # port_daily_ret = utils.daily_returns(portfolio_value)
    # #
    # # mean = pd.rolling_mean(port_daily_ret,window=2)
    # # risk = pd.rolling_std(port_daily_ret,window=2)
    # #
    # # mean = mean.dropna()
    # # risk = risk.dropna()
    # #
    # # plt.scatter(mean,risk)
    # # plt.show()
    #
    # portfolio_daily_return_mean = port_daily_ret.mean()
    # portfolio_standard_deviation = port_daily_ret.std()
    #
    # # print "daily ret: {}, std: {}".format(portfolio_daily_return_mean,portfolio_standard_deviation)
    #
    # sharpe = (math.sqrt(252) * (portfolio_daily_return_mean/ portfolio_standard_deviation)) * -1
    # print "Allocations: ", np.round(allocation, 2)
    # print "Portfolio Value: ", portfolio_value.tail(5)
    # print "Sharpe Ratio: ", sharpe * -1
    # print "symbols: ", symbol
    return sharpe


def optimize_for_sharpe_ratio(allocation, start_value, df, symbols):
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
    bnds = tuple((0, 1) for item in allocation)
    result = op.minimize(__get_sharpe_ratio, allocation, args=(df, symbols, start_value), method='SLSQP', bounds=bnds,
                         constraints=cons)
    return np.round(result.x, 2)
