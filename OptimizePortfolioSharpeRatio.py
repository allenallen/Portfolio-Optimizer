import numpy as np
import scipy.optimize as op

import utils


def __variance(L,allocation,sd,corr):
    var = 0.0
    for i in xrange(L):
        for j in xrange(L):
            var += (allocation[i] * allocation[j] * sd.ix[i] * sd.ix[j] * corr[i+1]) ** 2
    return var


def __get_sharpe_ratio(allocation,  df, symbol, start_value,corr):
    dr = utils.daily_returns(df)
    # corr = np.asarray(utils.get_correlation(dr))
    sd = dr.std()

    dr = dr.mean()

    var = __variance(len(symbol),allocation,sd,corr)

    df_normed = df / df.ix[0, :]

    df_alloced = df_normed * allocation
    pos_vals = df_alloced * start_value
    portfolio_value = pos_vals.sum(axis=1)

    return (np.sum(dr * allocation)/np.sqrt(var)) * -1


def optimize_for_sharpe_ratio(allocation, start_value, df, symbols):
    symbols = symbols[1:]
    dr = utils.daily_returns(df)
    df = df[symbols]
    # print df
    corr = np.asarray(utils.get_correlation(dr))
    corr = np.asarray([1,-.12,.00209,-.004,-.007,.005])
    # print symbols
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
    bnds = tuple((0, 1) for item in allocation)
    result = op.minimize(__get_sharpe_ratio, allocation, args=(df, symbols, start_value,corr), method='SLSQP', bounds=bnds,
                         constraints=cons)

    print "Correlation to SPY: ", corr[1:]

    return np.round(result.x, 2)
