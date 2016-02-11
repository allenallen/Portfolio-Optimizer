import sys
sys.path.insert(1, '/Library/Python/2.7/site-packages')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import scipy.optimize as spo
import utils
import Accuracy


# y1 = []
def error(h,args):


    x = args[0] #parameters
    y = args[1] #returns
    z = args[2] #covariance or correlation
    b = args[3] #bias (going long or short)
    p = args[4] #accuracy

    # print "z",args[2]
    y1 = np.transpose( h) * y
    z1 = np.sum((np.transpose(h) * z) ** 2)  #cov
    b1 = np.sum((np.transpose(h) * b) ** 2) #really penalize un-equal weights of long and short
    p1 = 1 / (np.sum(np.transpose(h) * p) ** 2)

    y_main = func(np.transpose(x),5,1,-5.0)
    plt.scatter(x,y1,color='red')

    # print "ymain",y_main
    # print "func",x, func(xdata,5,1,-5.0)
    # cost = np.sum( (func(x,5,1,-5.0) - (h * y)) ** 2)
    # print "Hypotheses", np.round(h,2)
    cost = np.sum( (y_main - y1) ** 2) * z1 * b1 * p1
    # print func(x,5,1,-5.0),h * y
    # cost = 0
    # print cost
    return cost

def sigmoid(x,a,b,c):
    z = a * (1 / (1 + np.exp(-x * b))) + c
    # print "{}".format(np.round(z ,6))
    return

def algebraic(x,a,b,c):
    z = a *  ( x / (np.sqrt(1 + np.exp(2)))) + c
    print np.sum( b * x)
    return z

# def algebraic(x,a,b,c):
#     return a * np.exp(-b * x) + c


def func(x,a,b,c):
    return -1 * (a * np.exp(-b * x) + c)





def main_run():

    # h = [.01,  .01,   .01,.01,  .01,.01,  .01,.01]
    # # h = pd.Series([0.0025,0.0025,0.0025,0.0025,0.0025,0.0025,0.0025,0.0025],dtype=float)
    #
    # # x = [3.,  2.,   1.5,2.,  1.5,3.,  4.5,4.9]
    # # y = [4.2,1.9,1.9,2.1,2.0  ,2.5,3.9,4.5]
    #
    # x = [1.5,  2.,   1.5,2.,  1.5,3.,  4.5,4.9]
    # y = [4.2,1.9,1.9,2.1,2.0  ,2.5,3.9,4.5]
    # z = [0.012,0.032,0.035,0.902,0.052,0.302,-0.909,0.902]
    # # df.corr()
    # b = [1.,  -1.,   -1.,1.,  1.,-1.,  1,-1]
    # p = [0.12,0.55,0.45,0.3,0.53,0.57,0.19,0.58]

    start_value = 1000000
    # symbols = ['AC','ALI','BDO','BPI','DMC','GLO',
    #            'GTCAP','HCP','JFC','MBT','MPI','MEG',
    #            'RLC','SECB','SM','SMPH','TEL','URC']
    # symbols = ['AAPL','IBM','XOM','GLD']
    # symbols = ['goog','aapl','msft','amzn']
    symbols = ['AAPL', 'MSFT','ORCL','QCOM','BBY','MU','GILD','YUM','NFLX','VZ','APA','RRC','MDLZ','CSCO','V','MET','SBUX','GGP','UA','GM']
    h = np.ones(len(symbols))

    addressable_dates = pd.date_range('2012-01-01','2015-12-31')
    df = utils.get_data_online(symbols, addressable_dates)
    df = pd.DataFrame(df,index=df.index,dtype=np.float64)
    df = df.dropna()

    x = np.asarray(utils.daily_returns(df[symbols[1:]]).std())
    # print x
    y = np.asarray(utils.daily_returns(df[symbols[1:]]).mean())
    # print y
    # b = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]
    z = np.asarray(utils.get_correlation(df)[1:])
    # print z
    # p = [.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5]
    b = np.ones(len(symbols) - 1)
    # print b
    p = Accuracy.get_accuracy(symbols,addressable_dates)
    # print p
    sharpe_ratios = utils.get_sharpe_ratio(utils.daily_returns(df[symbols[1:]]))
    # print sharpe_ratios

    data = (x,y,z,b,p)


    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    # print "h",np.transpose( h) * y

    # print "cost", error(h)

    # cons = ({'type':'eq', 'fun': lambda x: 1 - sum(x)})
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})

    # bounds = tuple((0,1) for it in h)
    bnds = tuple((0,1) for it in h)
    bounds = bnds
    # bounds = (0,1)

    min_result = spo.minimize(error,h,args=(data,), method='SLSQP',bounds=bounds, constraints=cons, options={'disp':True})

    print "Parameters = {}, Y = {}".format(np.round(min_result.x,2), np.abs( min_result.fun)) # for tracing

    # print "xdata",xdata
    model_bounds = max(x.max(),y.max())
    xdata = np.linspace(0,5,8)

    y_main = func(xdata,model_bounds,1,-model_bounds)


    print "bias", b * np.round(min_result.x,3)
    print "risk: ", x
    print "rewards: ", y
    print "correlation: ", z
    print "accuracy: ", p

    print "y main",y_main
    plt.plot(xdata,y_main)


    y = func(xdata,2.5,1.3,0.5)
    ydata = y + 0.2 * np.random.normal(size=len(xdata))


    coeffs, pcov = curve_fit(func,xdata,ydata)

    yaj = func(xdata, coeffs[0], coeffs[1], coeffs[2])

    print pcov


    # plt.scatter(xdata,ydata)
    # plt.plot(xdata,yaj)

    # c,p = curve_fit(sigmoid,x_sorted,y_sorted)


    plt.scatter(x_sorted,y_sorted)



    plt.show()





    return None


if __name__ == "__main__":
    list_df = []
    df = pd.DataFrame()
    list_df.append(df)

    main_run()

