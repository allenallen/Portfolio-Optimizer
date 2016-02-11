import numpy as np
import pandas as pd

import utils

list_df = []



def main():
    symbols = ['AC','ALI','BDO','BPI','DMC']
    dates = pd.date_range('2012-04-16','2013-06-13')
    c = ['Date','Close','5_day_Future_Return','Result','Prediction']

    # allocation = [ 0.21 , 0.18 , 0.19,  0.24 , 0.18]
    # allocation = [ 0.98 , 0.01 , 0.   , 0.01 , 0.  ]
    allocation = [.2,.2,.2,.2,.2]
    start_value = 1000000

    __calculate_profit_and_loss(symbols,dates,c,allocation,start_value)

    # print "Long Profit: ",np.sum([np.sum(x['Long Profit'].values) for x in list_df[:]])
    # print "Long Loss: ",np.sum([np.sum(x['Long Loss'].values) for x in list_df[:]])
    # print "Short Profit: ",np.sum([np.sum(x['Short Profit'].values) for x in list_df[:]])
    # print "Short Loss: ",np.sum([np.sum(x['Short Loss'].values) for x in list_df[:]])

    long_profit,long_loss,short_profit,short_loss = __sum_of_profit_and_loss()
    # print list_df
    print long_profit,long_loss,short_profit,short_loss
    print abs(long_profit/long_loss)

    # dfs = np.asarray(list_df)
    # print dfs

    # ac_value = allocation[0] * start_value
    #
    # df = utils.get_data_w_columns(symbols,dates,c)
    # df = utils.get_all_fridays(df)
    # # df = df[(df['Result'] == True) & (df['Prediction'] == 1)]
    # df["Long Profit"] = calculate_profit_long(df,ac_value)
    # df["Long Loss"] = calculate_loss_long(df,ac_value)
    # df["Short Profit"] = calculate_profit_short(df,ac_value)
    # df["Short Loss"] = calculate_loss_short(df,ac_value)
    # df = df.fillna(0)
    # print df


def __sum_of_profit_and_loss():
    df_temp = np.asarray(list_df)
    # print df_temp
    # print df_temp
    # print df_temp[:][:,0]
    # print np.sum(df_temp[:][:,0])
    long_p = np.sum(df_temp[:][:,0])
    long_l = np.sum(df_temp[:][:,1])
    short_p = np.sum(df_temp[:][:,2])
    short_l = np.sum(df_temp[:][:,3])
    return long_p,long_l,short_p,short_l

def __calculate_profit_and_loss(symbols,date_range,columns,allocation,start_value):
    alloc_index = 0
    for ticker in symbols:
        ticker_value = allocation[alloc_index] * start_value
        df_ticker = utils.get_data_w_columns([ticker],date_range,columns)
        df_ticker = utils.get_all_fridays(df_ticker)
        # print df_ticker
        df_ticker["Long Profit"] = calculate_profit_long(df_ticker,ticker_value)
        df_ticker["Long Loss"] = calculate_loss_long(df_ticker,ticker_value)
        df_ticker["Short Profit"] = calculate_profit_short(df_ticker,ticker_value)
        df_ticker["Short Loss"] = calculate_loss_short(df_ticker,ticker_value)
        df_ticker = df_ticker.fillna(0)
        list_df.append([df_ticker["Long Profit"].sum(),df_ticker["Long Loss"].sum(),df_ticker["Short Profit"].sum(),df_ticker["Short Loss"].sum()])
        alloc_index += 1


def calculate_loss_long(df,start_value):
    # print df
    df = df[(df['Result'] == False) & (df['Prediction'] == 1)]
    df_copy = df.copy()
    # print df
    value = start_value / df['Adj Close']
    # print value
    value = value * df['5_day_Future_Return']
    # print value
    df_copy['Long Loss'] = value
    return df_copy['Long Loss']


def calculate_profit_long(df,start_value):
    # print df
    df = df[(df['Result'] == True) & (df['Prediction'] == 1)]
    df_copy = df.copy()
    # print df
    value = start_value / df['Adj Close']
    # print value
    value = value * df['5_day_Future_Return']
    # print value
    df_copy['Long Profit'] = value
    return df_copy['Long Profit']


def calculate_loss_short(df,start_value):
     # print df
    df = df[(df['Result'] == True) & (df['Prediction'] == 0)]
    df_copy = df.copy()
    # print df
    value = start_value / df['Adj Close']
    # print value
    value = value * df['5_day_Future_Return']
    # print value
    df_copy['Short Profit'] = value
    return df_copy['Short Profit']


def calculate_profit_short(df,start_value):
     # print df
    df = df[(df['Result'] == False) & (df['Prediction'] == 0)]
    df_copy = df.copy()
    # print df
    value = start_value / df['Adj Close']
    # print value
    value = value * df['5_day_Future_Return']
    # print value
    df_copy['Short Loss'] = value
    return df_copy['Short Loss']


main()
