import numpy as np

import utils

#     # c = ['Date','Close','5_day_Future_Return','Result','Prediction']

'''returns long_profit, long_loss, short_profit, short_loss'''
def calculate_profit_and_loss(symbols,date_range,columns,allocation,start_value):
    alloc_index = 0
    list_df = []

    for ticker in symbols:
        ticker_value = allocation[alloc_index] * start_value
        df_ticker = utils.get_data_w_columns([ticker],date_range,columns)
        df_ticker = utils.get_all_fridays(df_ticker)
        # print df_ticker
        df_ticker["Long Profit"] = __calculate_profit_long(df_ticker,ticker_value)
        df_ticker["Long Loss"] = __calculate_loss_long(df_ticker,ticker_value)
        df_ticker["Short Profit"] = __calculate_profit_short(df_ticker,ticker_value)
        df_ticker["Short Loss"] = __calculate_loss_short(df_ticker,ticker_value)
        df_ticker = df_ticker.fillna(0)
        list_df.append([df_ticker["Long Profit"].sum(),df_ticker["Long Loss"].sum(),df_ticker["Short Profit"].sum(),df_ticker["Short Loss"].sum()])
        alloc_index += 1

    return __sum_of_profit_and_loss(list_df)


def __sum_of_profit_and_loss(list_df):
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


def __calculate_loss_long(df,start_value):
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


def __calculate_profit_long(df,start_value):
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


def __calculate_loss_short(df,start_value):
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


def __calculate_profit_short(df,start_value):
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

