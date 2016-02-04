import pandas as pd

import utils


def main():
    symbols = ['AC_SL']
    dates = pd.date_range('2010-01-01','2016-12-31')
    df = utils.get_data(symbols,dates)
    c = ['Date','Adj Close','5_day_Future_Return','Result','Prediction']

    allocation = [ 0.21 , 0.18 , 0.19,  0.24 , 0.18]
    start_value = 1000000

    ac_value = allocation[0] * start_value

    df = utils.get_data_w_columns(symbols,dates,c)
    df = utils.get_all_fridays(df)
    # df = df[(df['Result'] == True) & (df['Prediction'] == 1)]
    df["Long Profit"] = calculate_profit_long(df,ac_value)
    df["Long Loss"] = calculate_loss_long(df,ac_value)
    df["Short Profit"] = calculate_profit_short(df,ac_value)
    df["Short Loss"] = calculate_loss_short(df,ac_value)
    df = df.fillna(0)
    print df


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