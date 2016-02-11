import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from SMAMomentumIndicator import SMAMomentumIndicator

###begin point of code

#base_dir="//Users//clydemoreno//Downloads//data//DataLocal//"
base_dir="//Users//clydemoreno//Downloads//data//"
addressable_dates = pd.date_range('2012-01-01','2016-01-01')
initial_training_dates = pd.date_range('2012-05-01','2013-5-31')
#initial_test_date = pd.date_range('2012-12-01','2013-12-31')
initial_actual_test_date = pd.date_range('2013-06-01','2013-12-31')
sma_period = 100
std_dev_period = 63
t_bill = 0.23



def symbol_to_path(symbol, base_dir="//Users//clydemoreno//Downloads//data//"):
    """Return CSV file path given ticker symbol."""

    #return os.path.join(base_dir, "{}.csv".format(str(symbol)))
    return "{}{}.csv".format(base_dir,str(symbol))

def add_ticker_column(df,ticker):
    df['Ticker'] = pd.Series([0 for x in range(len(df.index))])
    df['Ticker'] = ticker

def get_data(symbols, dates):

    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates,)
    if ('SPY' not in symbols):  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:

        df_temp = pd.read_csv(symbol_to_path(symbol,base_dir=base_dir), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close','Volume'], na_values=['nan'])


        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade

            df = df.rename(columns={'Adj Close': 'BasePair_Close','Volume':'BasePair_Volume'})
            df = df.dropna(subset=["BasePair_Volume"])
        else:
            df = df.rename(columns={'Adj Close': 'Close'})

            #df_temp = df_temp.rename(columns={'Adj Close': 'Close'})
            #df = df_temp


    return df



# def get_data2(symbols, dates):
#     """Read stock data (adjusted close) for given symbols from CSV files."""
#     df = pd.DataFrame(index=dates)
#     if 'SPY' not in symbols:  # add SPY for reference, if absent
#         symbols.insert(0, 'SPY')
#
# #handle the multiple symbols.  Not working.....
#     for symbol in symbols:
#         #print symbol
#         df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
#                 parse_dates=True, usecols=['Date', 'Adj Close','Volume'], na_values=['nan'])
#         if symbol == 'SPY':  # drop dates SPY did not trade
#             #df["SPY"] = df.dropna(subset=["Close"])
#             df_temp["SPY"] = df_temp['Adj Close']
#             #print "here"
#         else:
#             df_temp["Last"] = df_temp['Adj Close']
#             df_temp = df_temp.rename(columns={'Volume': 'Vol'})
#         df = df.join(df_temp)
#         #add_ticker_column(df,symbol)
#         #print df
#
#
#     return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # plt.show()

def compute_future_returns(df,period):
    """Compute and return the daily return values."""

    #formula for future price (5 days to the future is -4 on zero based index array
    df_temp = (df.shift(period) / df) - 1

    return df_temp

def main_run2():
    dates = pd.date_range('2014-01-01', '2015-11-30')
    #symbols = ['SPY','AAPL', 'MSFT']
    symbols = ['MSFT']
    df = get_data(symbols, dates)




def apply_sma_volume(df):
    sma = pd.rolling_mean(df['Volume'],window=sma_period)
    df['SMA_Volume'] = sma

    sma = pd.rolling_mean(df['BasePair_Volume'],window=sma_period)
    df['BasePair_Volume_SMA'] = sma

    return df


def apply_sma(df):
    training_sma = pd.rolling_mean(df['Close'],window=sma_period)
    df['SMA'] = training_sma

    training_sma = pd.rolling_mean(df['BasePair_Close'],window=sma_period)
    df['BasePair_Close_SMA'] = training_sma

    return df

def apply_sma_to_price_ratio(df):
    df['SMA_To_Price_Ratio'] = (df["SMA"] / df['Close'])  - 1
    df['BasePair_SMA_To_Price_Ratio'] = (df["BasePair_Close_SMA"] / df['BasePair_Close'])  - 1
    return df


def apply_sma_vol_to_volume_ratio(df):
    df['BasePair_SMA_Vol_To_Price_Ratio'] = (df["BasePair_Volume_SMA"] / df['SMA_Volume'])  - 1
    return df

def apply_future_returns(df):

    df["Daily_Return"] = compute_future_returns(df['Close'],-1)
    df["5_day_Future_Return"] = compute_future_returns(df['Close'],-4)
    df["5_day_Discreet_Actual_Y"] = np.round(df["5_day_Future_Return"] * 1000,0)
    df["10_day_Future_Return"] = compute_future_returns(df['Close'],-9)
    df["10_day_Discreet_Actual_Y"] = np.round(df["10_day_Future_Return"] * 1000,0)
    return df

def apply_sma_momentum(df,day):
    sp500MomIndicator1 = SMAMomentumIndicator(data=df['Close'], period=100,day=day)

    df['1_Day_SMA_Momentum'] = sp500MomIndicator1.data

    # sp500MomIndicator = SMAMomentumIndicator(data=df['Close'], period=100,day=-5)
    #
    # df['5_Day_SMA_Momentum'] = sp500MomIndicator.data


    # sp500MomIndicator = SMAMomentumIndicator(data=df['Close'], period=100,day=-5)
    # df['5_Day_SMA_Momentum'] = sp500MomIndicator.data
    #
    # sp500MomIndicator = SMAMomentumIndicator(data=df['Close'], period=100,day=-10)
    # df['10_Day_SMA_Momentum'] = sp500MomIndicator.data
    return df

def apply_std_dev(df):
    df["Volatility"] = np.abs( pd.rolling_std(df['Daily_Return'],window=63))
    return df

def apply_sharpe_ratio(df):
    avg_daily_return = np.abs( pd.rolling_mean(df['Daily_Return'],window=63))
    avg_daily_risk = df["Volatility"]
    df["Sharpe_Ratio"] = (avg_daily_return - (t_bill)/ avg_daily_risk)
    return df
def process(ticker):
    #get data
    #dates = pd.date_range(addressable_dates)



    #symbols = ['SPY','AAPL', 'MSFT']
    symbols = [ticker]
    #print symbols
    #return
    df = get_data(symbols, addressable_dates)
    #print len(df)
    # dfbase =
    # print df
    df = apply_sma(df)
    # print len(df)
    df = apply_sma_to_price_ratio(df)
    # print len(df)
    df = apply_future_returns(df)
    # print len(df)
    df = apply_sma_momentum(df,-1)
    df = apply_std_dev(df)
    df = apply_sharpe_ratio(df)
    df = df.dropna()
  
    #todo: calculate sharpe ratio
    # get rolling mean 63 periods of daily returns
    # get the sdd of daily returns

    #modify this to apply all technical calculations on all addressable space
    #then just get subset for training data and test data.

    training_df = df.ix[initial_training_dates];
    training_df = training_df.dropna()

    training_x = training_df[['Volume','Close','SMA','SMA_To_Price_Ratio','1_Day_SMA_Momentum']]

    training_non_disc_y = training_df['5_day_Future_Return']
    training_y = training_df["5_day_Discreet_Actual_Y"]

    clf = GaussianNB()

    clf.fit(training_x,training_y)

    backtest_df = df.ix[initial_actual_test_date];
    backtest_df = backtest_df.dropna()

    backtest_x = backtest_df[['Volume','Close','SMA','SMA_To_Price_Ratio','1_Day_SMA_Momentum']]

    predicted_y = clf.predict(backtest_x)

    backtest_actual_y = backtest_df['5_day_Discreet_Actual_Y']

    product = predicted_y * backtest_actual_y
    bool_product = product > 0

    i = 0
    count = 0
    for y in predicted_y:

        if(y == 0) and (backtest_actual_y[i] == 0):
           product[i] = True
        #print y, backtest_actual_y[i],product[i] > 0

        i = i + 1
    num = np.count_nonzero(bool_product)
    total = len(product)
    accuracy = float(num) / float(total)
    #ACCURACY CALCULATION.  if abs(prediction) is more than the abs(actual) then it should be tagged as false

    print ticker, "wins:", num,"total trades:", total, "Accuracy:", round(accuracy * 100,2), "%"


def train(clf, df,y):
    clf.fit(df,y)


def main_run():
    for ticker in ['AAPL', 'MSFT','ORCL','QCOM','BBY','MU','GILD','YUM','NFLX','VZ','APA','RRC','MDLZ','CSCO','V','MET','SBUX','GGP','UA','GM']:
    # for ticker in ['AAPL']:
        process(ticker)

def main_run3():


    #['AC', 'ALI','BDO','BPI','DMC','GLO','JFC','MBT','MEG','MPI','RLC','SECB','SM','SMPH','TEL','URC']
    for ticker in ['AC', 'ALI','BDO','BPI','DMC','GLO','JFC','MBT','MEG','MPI','RLC','SECB','SM','SMPH','TEL','URC']:
    # for ticker in ['AC']:
        print "******************************************"
        process(ticker)
        #iterate_dates(ticker,addressable_dates,initial_training_dates,initial_test_date,initial_actual_test_date)
        print "******************************************"


if __name__ == "__main__":
    main_run()
