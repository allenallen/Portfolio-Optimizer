import numpy as np
import pandas as pd

list_df = []

def main():
    df1 = pd.DataFrame([1,2,3,4,5])
    df2 = pd.DataFrame([1,2,3,4,5])
    df3 = pd.DataFrame([1,2,3,4,5])
    df4 = pd.DataFrame([1,2,3,4,5])
    list_df.append(df1)
    list_df.append(df2)
    list_df.append(df3)
    list_df.append(df4)
    print np.sum([x.ix[1] for x in list_df[:]])

main()