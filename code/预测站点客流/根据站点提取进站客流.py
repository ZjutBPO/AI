import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)

def getData(Station,InOut):
    df = pd.read_csv("data/total2.csv")

    df = df.loc[df[InOut + "StationName"] == Station]
    df[InOut + "Time"] = df[InOut + "Time"].map(lambda item:item[0:10])
    df.insert(0,"num",0)
    df = df.groupby(InOut + "Time",as_index=False).count().iloc[:,[0,1]]
    df.rename(columns={InOut + "Time":"Date"}, inplace=True)
    # 与DateType外连接
    df = pd.merge(df,date,how="outer",on="Date")
    df = df.sort_values(by="Date")
    df = df.fillna(0)
    df.rename(columns={"Date":"date","num":"num","DateType":"dateType"}, inplace=True)
    df.to_csv("./data/{}-{}.csv".format(Station,InOut),index=0)
    return df

def showPlt(df):
    x_label = df["date"].to_numpy()
    y_label = df["num"].to_numpy()
    plt.plot(x_label,y_label)
    plt.gca().xaxis.set_major_locator(MultipleLocator(10))
    plt.show()

date = pd.read_csv("./data/Date.csv")

Station = "Sta126"

df1 = getData(Station,"In")
df2 = getData(Station,"Out")

showPlt(df1)