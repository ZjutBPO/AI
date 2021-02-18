import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
import datetime

def groupbyUserId(df):
    
    df2 = df
    df2.insert(0,"num",0)

    # total按照用户分组
    df2 = df2.groupby(["UserID"]).count().iloc[:,[0]]
    # 筛选出在trip中记录num>300的UserID
    # print(df2[df2["num"] > 300])

    df2.insert(0,"人数",0)
    df2 = df2.groupby(["num"]).count().iloc[:,[0]]
    #将输出中的省略号去掉：需要更改默认设置

    print(df2)


def groupbydate(df):

    df = df.loc[:,["UserID","InTime"]]
    # 修改"InTime"为"Date"
    df.rename(columns={"InTime":"Date"},inplace = True)
    
    # # 得到Series对象，对其中的每一个元素调用函数（截取时间），再更新原数据
    df["Date"] = df["Date"].map(lambda item:item[5:10])

    # 按照进栈时间分组
    df2 = df.groupby(["Date"]).count()
    # 修改"UserID"为"num"
    df2.rename(columns={"UserID":"num"},inplace = True)

    # 统计每一天共有多少条数据
    num = df2["num"].to_numpy()
    date = df2.index.values

    # 一共有几天
    print("一共有" + str(len(num)) + "天")

    print(df2)

    plt.plot(date,num)
    # 设置横坐标间隔为10
    plt.gca().xaxis.set_major_locator(MultipleLocator(10))
    plt.show()

def groupbystation(df):
    
    df2 = df
    df2.insert(0,"num",0)

    # total按照InStationName分组
    df3 = df2.groupby(["InStationName"]).count().iloc[:,[0]]
    df3 = df3.sort_values(by="num")

    # print("每个站点总进站记录")
    # print(df3)

    df4 = df2.groupby(["OutStationName"]).count().iloc[:,[0]]
    df4 = df4.sort_values(by="num")

    # print("每个站点总出站记录")
    # print(df4)

    df5 = df3 + df4
    #设置切分区域
    listBins = np.arange(0,75000,1000)

    #设置切分后对应标签
    # listLabels = ['0_10','11_20','21_30','31_40','41_50','51_60','61及以上']

    groups=pd.cut(df5["num"],bins=listBins)

    df5 = df5.groupby(groups).count()
    # print(df5)
    df5 = df5.loc[(df5 != 0).all(axis=1),:]
    x_label = df5.index.values
    y_label = df5.to_numpy()

    x_label1 = []
    
    len = 0

    for i in x_label:
        x_label1.append(i.right)


    plt.plot(x_label1,y_label)
    plt.show()

pd.set_option('display.max_columns',None)  #设置列不限制数量
pd.set_option('display.max_rows',None)     #设置行不限制数量

df = pd.read_csv("data/total2.csv")
print("total中有" + str(df.shape[0]) + "数据")
# groupbyUserId(df)
# groupbydate(df)
groupbystation(df)