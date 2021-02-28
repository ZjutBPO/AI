import pandas as pd
import numpy as np

total = pd.read_csv("data/total2.csv")
covid = pd.read_csv("用户画像/date-num-COVID-diff.csv")

# total = total.head()
# 新增一列Date，提取出InTime的年月日
total.insert(0, "Date", total["InTime"].map(lambda item:item[0:10]))

# 删去无用的通道编号和价格
total = total.drop(["ChannelNumber","Price"],axis = 1)
# 只保留日期和每日新增人数
covid = covid.drop(["Num","SuspectedCount",'CurrentConfirmedCount',"ConfirmedCount",'DeadCount','CuredCount','DateType'],axis=1)
print(covid)

# 合并数据，得到每条出行记录以及出行当天的新增人数
total = pd.merge(total,covid,how = "left",on="Date")
total = total.drop(["Date"],axis=1)
print(total)
total.to_csv("data/total2-covid.csv",index = 0)