from pandas import read_csv
from matplotlib import pyplot
import pandas as pd
import numpy as np

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)

# load dataset
dataset = read_csv('预测总出行人数/date-num-COVID.csv')
data = dataset[14:]
data.fillna(0,inplace = True)

ConfirmedCount = data["ConfirmedCount"]
# 求比前一天增长了多少确诊人数
ConfirmedCount = ConfirmedCount.diff()
ConfirmedCount[0] = 0
print(ConfirmedCount)

# 有一天增长有15000，是因为调整了判断指标
index = ConfirmedCount.idxmax()
print(index)

# 修正增长人数
ConfirmedCount[index] = 3700
ConfirmedCount[index+1] = 3300
values = data.values

pyplot.figure()
for i in range(1,7):
	pyplot.subplot(7, 1, i)
	pyplot.plot(values[:, i])
	pyplot.title(dataset.columns[i], y=0.5, loc='right')

pyplot.subplot(7, 1, 7)
pyplot.plot(ConfirmedCount)
pyplot.title("DiffConfirmedCount", y=0.5, loc='right')
pyplot.show()

data.insert(data.shape[1],"DiffConfirmedCount",ConfirmedCount)
# data["DiffConfirmedCount"] = ConfirmedCount
print(data)
data.to_csv('预测总出行人数/date-num-COVID-diff.csv',index = 0)