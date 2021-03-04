# 输入IndividualUserTrips.csv
# 对用户的出行数据，统计出每个小时的出行数量
# 计算出24小时出行的概率
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

IndividualUserTrips = pd.read_csv("用户画像/IndividualUserTrips.csv")
print(type(IndividualUserTrips))
IndividualUserTrips.insert(0,"count",0)
IndividualUserTrips["InTime"] = IndividualUserTrips["InTime"].map(lambda item:item[-2:])
IndividualUserTrips = IndividualUserTrips.groupby("InTime").count().iloc[:,0]
# print(type(IndividualUserTrips))

array = np.zeros(24)

for key,values in IndividualUserTrips.items():
    array[int(key)] = values

array = np.around(array / array.sum() * 100,2)
print(array)
# plt.plot(array)
# plt.show()
