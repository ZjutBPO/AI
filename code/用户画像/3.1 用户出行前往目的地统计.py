# 输入IndividualUserTrips.csv
# 对已经转换过标签的出行记录，统计出住宅-公司、住宅-其他、公司-住宅、公司-其他、其他-住宅、其他-公司、其他-公司的数量
# 统计出7种出行情况的概率
import pandas as pd
import numpy as np

IndividualUserTrips = pd.read_csv("用户画像/IndividualUserTrips.csv")
# print(IndividualUserTrips)
IndividualUserTrips.insert(0,"count",0)
IndividualUserTrips = IndividualUserTrips.groupby(["origin","destination"]).count().iloc[:,0]
print(IndividualUserTrips)

array = np.zeros((3,3))

for i in range(3):
    for j in range(3):
        if (i == j) & (i != 2):
            continue
        array[i,j] = IndividualUserTrips[i,j]

array = array / array.sum() * 100
array = np.around(array,2)

output = pd.DataFrame(array,index=["住宅","公司","其他"],columns=["住宅","公司","其他"])

print(output)
# dict = output.to_dict()
