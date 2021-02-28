import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

data = pd.read_csv("data/total2.csv")

# data = data.loc[data["DateType"] == 1]
# 将时间全部转换成小时
data['InTime'] = data['InTime'].map(lambda item:item[11:13])

# 提取出出行过的userId
# userIds = data.drop_duplicates(subset = ['UserID'],keep='last',inplace=False).iloc[:,0]
# print(userIds.head())
# print(userIds.shape)
# userIds.to_csv("用户画像/UserID.csv",index = 0)
# 用户ID，代表家的地铁线、地铁站，代表公司的地铁线、地铁站
output = pd.DataFrame(columns = ["UserID","HomeLine","HomeStation","WorkLine","WorkStation"])

def GetStation(item,L):
    L = pd.DataFrame(L)
    # 不存在候选项则认为其无住宅（公司）
    if (len(L) == 0):
        item.append("")
        item.append("")
        return
    # 统计每个站点的出现次数
    L.insert(0,"cnt",0)
    L = L.groupby([L["Line"],L["StationName"]]).count()
    # print(L)
    # 如果出现次数最多的站点占全部站点的40%以上，则判断其为家（公司）
    if L.max().cnt / L.sum().cnt > 0.4:
        tmp = L.idxmax().cnt
        item.append(tmp[0])
        item.append(tmp[1])
    else:
        item.append("")
        item.append("")
    # return item

# 将所有出行数据按照用户ID进行分类
for df1 ,df2 in data.groupby(['UserID']):
    # print(df1)
    # print(df2)

    # 如果出行记录小于20条，则认为出行数据太小，无法进行预测。
    if df2.shape[0] < 20:
        continue

    # 添加到输出的一行
    item = [df1]
    # 可以被认为是家（公司）的地铁线、地铁站
    home,work = [],[]

    for index, row in df2.iterrows():
        H = int(row["InTime"])
        # 用户在10点之前出行，认为起始站点是家，目标站点是公司
        if  H < 10:
            home.append({"Line":row["InLine"],
                        "StationName":row["InStationName"]})
            work.append({"Line":row["OutLine"],
                        "StationName":row["OutStationName"]})
        # 用户在17点之后出行，认为起始站点是公司，目标站点是家
        elif H > 17:
            work.append({"Line":row["InLine"],
                        "StationName":row["InStationName"]})
            home.append({"Line":row["OutLine"],
                        "StationName":row["OutStationName"]})

    # if (len(home) < 20):
    #     continue

    # 从所有候选项中选出恰当的可以代表为家的地铁线、地铁站。并将其添加到item中
    GetStation(item,home)
    GetStation(item,work)

    item = np.array(item)

    item = pd.DataFrame(item.reshape(1,5),columns = ["UserID","HomeLine","HomeStation","WorkLine","WorkStation"])
    # 添加item到output
    output = output.append(item,ignore_index=True)
    # print(output)
    # break

output.sort_values(by = "UserID")
output.to_csv("用户画像/Personas.csv",index= 0)

# 展示出行条数为x的人有几个
# NumberOfTrips = pd.read_csv("data/NumberOfTrips.csv")
# plt.plot(NumberOfTrips["NumberOfTrips"].to_numpy(),NumberOfTrips["NumberOfPeople"].to_numpy())
# # 设置横坐标间隔为10
# plt.gca().xaxis.set_major_locator(MultipleLocator(50))
# plt.show()