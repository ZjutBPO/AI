# 输入data/total2.csv，所有用户的出行数据。
# 筛选出出行记录>threshold的用户。
# 对每个用户的出行记录，如果该出行记录在早上10点前，就把起始站添加到home中,终点站添加到work中
# 对每个用户的出行记录，如果该出行记录在下午17点后，就把起始站添加到work中,终点站添加到home中
# 如果home中出现次数最多的站，其次数占全部的40%以上，则认为其为住宅
# work同理。
# 输出Personas-50.csv。包含UserID,HomeLine,HomeStation,WorkLine,WorkStation
# UserTrips-50.csv。从total中选取 出行记录数量>threshold的用户的出行记录
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import datetime

start = datetime.datetime.now()

data = pd.read_csv("data/total2.csv")

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

# 如果出行记录小于threshold条，则认为出行数据太小，无法进行统计。
threshold = 20
trips = pd.DataFrame(columns=["UserID","InStationName","InLine","InTime","OutStationName","OutLine","OutTime","ChannelNumber","Price","DateType"])
# 将所有出行数据按照用户ID进行分类
for df1 ,df2 in data.groupby(['UserID']):
    # print(df1)
    # print(df2)

    if df2.shape[0] < threshold:
        continue

    trips = trips.append(df2)

    # 添加到输出的一行
    item = [df1]
    # 可以被认为是家（公司）的地铁线、地铁站
    home,work = [],[]

    for index, row in df2.iterrows():
        H = int(row["InTime"][11:13])
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
output.to_csv("用户画像/Personas-{}.csv".format(threshold),index= 0)

trips.drop(["ChannelNumber"],inplace=True)
trips.to_csv("用户画像/UserTrips-{}.csv".format(threshold),index = 0)
# print(trips)

# 展示出行条数为x的人有几个
# NumberOfTrips = pd.read_csv("data/NumberOfTrips.csv")
# plt.plot(NumberOfTrips["NumberOfTrips"].to_numpy(),NumberOfTrips["NumberOfPeople"].to_numpy())
# # 设置横坐标间隔为10
# plt.gca().xaxis.set_major_locator(MultipleLocator(50))
# plt.show()

end = datetime.datetime.now()
print(end - start)