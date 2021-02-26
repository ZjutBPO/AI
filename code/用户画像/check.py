import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

data = pd.read_csv("data/total2.csv")

data = data.loc[data["DateType"] == 1]
data['InTime'] = data['InTime'].map(lambda item:item[11:13])

# 提取出出行过的userId
# userIds = data.drop_duplicates(subset = ['UserID'],keep='last',inplace=False).iloc[:,0]
# print(userIds.head())
# print(userIds.shape)
# userIds.to_csv("用户画像/UserID.csv",index = 0)
output = pd.DataFrame(columns = ["UserID","HomeLine","HomeStation","WorkLine","WorkStation"])

def GetStation(item,L):
    L = pd.DataFrame(L)
    if (len(L) == 0):
        item.append("")
        item.append("")
        return
    L.insert(0,"cnt",0)
    L = L.groupby([L["Line"],L["StationName"]]).count()
    # print(L)
    if L.max().cnt / L.sum().cnt > 0.4:
        tmp = L.idxmax().cnt
        item.append(tmp[0])
        item.append(tmp[1])
    else:
        item.append("")
        item.append("")
    # return item

ttt = 0

for df1 ,df2 in data.groupby(['UserID']):
    # print(df1)
    # print(df2)

    if df2.shape[0] < 20:
        ttt += 1
        continue

    item = [df1]
    home,work = [],[]

    for index, row in df2.iterrows():
        H = int(row["InTime"])
        if  H < 10:
            home.append({"Line":row["InLine"],
                        "StationName":row["InStationName"]})
            work.append({"Line":row["OutLine"],
                        "StationName":row["OutStationName"]})
        elif H > 17:
            work.append({"Line":row["InLine"],
                        "StationName":row["InStationName"]})
            home.append({"Line":row["OutLine"],
                        "StationName":row["OutStationName"]})
    # if (len(home) < 20):
    #     continue
    GetStation(item,home)
    GetStation(item,work)

    item = np.array(item)

    item = pd.DataFrame(item.reshape(1,5),columns = ["UserID","HomeLine","HomeStation","WorkLine","WorkStation"])
    output = output.append(item,ignore_index=True)
    # print(output)
    # break

output.sort_values(by = "UserID")
output.to_csv("用户画像/Personas2.csv",index= 0)
print(ttt)

# 展示出行条数为x的人有几个
# NumberOfTrips = pd.read_csv("data/NumberOfTrips.csv")
# plt.plot(NumberOfTrips["NumberOfTrips"].to_numpy(),NumberOfTrips["NumberOfPeople"].to_numpy())
# # 设置横坐标间隔为10
# plt.gca().xaxis.set_major_locator(MultipleLocator(50))
# plt.show()