# 输入Personas
# 对于已经转换过标签的所有用户出行记录，将用户的住宅、公司标签与站点关联起来。
# 统计该站点对于多少用户来说是住宅（公司）（其他）
import pandas as pd
import numpy as np
import datetime

start = datetime.datetime.now()

AllUserTrips = pd.read_csv("用户画像/AllUserTrips.csv")
Station = pd.read_csv("data/station.csv")

AllUserTrips["InStationName"] = AllUserTrips["InStationName"].map(lambda item:item[3:])
AllUserTrips["OutStationName"] = AllUserTrips["OutStationName"].map(lambda item:item[3:])
Station["StationName"] = Station["StationName"].map(lambda item:item[3:])

StationName = Station["StationName"].values

# 存每个站点被
MarkStation = pd.DataFrame(columns=[0,1,2])

for item in StationName:
    MarkStation.loc[item,:] = 0

flag = 0

M = np.array([0.5, 0,  0,  0,  0,  0.5])
M1 = np.array([0,   0.7,0,  0,  0.3,0])

def calc(M1,M2):
    sum = 0

    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            sum += M1[i]  * M2[j]
        sum += M1[i] * M2[5-i]
    k = 1 - sum
    print(k)
    k = 1.0 / k
    M3 = np.zeros(6)

    M3[0] = M1[0] * M2[0] + M1[0] * M2[3] + M1[0] * M2[4] +\
            M1[3] * M2[0] + M1[3] * M2[4] +\
            M1[4] * M2[0] + M1[4] * M2[3]
    M3[1] = M1[1] * M2[1] + M1[1] * M2[3] + M1[1] * M2[5] +\
            M1[3] * M2[1] + M1[3] * M2[5] +\
            M1[5] * M2[1] + M1[5] * M2[3]
    M3[2] = M1[2] * M2[2] + M1[2] * M2[4] + M1[2] * M2[5] +\
            M1[4] * M2[2] + M1[4] * M2[5] +\
            M1[5] * M2[2] + M1[5] * M2[4]
    M3[3] = M1[3] * M2[3]
    M3[4] = M1[4] * M2[4]
    M3[5] = M1[5] * M2[5]
    M3 *= k
    print(M3)
    return M3

M = calc(M,M1)
M = calc(M,M1)
M = calc(M,np.array([0.5, 0,  0,  0,  0,  0.5]))
M = calc(M,np.array([0.5, 0,  0,  0,  0,  0.5]))

for df1 ,df2 in AllUserTrips.groupby(['UserID']):
    len = df2.shape[0]
    for index,row in df2.iterrows():
        MarkStation.loc[row["InStationName"],row["origin"]] += 1.0 / len
        MarkStation.loc[row["OutStationName"],row["destination"]] += 1.0 / len

MarkStation.to_csv("用户画像/StationTypeCount.csv")

MarkStation = MarkStation.astype("float64")
MarkStation = MarkStation.idxmax(axis = 1)

MarkStation.replace("0","住宅区",inplace = True)
MarkStation.replace("1","商务区",inplace = True)
MarkStation.replace("2","生活区",inplace = True)

MarkStation = pd.DataFrame(MarkStation,columns=["StationType"])

output = pd.merge(Station,MarkStation,how="left",left_on="StationName",right_index=True)

output.to_csv("用户画像/StationMark.csv",index = 0)

end = datetime.datetime.now()
print(end-start)
