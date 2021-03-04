# 输入Personas
# 对于已经转换过标签的所有用户出行记录，将用户的住宅、公司标签与站点关联起来。
# 统计该站点对于多少用户来说是住宅（公司）（其他）
import pandas as pd
import numpy as np
import datetime

start = datetime.datetime.now()

AllUserTrips = pd.read_csv("用户画像/AllUserTrips.csv")
Station = pd.read_csv("data/station.csv")


# InStationName,origin,UserID
origin = AllUserTrips.drop(["InTime","OutStationName","OutTime","Price","DateType","destination"],axis=1)
origin= origin.drop_duplicates()
origin.columns = ["Station","Type","UserID"]
print(origin)

# OutStationName,destination,UserID
destination = AllUserTrips.drop(["InStationName","InTime","OutTime","Price","DateType","origin"],axis=1)
destination = destination.drop_duplicates()
destination.columns = ["Station","Type","UserID"]
print(destination)

union_result = pd.merge(origin, destination, how='outer')
print(union_result)

StationName = Station["StationName"].values

MarkStation = pd.DataFrame(columns=[0,1,2])

for item in StationName:
    MarkStation.loc[item,:] = 0

for index,record in union_result.iterrows():
    MarkStation.loc[record["Station"],record["Type"]] += 1

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