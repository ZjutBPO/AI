# 将所有用户的出行记录中的站点替换成0(住宅)，1(公司)，2(其他)
# 输出AllUserTrips。包含InStationName,InTime,OutStationName,OutTime,Price,DateType,origin,destination
# origin表示进站站点的属性，destination表示出站站点的属性
import pandas as pd
import numpy as np
import datetime

start = datetime.datetime.now()

total = pd.read_csv("用户画像/UserTrips-{}.csv".format(50))
user = pd.read_csv("用户画像/Personas-{}.csv".format(50))
print(user)

def getState(record,Line,Station):
    if (Line == record["HomeLine"]) & (Station== record["HomeStation"]):
        return 0    #home
    elif (Line == record["WorkLine"]) & (Station== record["WorkStation"]):
        return 1    #work
    else:
        return 2    #other

output = pd.DataFrame(columns=["InStationName","InTime","OutStationName","OutTime","ChannelNumber","Price","DateType","origin","destination"])

for index,record in user.iterrows():
    username = record["UserID"]
    usertrips = total.loc[total["UserID"] == username]

    # 判断出行记录的起始站是住宅、公司、其他中的哪一种
    usertrips['origin'] = usertrips.apply(lambda x: getState(record,x.InLine, x.InStationName), axis = 1)
    # 判断出行记录的目的站是住宅、公司、其他中的哪一种
    usertrips['destination'] = usertrips.apply(lambda x: getState(record,x.OutLine, x.OutStationName), axis = 1)

    # print(usertrips)
    # 出行记录只保留年、月、日、小时
    # usertrips = usertrips.drop(["InLine","InStationName","OutLine","OutStationName","UserID"],axis = 1)
    output = output.append(usertrips)

output = output.fillna(0)
output["InTime"] = output["InTime"].map(lambda item:item[0:13])
output["OutTime"] = output["OutTime"].map(lambda item:item[0:13])
output = output.drop(["InLine","OutLine","ChannelNumber"],axis = 1)
# usertrips.insert(len(usertrips.columns),"Trip",1)
output = output.sort_values(by = ["UserID","InTime"])
output.to_csv("用户画像/AllUserTrips.csv",index = 0)

end = datetime.datetime.now()
print(end - start)