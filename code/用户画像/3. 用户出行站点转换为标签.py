# 将用户出行的站分为住宅、公司、其他三类
import pandas as pd
import numpy as np

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns',None)  #设置列不限制数量
pd.set_option('display.max_rows',None)     #设置行不限制数量

total = pd.read_csv("data/total2-covid.csv")
user = pd.read_csv("用户画像/Personas.csv")

username="0006366a03e947e45df7082b8546afb7"
usertrips = total.loc[total["UserID"] == username]
# usertrips["InTime"] = usertrips["InTime"].map(lambda item:item[11:13])
usertrips = usertrips.sort_values(by = "InTime")
record = user.loc[user["UserID"] == username]
print(record)
record = record.iloc[0,:]

def getState(record,Line,Station):
    if (Line == record["HomeLine"]) & (Station== record["HomeStation"]):
        return 0    #home
    elif (Line == record["WorkLine"]) & (Station== record["WorkStation"]):
        return 1    #work
    else:
        return 2    #other

# 判断出行记录的起始站是住宅、公司、其他中的哪一种
usertrips['origin'] = usertrips.apply(lambda x: getState(record,x.InLine, x.InStationName), axis = 1)
# 判断出行记录的目的站是住宅、公司、其他中的哪一种
usertrips['destination'] = usertrips.apply(lambda x: getState(record,x.OutLine, x.OutStationName), axis = 1)

usertrips = usertrips.fillna(0)
# print(usertrips)
# 出行记录只保留年、月、日、小时
usertrips["InTime"] = usertrips["InTime"].map(lambda item:item[0:13])
usertrips["OutTime"] = usertrips["OutTime"].map(lambda item:item[0:13])
usertrips = usertrips.drop(["InLine","InStationName","OutLine","OutStationName","UserID"],axis = 1)
usertrips.insert(len(usertrips.columns),"Trip",1)

# 2020-03-13 18:15:18
usertrips.to_csv("用户画像/UserTrips.csv",index = 0)