import numpy as np
import pandas as pd
import datetime

covid = pd.read_csv("用户画像/date-num-COVID-diff.csv",index_col='Date')

# print(usertrips)
print(covid)

date = datetime.datetime.strptime("2020-01-01 00", "%Y-%m-%d %H")
# print(date)

output = pd.DataFrame(columns=["Time","DiffConfirmedCount"])

lists = []

while date.__le__(datetime.datetime.strptime("2020-07-16 23", "%Y-%m-%d %H")):
    if date.hour < 6:
        date += datetime.timedelta(hours=1)
        continue
    time = date.strftime("%Y-%m-%d %H")
    day = date.strftime("%Y-%m-%d")
    record = covid.loc[day]
    # print(record.DiffConfirmedCount)
    lists.append([time,record.DiffConfirmedCount,record.DateType])
    date += datetime.timedelta(hours=1)

# print (datetime.datetime.now()+).strftime("%Y-%m-%d %H:%M:%S")
print("############")
# print(date)
output = output.append(pd.DataFrame(lists,columns=["Time","DiffConfirmedCount","DateType"]),ignore_index=True)
print(output)
output.to_csv("用户画像/AllTime-Covid.csv",index=0)