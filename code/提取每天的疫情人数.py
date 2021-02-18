import numpy as np
import pandas as pd
import datetime

date = datetime.date(2020,1,24)
ans = pd.DataFrame()
while (date.__le__(datetime.date(2020,7,16))):
    data = pd.read_json("./COVID-19/COVID-19_{}(CN-DATA)by_DXY.json".format(date))
    data = data.sum()
    data['date'] = date
    if date.__le__(datetime.date(2020,2,13)):
        data['currentConfirmedCount'] = None
    ans = ans.append(
        pd.Series([date,data['currentConfirmedCount'],data['confirmedCount'],data['suspectedCount'],data['curedCount'],data['deadCount']])
        ,ignore_index = True)
    date+=datetime.timedelta(days=1)

ans.columns = ['Date','CurrentConfirmedCount','ConfirmedCount','SuspectedCount','CuredCount','DeadCount']
ans.to_csv('data/COVID-19.csv',index = 0)