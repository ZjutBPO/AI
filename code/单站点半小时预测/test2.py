# -!- coding: utf-8 -!-
import pandas as pd

pd.set_option('display.max_columns',None)  #设置列不限制数量
pd.set_option('display.max_rows',None)     #设置行不限制数量

for t in range(80):
    df=pd.read_csv('train/record_2019-01-11.csv')
    p=t
    t = df.loc[(df['stationID'] == t)]
    a=[]
    for i in range(48):
        a.append(0)
    for index,row in t.iterrows():
        hour = int(int(row[0][11:13]))
        minute = int(int(row[0][14:16])/30)
        a[hour*2+minute] = a[hour*2+minute]+1
    print(a)
    data=pd.DataFrame(a)
    name=str(p)+'.csv'
    print(name)
    data.to_csv('data/2019-01-11/'+name)