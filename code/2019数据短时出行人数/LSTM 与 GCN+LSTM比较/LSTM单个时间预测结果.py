import datetime
import requests
import json

T = datetime.datetime.strptime("2019-01-08 16:15:00","%Y-%m-%d %H:%M:%S")

L = []

while T.__le__(datetime.datetime.strptime("2019-01-08 20:30:00","%Y-%m-%d %H:%M:%S")):
    # print(T)    
    res = requests.get("http://127.0.0.1:5000/forcast_15/forcast.do?Station=46&PredictTime=" + T.strftime("%Y-%m-%d %H:%M:%S"))
    obj = json.loads(res.text)
    print(obj)
    L.append(obj["PredictiveValue"])
    T += datetime.timedelta(minutes=15)

print(L)