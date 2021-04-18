import keras
import joblib
from keras.models import load_model
import datetime
import pandas as pd
import numpy as np
from queue import Queue
from sklearn.preprocessing import MinMaxScaler
from keras_gcn import *
import keras.backend as K
import time
import tensorflow as tf


st = time.time()
# 需要预测的站点
Station = 46
# 需要预测的时间
PredictTime = "2019-01-08 20:30:44"
PredictTime = datetime.datetime.strptime(PredictTime,"%Y-%m-%d %H:%M:%S")
# 时间间隔
TimeInterval = 15
# 单个lstm的步长
TimeStep = 1

# 邻近站点阈值
AdjacentThreshold = 7
# 特征长度
features = 4

# 一天有多少个时间段
DayInterval = int((60 / TimeInterval) * 24)
# 一周有多少个时间段
WeekInterval = int((60 / TimeInterval) * 24 * 7)
# 地铁运行时间6:00-23:30    [TimeStart,TimeEnd)
TimeStart = int((60 / TimeInterval) * 6)
TimeEnd = DayInterval - int((30 / TimeInterval))


Map = pd.read_csv("Metro_roadMap.csv",index_col=0)
# 宽搜，将距离Station在AdjacentThreshold内的站点提取出来
def bfs(Station,Map,AdjacentThreshold):
    q = Queue(maxsize=0)
    q.put(Station)
    Flag = np.zeros(Map.shape[0])
    AdjacentStations = [Station]
    Flag[Station] = 1
    Flag[54] = 1

    while AdjacentThreshold > 0:
        AdjacentThreshold -= 1
        size = q.qsize()
        while size > 0:
            size -= 1
            st = q.get()
            for k in range(Map.shape[1]):
                if (Map[st,k] == 1) & (Flag[k] == 0):
                    Flag[k] = 1
                    AdjacentStations.append(k)
                    q.put(k)
    return AdjacentStations

AdjacentStations = bfs(Station,Map.to_numpy(),AdjacentThreshold)
# 站点按照编号从小到大排序
AdjacentStations.sort()
# 邻近站点
# print("邻近站点")
# print(AdjacentStations)
LocalMap = Map.iloc[AdjacentStations,AdjacentStations].values

# day,Time
day = PredictTime.strftime("%Y-%m-%d")
Time = str(PredictTime.hour) + ":" + str(int(PredictTime.minute / TimeInterval) * TimeInterval)
print("Time = " + Time)

col_days,col_Times = [],[]

for i in range(TimeStep):
    PredictTime = PredictTime + datetime.timedelta(minutes=-TimeInterval)
    col_days.append(PredictTime.strftime("%Y-%m-%d"))
    col_Times.append(str(PredictTime.hour) + ":" + str(int(PredictTime.minute / TimeInterval) * TimeInterval))

# 时间随着下标增大
col_days.reverse()
col_Times.reverse()

# (3,41)
Ridership = np.zeros(shape=(TimeStep,len(AdjacentStations)))

col = 0
for AdjacentStation in AdjacentStations:
    AdjacentData = pd.read_csv("by{}minutes/sta{}.csv".format(TimeInterval,AdjacentStation))
    for i in range(TimeStep):
        Ridership[i,col] = AdjacentData.loc[(AdjacentData["day"] == col_days[i]) & (AdjacentData["Time"] == col_Times[i]),"Num"].values[0]
    col += 1

RidershipScaler = joblib.load('LSTM 与 GCN+LSTM比较/Station46_RidershipScaler')
Ridership = RidershipScaler.transform(Ridership)

InputData = []

for i in range(TimeStep):
    InputData.append(Ridership[i,:])
    InputData[i] = InputData[i].reshape(1,InputData[i].shape[0],1)
    # print(InputData[i].shape)

LocalMap = LocalMap.reshape(1,LocalMap.shape[0],LocalMap.shape[1])
InputData += [LocalMap]

model = load_model("LSTM 与 GCN+LSTM比较/GCN-Station46.h5",custom_objects={"GraphConv":GraphConv},compile=False)

Output = model.predict(InputData)
print(Output)
TrueValueScaler = joblib.load('LSTM 与 GCN+LSTM比较/Station46_TrueValueScaler')
Output = TrueValueScaler.inverse_transform(Output)
Output = Output[0][0]
print(type(Output))
print(Output)
print("耗时：{}".format(time.time() - st))