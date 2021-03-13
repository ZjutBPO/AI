import numpy as np
import pandas as pd
import scipy.sparse as sp       # python中稀疏矩阵相关库
from queue import Queue
from sklearn.preprocessing import MinMaxScaler

# 需要预测的站点
Station = 46
# 时间间隔
TimeInterval = 15
# 邻近站点阈值
AdjacentThreshold = 7
# 单个lstm的步长
timestep = 3
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
AdjacentStations.sort()
# 邻近站点
print("邻近站点")
print(AdjacentStations)
LocalMap = Map.iloc[AdjacentStations,AdjacentStations]

# 保留参数：Num,Minute,DateType,temperature,weather
MainStationData = pd.read_csv("by{}minutes/sta{}.csv".format(TimeInterval,Station)).drop(["day","Time","maxtemperature","mintemperature"],axis = 1)
print(MainStationData.loc[(MainStationData["Minute"] >= TimeStart) & (MainStationData["Minute"] < TimeEnd)])
DataLength = MainStationData.loc[(MainStationData["Minute"] >= TimeStart) & (MainStationData["Minute"] < TimeEnd)].shape[0]
print("DataLength")
print(DataLength)

MainStationData = MainStationData.to_numpy()
AdjacentStationsData = {}

# print(AllData is None)

Flag = 0

# 原始数据中有参数：Num,Minute,DateType,day,Time,maxtemperature,mintemperature,temperature,weather
# 保留参数：Num,DateType,temperature,weather

for AdjacentStation in AdjacentStations:
    if Flag == 0:
        AllData = pd.read_csv("by{}minutes/sta{}.csv".format(TimeInterval,AdjacentStation)).drop(["Minute","day","Time","maxtemperature","mintemperature"],axis = 1).to_numpy()
        Flag = 1
    else:
        AllData = np.vstack((AllData,pd.read_csv("by{}minutes/sta{}.csv".format(TimeInterval,AdjacentStation)).drop(["Minute","day","Time","maxtemperature","mintemperature"],axis = 1).to_numpy()))

def LabelToNum(labels):
    classes = set(labels)
    # 天气类型
    print("天气类型")
    print(classes)
    classes_dict = {c: i for i, c in enumerate(classes)}
    # map()函数根据提供的函数对指定序列做映射
    # map(function, iterable)
    # 第一个参数function以参数序列中的每一个元素调用function函数，返回包含每次function函数返回值的新列表
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

# 将天气标签转换成int类型
AllData[:,-1] = LabelToNum(AllData[:,-1])
MainStationData[:,-1] = LabelToNum(MainStationData[:,-1])

# # 归一化
# scaler = MinMaxScaler(feature_range=(0, 1))
# AllData = scaler.fit_transform(AllData)

StationRecordLength = MainStationData.shape[0]

AllData = AllData.reshape(-1,StationRecordLength,features)

for i in range(AllData.shape[0]):
    AdjacentStationsData[AdjacentStations[i]] = AllData[i]

# 30  29  28  27  26  25  24  23  22  21  20  19  18  17  16  15  14  13  12  11  10  9   8   7   6   5    4
# 14  13  12  11  10  9   8   7   6   5   4   3   2   1   24  23  22  21  20  19  18  17  16  15  14  13  12

def GetFeatures(index,AdjacentStationsData,timestep,features):
    InputData = np.zeros(shape=(timestep,len(AdjacentStationsData)))
    row = 0
    for key,value in AdjacentStationsData.items():
        for i in range(timestep):
            InputData[timestep - i - 1,row] = value[index-i,0]
        row += 1
    return InputData

# 当前时间的前三个时间段客流量数据
# 客流量数据为所有站点在那个时间段的客流量
Ridership = np.zeros(shape=(DataLength,timestep,len(AdjacentStationsData)))

# 当前时刻的真实客流。作为需要预测的值
TrueValue = np.zeros(shape=(DataLength,1))

Features = np.zeros(shape=(DataLength,3))

Flag = 0
row = 0

for i in range(0,MainStationData.shape[0]):
    record = MainStationData[i,:]
    if (record[1] >= TimeStart) & (record[1] < TimeEnd):
        # 当天前timestep个相邻时段的特征
        Ridership[row] = GetFeatures(i-1,AdjacentStationsData,timestep,features)
        # DateType,temperature,weather特征
        Features[row] = MainStationData[i,2:]
        # 真实值
        TrueValue[row] = record[0]
        row += 1

# Ridership = np.array(Ridership)
np.save('Ridership{}'.format(Station),Ridership)
np.save('Features{}'.format(Station),Features)
np.save('TrueValue{}'.format(Station),TrueValue)
np.save('LocalMap{}'.format(Station),LocalMap)