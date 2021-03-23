from __future__ import print_function

from keras.utils import plot_model
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import TensorBoard
import scipy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from queue import Queue
from keras_gcn import *
from keras import backend
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from math import *
import keras.backend as K
from queue import Queue
import joblib
import time

st = time.time()

def r2(y_true, y_pred):
    SS_reg = K.sum(K.square(y_pred - y_true))
    mean_y = K.mean(y_true)
    SS_tot = K.sum(K.square(y_true - mean_y))
    f = 1 - SS_reg/SS_tot
    return f

# 需要预测的站点
Station = 46
# 时间间隔
TimeInterval = 15
# 邻近站点阈值
AdjacentThreshold = 7
# 单个lstm的步长
TimeStep = 3
# 特征长度
features = 4

# 一天有多少个时间段
DayInterval = int((60 / TimeInterval) * 24)
# 一周有多少个时间段
WeekInterval = int((60 / TimeInterval) * 24 * 7)
# 地铁运行时间6:00-23:30    [TimeStart,TimeEnd)
TimeStart = int((60 / TimeInterval) * 6)
TimeEnd = DayInterval - int((30 / TimeInterval))

def getData():
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
    # 站点按照编号从小到大排序
    AdjacentStations.sort()
    # 邻近站点
    print("邻近站点")
    print(AdjacentStations)
    LocalMap = Map.iloc[AdjacentStations,AdjacentStations]

    # 保留参数：Num,Minute,DateType,temperature,weather
    MainStationData = pd.read_csv("by{}minutes/sta{}.csv".format(TimeInterval,Station)).drop(["day","Time","maxtemperature","mintemperature"],axis = 1)
    # 统计时间范围在6:00-23:30之间的数据总共有多少条
    # print(MainStationData.loc[(MainStationData["Minute"] >= TimeStart) & (MainStationData["Minute"] < TimeEnd)])
    DataLength = MainStationData.loc[(MainStationData["Minute"] >= TimeStart) & (MainStationData["Minute"] < TimeEnd)].shape[0]
    print("DataLength")
    print(DataLength)

    MainStationData = MainStationData.to_numpy()
    AdjacentStationsData = {}

    # print(AllData is None)

    Flag = 0

    # 原始数据中有参数：Num,Minute,DateType,day,Time,maxtemperature,mintemperature,temperature,weather
    # 保留参数：Num,DateType,temperature,weather
    # （按照站点编号从小到大）将所有站点（包括主站点）的记录存储在AllData中。
    for AdjacentStation in AdjacentStations:
        if Flag == 0:
            AllData = pd.read_csv("by{}minutes/sta{}.csv".format(TimeInterval,AdjacentStation)).drop(["Minute","day","Time","maxtemperature","mintemperature"],axis = 1).to_numpy()
            Flag = 1
        else:
            AllData = np.vstack((AllData,pd.read_csv("by{}minutes/sta{}.csv".format(TimeInterval,AdjacentStation)).drop(["Minute","day","Time","maxtemperature","mintemperature"],axis = 1).to_numpy()))

    classes = set(AllData[:,-1])
    # 天气类型
    print("天气类型")
    print(classes)
    classes_dict = {c: i for i, c in enumerate(classes)}
    # map()函数根据提供的函数对指定序列做映射
    # map(function, iterable)
    # 第一个参数function以参数序列中的每一个元素调用function函数，返回包含每次function函数返回值的新列表
    # 将天气标签转换成int类型
    AllData[:,-1] = np.array(list(map(classes_dict.get, AllData[:,-1])), dtype=np.int32)
    MainStationData[:,-1] = np.array(list(map(classes_dict.get, MainStationData[:,-1])), dtype=np.int32)

    # 每个站点记录数都相同，以主站点的数量作为每个站点记录的数量。
    StationRecordLength = MainStationData.shape[0]
    # 三个维度分别是（站点数量，每个站点的记录数量，每条记录的特征数量）
    AllData = AllData.reshape(-1,StationRecordLength,features)
    # 将每个站点与其站点标号相对应（之前已经按照站点编号从小到大排序）
    for i in range(AllData.shape[0]):
        AdjacentStationsData[AdjacentStations[i]] = AllData[i]

    # 30  29  28  27  26  25  24  23  22  21  20  19  18  17  16  15  14  13  12  11  10  9   8   7   6   5    4
    # 14  13  12  11  10  9   8   7   6   5   4   3   2   1   24  23  22  21  20  19  18  17  16  15  14  13  12

    def GetFeatures(index,AdjacentStationsData,TimeStep,features):
        InputData = np.zeros(shape=(TimeStep,len(AdjacentStationsData)))
        row = 0
        for key,value in AdjacentStationsData.items():
            for i in range(TimeStep):
                InputData[TimeStep - i - 1,row] = value[index-i,0]
            row += 1
        return InputData

    # 当前时间的前三个时间段客流量数据
    # 客流量数据为所有站点在那个时间段的客流量
    Ridership = np.zeros(shape=(DataLength,TimeStep,len(AdjacentStationsData)))

    # 当前时刻的真实客流。作为需要预测的值
    TrueValue = np.zeros(shape=(DataLength,1))
    # 需要预测的时刻的其他特征值（DateType,temperature,weather）
    Features = np.zeros(shape=(DataLength,3))

    Flag = 0
    row = 0

    for i in range(0,MainStationData.shape[0]):
        record = MainStationData[i,:]
        if (record[1] >= TimeStart) & (record[1] < TimeEnd):
            # 当天前timestep个相邻时段的特征
            Ridership[row] = np.zeros(shape=(TimeStep,len(AdjacentStationsData)))
            row2 = 0
            for key,value in AdjacentStationsData.items():
                for j in range(TimeStep):
                    Ridership[row][TimeStep - j - 1,row2] = value[i - 1 - j,0]
                row2 += 1


            # DateType,temperature,weather特征
            Features[row] = MainStationData[i,2:]
            # 真实值
            TrueValue[row] = record[0]
            row += 1

    return [Ridership,Features,TrueValue,LocalMap]

Ridership,Features,TrueValue,LocalMap = getData()

MapSize = LocalMap.shape[0]

print(Ridership.shape)
print(Features.shape)
print(TrueValue.shape)
print(LocalMap.shape)

# 标准化
# Ridership = Ridership.reshape(-1,Ridership.shape[-1])
# RidershipScaler = StandardScaler().fit(Ridership)
# Ridership = RidershipScaler.transform(Ridership)

# Ridership = Ridership.reshape(-1,TimeStep,Ridership.shape[-1])

# TrueValueScaler = StandardScaler().fit(TrueValue)
# TrueValue = TrueValueScaler.transform(TrueValue)

# 最大最小归一化
Ridership = Ridership.reshape(-1,Ridership.shape[-1])
RidershipScaler = MinMaxScaler(feature_range=(0, 1))
Ridership = RidershipScaler.fit_transform(Ridership)
joblib.dump(RidershipScaler, 'Station{}_RidershipScaler'.format(Station))

Ridership = Ridership.reshape(-1,TimeStep,Ridership.shape[-1])

TrueValueScaler = MinMaxScaler(feature_range=(0, 1))
TrueValue = TrueValueScaler.fit_transform(TrueValue)
joblib.dump(TrueValueScaler, 'Station{}_TrueValueScaler'.format(Station))

LocalMap = np.expand_dims(LocalMap,axis=0)
LocalMap = np.repeat(LocalMap,TrueValue.shape[0],axis=0)

RidershipList = []
for i in range(TimeStep):
    RidershipList.append(Ridership[:,i,:])
    RidershipList[i] = RidershipList[i].reshape(RidershipList[i].shape[0],RidershipList[i].shape[1],1)
    print(RidershipList[i].shape)

train_X_3,test_X_3,train_X_2,test_X_2,train_X_1,test_X_1,train_Map,test_Map,train_y,test_y = train_test_split(RidershipList[0],RidershipList[1],RidershipList[2],LocalMap,TrueValue,test_size = 0.1,random_state = 0)

train_X = [train_X_3,train_X_2,train_X_1,train_Map,train_Map,train_Map]
test_X = [test_X_3,test_X_2,test_X_1,train_Map,train_Map,train_Map]

X_in = Input(shape=(MapSize,1),name = "StationFeature")
Map_in = Input(shape=(MapSize,MapSize),name = "Map")
GCN1 = GraphConv(4,name="GCN1")([X_in,Map_in])
GCN2 = GraphConv(4,name="GCN2")([GCN1,Map_in])
Output = Flatten()(GCN2)
Output = Dense(100,name="ExtractFeature")(Output)

GCN_Model = Model(inputs = [X_in,Map_in],output = Output,name = "GCN_Part")
# plot_model(GCN_Model,to_file="GCN+LSTM预测(GCN部分.png",show_shapes=True)

Inputs = []
Map_Input = []
GCN_Models = []

for i in range(TimeStep):
    Inputs.append(Input(shape=(MapSize,1),name = "StationFeature_t-{}".format(TimeStep - i)))
    Map_Input.append(Input(shape=(MapSize,MapSize),name = "Map_t-{}".format(TimeStep - i)))
    GCN_Models.append(GCN_Model([Inputs[i],Map_Input[i]]))

MergeLayres = concatenate([GCN_Models[i] for i in range(TimeStep)])
LSTM_Input = Reshape((TimeStep,-1))(MergeLayres)
LSTM1 = LSTM(64,activation="relu",return_sequences=True,name = "LSTM1")(LSTM_Input)
LSTM2 = LSTM(64,activation="relu",name="LSTM2")(LSTM1)
Predict = Dense(1,name="Predict")(LSTM2)

model = Model(inputs = Inputs + Map_Input,output = Predict,name = "GCN+LSTM")

BatchSize = train_X_3.shape[0]
model.compile(loss='mse', optimizer=Adam(lr=0.01),metrics=[r2])
history = model.fit(train_X,train_y,batch_size=BatchSize, epochs=300, shuffle=False, verbose=1,validation_split=0.2)

GCN_Model.summary()
model.summary()
# plot_model(model,to_file="GCN+LSTM预测(整体).png",show_shapes=True)
tensorboard_callback = TensorBoard('./keras')
tensorboard_callback.set_model(model)
tensorboard_callback.writer.flush()


pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.legend()
# pyplot.show()

output = model.predict(test_X)
# print(output)

output = TrueValueScaler.inverse_transform(output)
test_y = TrueValueScaler.inverse_transform(test_y)

rmse = sqrt(mean_squared_error(output,test_y))
mae = mean_absolute_error(output,test_y)
r2 = r2_score(output,test_y)
score = explained_variance_score(output,test_y)

def MAPE(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)

mape = MAPE(output,test_y)

print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)
print('Test R2: %.3f' % r2)
print('Test explained_variance_score: %.3f' % score)
print('Test MAPE: %.3f' % mape)

# print('%.3f' % rmse)
# print('%.3f' % mae)
# print('%.3f' % r2)
# print('%.3f' % score)
# print('%.3f' % mape)

pyplot.plot(output, label='real')
pyplot.plot(test_y, label='pre')
pyplot.legend()
# pyplot.show()
# model.save("{}-minute forecast/Station{}.h5".format(TimeInterval,Station))
print(time.time() - st)