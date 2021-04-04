from __future__ import print_function

import scipy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from math import *
import pandas as pd
import time
import datetime
import joblib

# 需要预测的站点
Station = 46
# 时间间隔
TimeInterval = 15
# 一天有多少个时间段
DayInterval = int((60 / TimeInterval) * 24)
# 一周有多少个时间段
WeekInterval = int((60 / TimeInterval) * 24 * 7)
# 地铁运行时间6:00-23:30    [TimeStart,TimeEnd)
TimeStart = int((60 / TimeInterval) * 6)
TimeEnd = DayInterval - int((30 / TimeInterval))

for i in range(81):
    if i == 54:
        continue
    Station = i
    print(Station)
    # Num,Minute,DateType,day,Time,maxtemperature,mintemperature,temperature,weather
    # Num,DateType,temperature,weather作为输入参数，Minute作为判断需要预测的时间是不是6:00-23:00
    MainStationData = pd.read_csv("by{}minutes/sta{}.csv".format(TimeInterval,Station)).drop(["maxtemperature","mintemperature"],axis = 1)
    def Get_Predict_Time(_date,_time):
        return datetime.datetime.strptime(_date + " " + _time + ":00", "%Y-%m-%d %H:%M:%S")    

    MainStationData["Predict_Time"] = MainStationData.apply(lambda item:Get_Predict_Time(item.day,item.Time),axis = 1)
    MainStationData = MainStationData.drop(["day","Time"],axis = 1)


    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        # 获取特征值数量n_vars
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        # print(df)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        # 创建8个v(t-1)作为列名
        for i in range(n_in, 0, -1):
            # 向列表cols中添加一个df.shift(1)的数据
            cols.append(df.shift(i))
            # print(cols)
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            # 向列表cols中添加一个df.shift(-1)的数据
            cols.append(df.shift(-i))
            # print(cols)
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # print(cols)
        # 将列表中两个张量按照列拼接起来，list(v1,v2)->[v1,v2],其中v1向下移动了一行，此时v1,v2是监督学习型数据
        agg = pd.concat(cols, axis=1)
        # print(agg)
        # 重定义列名
        agg.columns = names
        # print(agg)
        # 删除空值
        if dropnan:
            agg.dropna(inplace=True)
        return agg


    def LabelToNum(labels):
        classes = set(labels)
        # 天气类型
        # print("天气类型")
        # print(classes)
        classes_dict = {c: i for i, c in enumerate(classes)}
        # map()函数根据提供的函数对指定序列做映射
        # map(function, iterable)
        # 第一个参数function以参数序列中的每一个元素调用function函数，返回包含每次function函数返回值的新列表
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return labels_onehot

    values = MainStationData.values
    # 将倒数第二列天气从 中文 转换成 数字
    values[:,-2] = LabelToNum(values[:,-2])
    # Num,Minute,DateType,temperature,weather,Predict_Time
    MainStationData = pd.DataFrame(values)
    # print(MainStationData)

    n_hours = 4
    n_features = 4
    # 1 6 11 16 21
    # 将数据格式化成监督学习型数据
    reframed = series_to_supervised(MainStationData, n_hours, 1)
    reframed = reframed.loc[(reframed["var2(t)"] >= TimeStart) & (reframed["var2(t)"] < TimeEnd)]
    # print(reframed)
    # 删除每个时间步的第二列(Minute)
    reframed = reframed.drop(["var2(t-{})".format(i+1) for i in range(n_hours)] + ["var2(t)"],axis = 1)
    # 删除除最后一个之外的Predict_Time列
    reframed = reframed.drop(["var6(t-{})".format(i+1) for i in range(n_hours)],axis = 1)
    # print("reframed")
    # print(reframed)

    values = reframed.iloc[:,:n_hours * n_features + 1].values
    # print(values)

    X = values[:,:-1]
    Y = values[:,-1]

    Y = Y.reshape(-1,1)

    # 将所有数据缩放到（0，1）之间
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    X = X_scaler.fit_transform(X)
    # joblib.dump(X_scaler, "Predict_Data_{}/Station{}_X_scaler".format(TimeInterval,Station))

    Y_scaler = MinMaxScaler(feature_range=(0, 1))
    Y = Y_scaler.fit_transform(Y)
    joblib.dump(Y_scaler, "Predict_Data_{}/Station{}_Y_scaler".format(TimeInterval,Station))

    Predict_time = reframed.iloc[:,-1].values

    Predict_data = pd.DataFrame(X)
    Predict_data.insert(0, "PredictTime", Predict_time)
    # print(Predict_data)
    Predict_data.to_csv("Predict_Data_{}/Station{}.csv".format(TimeInterval,Station),index = 0)
