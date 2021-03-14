from __future__ import print_function

from keras.utils import plot_model
from keras.layers import *
from keras.models import Model,Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
import scipy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras_gcn import GraphConv
from keras import backend
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from math import *
import keras.backend as K
import pandas as pd
import time

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


# Num,Minute,DateType,day,Time,maxtemperature,mintemperature,temperature,weather
# Num,DateType,temperature,weather作为输入参数，Minute作为判断需要预测的时间是不是6:00-23:00
MainStationData = pd.read_csv("by{}minutes/sta{}.csv".format(TimeInterval,Station)).drop(["day","Time","maxtemperature","mintemperature"],axis = 1)

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
    print("天气类型")
    print(classes)
    classes_dict = {c: i for i, c in enumerate(classes)}
    # map()函数根据提供的函数对指定序列做映射
    # map(function, iterable)
    # 第一个参数function以参数序列中的每一个元素调用function函数，返回包含每次function函数返回值的新列表
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def r2(y_true, y_pred):
    SS_reg = K.sum(K.square(y_pred - y_true))
    mean_y = K.mean(y_true)
    SS_tot = K.sum(K.square(y_true - mean_y))
    f = 1 - SS_reg/SS_tot
    return f

values = MainStationData.values
values[:,-1] = LabelToNum(values[:,-1])
values = values.astype('float32')
MainStationData = pd.DataFrame(values)

n_hours = 4
n_features = 4
# 1 6 11 16 21
# 将数据格式化成监督学习型数据
reframed = series_to_supervised(MainStationData, n_hours, 1)
reframed = reframed.loc[(reframed["var2(t)"] >= TimeStart) & (reframed["var2(t)"] < TimeEnd)]
reframed = reframed.drop(["var2(t-{})".format(i+1) for i in range(n_hours)] + ["var2(t)"],axis = 1)

reframed = reframed.iloc[:,:n_hours * n_features + 1]

values = reframed.values
print(values)

X = values[:,:-1]
Y = values[:,-1]

Y = Y.reshape(-1,1)

# 将所有数据缩放到（0，1）之间
X_scaler = MinMaxScaler(feature_range=(0, 1))
X = X_scaler.fit_transform(X)

Y_scaler = MinMaxScaler(feature_range=(0, 1))
Y = Y_scaler.fit_transform(Y)

train_x,test_X,train_y,test_y = train_test_split(X,Y,test_size = 0.1,random_state = 5)
train_x = train_x.reshape((train_x.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_x.shape, train_y.shape, test_X.shape, test_y.shape)

model = Sequential()
model.add(LSTM(64, activation="relu",input_shape=(train_x.shape[1], train_x.shape[2]),return_sequences=True))
model.add(LSTM(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(16))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam',metrics=[r2])

# 拟合网络
history = model.fit(train_x, train_y, epochs=400, batch_size=train_x.shape[0], validation_split=0.2, verbose=2, shuffle=False)

model.summary()
from keras.utils import plot_model
plot_model(model,to_file="lstm预测模型.png",show_shapes=True)

# 图像展示训练损失
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.legend()
pyplot.show()

output = model.predict(test_X)

output = Y_scaler.inverse_transform(output)
test_y = Y_scaler.inverse_transform(test_y)

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
pyplot.show()