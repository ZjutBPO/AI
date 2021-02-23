# -!- coding: utf-8 -!-
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
import pandas as pd

pd.set_option("display.width",None)
pd.set_option("display.max_rows",None)

import keras.backend as K
def r2(y_true, y_pred):
    SS_reg = K.sum(K.square(y_pred - y_true))
    mean_y = K.mean(y_true)
    SS_tot = K.sum(K.square(y_true - mean_y))
    f = 1 - SS_reg/SS_tot
    return f

# load dataset
dataset = read_csv('2020-5-1~2020-7-16.csv')
# 删掉那些我们不想预测的列
dataset = dataset.drop(["Date"],axis=1)
values = dataset.values

# 数据转换为浮点型
values = values.astype('float32')
# 将所有数据缩放到（0，1）之间
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
scaled = DataFrame(scaled)
scaled.columns=['Num','Hour','DateType']

# 将数据格式化成监督学习型数据
reframed = concat([scaled.shift(-1),scaled],axis=1)
reframed = reframed.iloc[:,:4]
reframed.dropna(inplace=True)

# split into train and test sets
values = reframed.values


# 取出一天的数据作为训练数据，剩下的做测试数据
n_train = -18
train = values[:n_train, :]
test = values[n_train:, :]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# 将输入数据转换成3D张量 [samples, timesteps, features]，[n条数据，每条数据1个步长，8个特征值]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# 最终生成的数据形状，X:(1364,1,2)  Y:(1364,)
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# 设计网络结构
model = Sequential()
model.add(LSTM(256, activation="relu",input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
# model.add(LSTM(256, activation="relu",return_sequences=True))
model.add(LSTM(256, activation="relu"))
model.add(Dropout(0.3))
# model.add(Dense(128))
model.add(Dense(16))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam',metrics=[r2])
# 拟合网络
history = model.fit(train_X, train_y, epochs=100, batch_size=20, validation_data=(test_X, test_y), verbose=2, shuffle=False)
model.save("testmodle.h5")
# 图像展示训练损失
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

yhat = model.predict(test_X)

#重构预测数据形状，进行逆缩放
# 之所有下面奇怪的数据拼接操作，是因为：数据逆缩放要求输入数据的形状和缩放之前的输入保值一致
# 将3D转换为2D
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# 拼接y，x为[y,x],即将test_X中的第一列数据替换成预测出来的yhat值
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
# 对替换后的inv_yhat预测数据逆缩放
inv_yhat = scaler.inverse_transform(inv_yhat)
# 逆缩放后取出第一列(预测列)y
inv_yhat = inv_yhat[:,0]


# 重构真实数据形状，进行逆缩放
test_y = test_y.reshape((len(test_y), 1))
# 因为test_X的第一列数据在上面被修改过，这里要重新还原一下真实数据。将test_X的第一列换成原始数据test_y值
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
# 对重构后的数据进行逆缩放
inv_y = scaler.inverse_transform(inv_y)
# 逆缩放后取出第一列(真实列)y
inv_y = inv_y[:,0]
# 计算预测列和真实列的误差RMSE值
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
pyplot.plot(inv_y, label='real')
pyplot.plot(inv_yhat, label='pre')
pyplot.legend()
pyplot.show()