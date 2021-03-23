from __future__ import print_function

from keras.utils import plot_model
from keras.layers import *
from keras.models import Model
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
import time

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

Ridership = np.load('Ridership{}.npy'.format(Station))
Features = np.load('Features{}.npy'.format(Station))
TrueValue = np.load('TrueValue{}.npy'.format(Station))
LocalMap = np.load('LocalMap{}.npy'.format(Station))

MapSize = LocalMap.shape[0]

print(Ridership.shape)
print(Features.shape)
print(TrueValue.shape)
print(LocalMap.shape)

Ridership = Ridership.reshape(-1,Ridership.shape[-1])
RidershipScaler = MinMaxScaler(feature_range=(0, 1))
Ridership = RidershipScaler.fit_transform(Ridership)

Ridership = Ridership.reshape(-1,TimeStep,Ridership.shape[-1])

TrueValueScaler = MinMaxScaler(feature_range=(0, 1))
TrueValue = TrueValueScaler.fit_transform(TrueValue)

LocalMap = np.expand_dims(LocalMap,axis=0)
LocalMap = np.repeat(LocalMap,TrueValue.shape[0],axis=0)

RidershipList = [[],[],[]]
for i in range(TimeStep):
    RidershipList[i] = Ridership[:,i,:]
    RidershipList[i] = RidershipList[i].reshape(RidershipList[i].shape[0],RidershipList[i].shape[1],1)
    print(RidershipList[i].shape)

train_X,test_X,train_Map,test_Map,train_y,test_y = train_test_split(RidershipList[2],LocalMap,TrueValue,test_size = 0.1,random_state = 5)
# split = 96
# train_X,test_X,train_Map,test_Map,train_y,test_y = []

X_in = Input(shape=(MapSize,1))
Map_in = Input(shape=(MapSize,MapSize))
gcn1 = GraphConv(4)([X_in,Map_in])
gcn2 = GraphConv(4)([gcn1,Map_in])
OutPut = Flatten()(gcn2)
OutPut = Dense(MapSize,activation="relu")(OutPut)
OutPut = Dense(1)(OutPut)

model = Model(inputs = [X_in,Map_in],output = OutPut)
model.compile(loss='mse', optimizer=Adam(lr=0.01),metrics=[r2])

history = model.fit([train_X,train_Map],train_y,batch_size=train_X.shape[0], epochs=100, shuffle=False, verbose=1,validation_split=0.2)

model.summary()
from keras.utils import plot_model
plot_model(model,to_file="单个GCN预测模型.png",show_shapes=True)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.legend()
pyplot.show()

output = model.predict([test_X,test_Map])
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

# print('Test RMSE: %.3f' % rmse)
# print('Test MAE: %.3f' % mae)
# print('Test R2: %.3f' % r2)
# print('Test explained_variance_score: %.3f' % score)
# print('Test MAPE: %.3f' % mape)

print('%.3f' % rmse)
print('%.3f' % mae)
print('%.3f' % r2)
print('%.3f' % score)
print('%.3f' % mape)

pyplot.plot(output, label='real')
pyplot.plot(test_y, label='pre')
pyplot.legend()
pyplot.show()
model.save("keras模型导出/GCN-Station{}.h5".format(Station))