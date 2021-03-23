from __future__ import print_function

from keras.utils import plot_model
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import scipy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras_gcn import *
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

train_X_3,test_X_3,train_X_2,test_X_2,train_X_1,test_X_1,train_Map,test_Map,train_y,test_y = train_test_split(RidershipList[0],RidershipList[1],RidershipList[2],LocalMap,TrueValue,test_size = 0.1,random_state = 0)

train_X = [train_X_3,train_X_2,train_X_1,train_Map,train_Map,train_Map]
test_X = [test_X_3,test_X_2,test_X_1,train_Map,train_Map,train_Map]

X_in = Input(shape=(MapSize,1),name = "StationFeature")
Map_in = Input(shape=(MapSize,MapSize),name = "Map")
GCN1 = GraphConv(4,name="GCN1")([X_in,Map_in])
GCN2 = GraphConv(4,name="GCN2")([GCN1,Map_in])
Output = Flatten()(GCN2)
Output = Dense(64,name="ExtractFeature")(Output)

GCN_Model = Model(inputs = [X_in,Map_in],output = Output,name = "GCN_Part")
plot_model(GCN_Model,to_file="GCN+LSTM预测(GCN部分).png",show_shapes=True)

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
plot_model(model,to_file="GCN+LSTM预测(整体).png",show_shapes=True)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.legend()
pyplot.show()

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