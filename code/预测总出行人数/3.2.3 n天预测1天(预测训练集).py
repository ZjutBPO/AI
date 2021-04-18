from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout


from math import sqrt
import pandas as pd
import numpy as np
import keras.backend as K
import keras.backend as K
from sklearn.metrics import mean_squared_error,mean_absolute_error

def r2(y_true, y_pred):
    SS_reg = K.sum(K.square(y_pred - y_true))
    mean_y = K.mean(y_true)
    SS_tot = K.sum(K.square(y_true - mean_y))
    f = 1 - SS_reg/SS_tot
    return f

def R2(y_true, y_pred):
    SS_reg = np.sum((y_pred - y_true)**2)
    mean_y = y_true.mean()
    SS_tot = np.sum((y_true - mean_y)**2)
    f = 1 - SS_reg/SS_tot
    return f

def r2(y_true, y_pred):
    SS_reg = K.sum(K.square(y_pred - y_true))
    mean_y = K.mean(y_true)
    SS_tot = K.sum(K.square(y_true - mean_y))
    f = 1 - SS_reg/SS_tot
    return f

def MAPE(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)


def get_evaluation(inv_y, inv_yhat):
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('|RMSE|%.3f|' % rmse)
    Test_R2 = R2(inv_y, inv_yhat)
    print('|R2|%.3f|' % Test_R2)
    mae = mean_absolute_error(inv_y, inv_yhat)
    print('|MAE|%.3f|' % mae)
    mape = MAPE(inv_y, inv_yhat)
    print('|MAPE|%.3f|' % mape)

pd.set_option("display.width",None)
pd.set_option("display.max_rows",None)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # 获取特征值数量n_vars
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
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
    agg = concat(cols, axis=1)
    # print(agg)
    # 重定义列名
    agg.columns = names
    # print(agg)
    # 删除空值
    if dropnan:
    	agg.dropna(inplace=True)
    return agg


# load dataset
dataset = read_csv('预测总出行人数/date-num-COVID-diff.csv')
# 删掉那些我们不想预测的列
dataset = dataset.drop(["Date","SuspectedCount",'CurrentConfirmedCount','DeadCount','DiffConfirmedCount','CuredCount'],axis=1)
values = dataset.values

# 数据转换为浮点型
values = values.astype('float32')
# 将所有数据缩放到（0，1）之间
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
scaled = DataFrame(scaled)
scaled.columns=['Num','ConfirmedCount','DateType']

n_days = 5
n_features = len(scaled.columns)

# 将数据格式化成监督学习型数据
reframed = series_to_supervised(scaled, n_days, 1)
reframed = reframed.iloc[:,:n_days * n_features + 1 ]
# reframed.dropna(inplace=True)

# split into train and test sets
values = reframed.values
print(values)
# 取出一年的数据作为训练数据，剩下的做测试数据
n_train = -31
train = values[:n_train, :]
test = values[n_train:, :]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# 将输入数据转换成3D张量 [samples, timesteps, features]，[n条数据，每条数据1个步长，13个特征值]
train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
test_X = test_X.reshape((test_X.shape[0], n_days, n_features))
# 最终生成的数据形状，X:(152,1,7)  Y:(152,)
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# 设计网络结构
model = Sequential()
model.add(LSTM(300, activation="relu",input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
# model.add(LSTM(256, activation="relu",return_sequences=True))
model.add(LSTM(300, activation="relu"))
# model.add(Dropout(0.2))
# model.add(Dense(128))
# model.add(Dense(16))
model.add(Dropout(0.15))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam',metrics=[r2])
# 拟合网络
history = model.fit(train_X, train_y, epochs=200, batch_size=train_X.shape[0], validation_data=(test_X, test_y), shuffle=False)
# model.save("预测总出行人数/testmodle.h5")

model.summary()
from keras.utils import plot_model
plot_model(model,to_file="预测总出行人数/lstm预测模型.png",show_shapes=True)

# 图像展示训练损失
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

yhat = model.predict(train_X)
print(yhat)

#重构预测数据形状，进行逆缩放
# 之所有下面奇怪的数据拼接操作，是因为：数据逆缩放要求输入数据的形状和缩放之前的输入保值一致
# 将3D转换为2D
train_X = train_X.reshape((train_X.shape[0], n_days*n_features))
# 拼接y，x为[y,x],即将test_X中的第一列数据替换成预测出来的yhat值
inv_yhat = concatenate((yhat, train_X[:, -(n_features-1):]), axis=1)
print(inv_yhat)
# 对替换后的inv_yhat预测数据逆缩放
inv_yhat = scaler.inverse_transform(inv_yhat)
# 逆缩放后取出第一列(预测列)y
inv_yhat = inv_yhat[:,0]


# 重构真实数据形状，进行逆缩放
train_y = train_y.reshape((len(train_y), 1))
# 因为test_X的第一列数据在上面被修改过，这里要重新还原一下真实数据。将test_X的第一列换成原始数据test_y值
inv_y = concatenate((train_y, train_X[:, -(n_features-1):]), axis=1)
# 对重构后的数据进行逆缩放
inv_y = scaler.inverse_transform(inv_y)
# 逆缩放后取出第一列(真实列)y
inv_y = inv_y[:,0]
print(inv_y)
print(inv_yhat)
# 计算预测列和真实列的误差RMSE值
get_evaluation(inv_y, inv_yhat)

pyplot.plot(inv_y, label='real')
pyplot.plot(inv_yhat, label='pre')
pyplot.legend()
pyplot.show()
