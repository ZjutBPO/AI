import pandas as pd
import numpy as np
import random
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout

def ReadFromCsv(filename):
    data = pd.read_csv(filename)
    data["Time"] = data["Time"].map(lambda item:item[-2:])
    data = data.drop(["OutTime","InTime"],axis=1)
    data = data.drop(["origin","destination"],axis=1)
    data = data.fillna(0)
    return data

SupplementaryTrips = ReadFromCsv("用户画像/SupplementaryTrips.csv")
print(SupplementaryTrips)

values = SupplementaryTrips.values


# pd.DataFrame(scaled).to_csv("用户画像/tmp.csv",index = 0)

time_step = 1
n_features = 3

# print(random.randint(scaled.shape[0]))

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]


n_train = - 541
train,test = values[:n_train,:],values[n_train:-1,:],
train_x,train_y = train[:,:3],train[:,-1]
test_x,test_y = test[:,:3],test[:,-1]

# 将train_x和test_x归一化
scaler = MinMaxScaler(feature_range=(0, 1))
train_x = scaler.fit_transform(train_x)
test_x = scaler.fit_transform(test_x)

# 将train_y和test_y转换成one-hot编码

train_y = train_y.astype("int")
test_y = test_y.astype("int")
train_y = convert_to_one_hot(train_y,2)

print(train_y)

train_x = train_x.reshape((train_x.shape[0],time_step,n_features))
test_x = test_x.reshape((test_x.shape[0],time_step,n_features))
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

batch_size = 18

model = Sequential()
model.add(LSTM(256, activation="relu",batch_input_shape=(batch_size,train_x.shape[1], train_x.shape[2]),return_sequences=True,stateful=True))
# model.add(LSTM(256, activation="relu",return_sequences=True))
model.add(LSTM(256, activation="relu"))
model.add(Dropout(0.3))
# model.add(Dense(128))
model.add(Dense(16))
model.add(Dense(2,activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


for i in range(100):
    history = model.fit(train_x, train_y, epochs=1, batch_size=batch_size, validation_split=0.25, verbose=1, shuffle=False)
    print(history.history['loss'],history.history['val_loss'])
    # loss.append(history.history['loss'])
    # val_loss.append(history.history['val_loss'])
    # model3.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()


pyplot.plot(history.history['loss'], label='loss')
pyplot.plot(history.history['accuracy'], label='acc')
pyplot.plot(history.history['val_accuracy'], label='val_acc')
pyplot.legend()
# pyplot.show()

yhat = model.predict(test_x,batch_size=batch_size)
yhat = pd.DataFrame(yhat)
yhat = yhat.idxmax(axis=1)

output = pd.DataFrame(test_y,columns=["ture"])
output.insert(1,"test",yhat)

output.to_csv("用户画像/输出比较.csv",index = 0)