from keras.models import Sequential,Model
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# 期望输入数据尺寸: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,input_shape=(8, 16),name = "L1"))  # 返回维度为 32 的向量序列，返回(100, 8, 32)
model.add(LSTM(32, return_sequences=True,name="L2"))  # 返回维度为 32 的向量序列，返回(100, 8, 32)
model.add(LSTM(32,name="L3"))  # 返回维度为 32 的单个向量，返回(100, 32)
model.add(Dense(10, activation='softmax',name="D1"))    #返回(100, 10)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 生成虚拟训练数据
x_train = np.random.random((1000, 8, 16))
y_train = np.random.random((1000, num_classes))

# 生成虚拟验证数据
x_val = np.random.random((100, 8, 16))
y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))

def get_layer(layer_name,model):
    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(x_val)
    # print(intermediate_output)
    print(intermediate_output.shape)

get_layer("L1",model)
get_layer("L2",model)
get_layer("L3",model)
get_layer("D1",model)