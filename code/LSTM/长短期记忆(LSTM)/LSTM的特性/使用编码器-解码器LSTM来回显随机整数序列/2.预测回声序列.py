from random import randint
from numpy import array
from numpy import argmax
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed

# generate a sequence of random numbers in [0, 99]
def generate_sequence(length=25):
    return [randint(0, 99) for _ in range(length)]

# 生成一个one hot encode 序列
def one_hot_encode(sequence, n_unique=100):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)

# 解码one hot encoded 的字符串
def one_hot_decode(encoded_seq):
        return [argmax(vector) for vector in encoded_seq]

# 将编码序列转换为监督学习
def to_supervised(sequence, n_in, n_out):
    # 创建序列的滞后副本
    df = DataFrame(sequence)
    # 每个df是一个25*100的矩阵，此时横向将5个25*100的矩阵拼接起来，成为25*500的矩阵
    # 这5个矩阵块，依次向下位移4,3,2,1,0个竖直单位
    df = concat([df.shift(n_in-i-1) for i in range(n_in)], axis=1)
    # 删除缺失数据的行
    df.dropna(inplace=True)
    # 指定输入和输出对的列
    values = df.values
    width = sequence.shape[1]
    X = values.reshape(len(values), n_in, width)
    y = values[:, 0:(n_out*width)].reshape(len(values), n_out, width)
    #print(X)
    #print(y)
    return X, y

# 为LSTM准备数据
def get_data(n_in, n_out):
    # 生成随机序列
    sequence = generate_sequence()
    # one hot encode
    encoded = one_hot_encode(sequence)
    # convert to X,y pairs
    X,y = to_supervised(encoded, n_in, n_out)
    return X,y

# 定义 LSTM
n_in = 5
n_out = 5
encoded_length = 100
batch_size = 7
model = Sequential()
#构建一个有状态的网络
model.add(LSTM(20, batch_input_shape=(batch_size, n_in, encoded_length), return_sequences=True, stateful=True))
model.add(TimeDistributed(Dense(encoded_length, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# train LSTM
for epoch in range(500):
    # 生成随机序列
    X,y = get_data(n_in, n_out)
    # 调用模型执行一个时间周期
    model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()
# 评估网络 LSTM，进行一个预测看看效果
X,y = get_data(n_in, n_out)
yhat = model.predict(X, batch_size=batch_size, verbose=0)
# 解码one hot wncoder数据
for i in range(len(X)):
    print('Expected:', one_hot_decode(y[i]), 'Predicted', one_hot_decode(yhat[i]))