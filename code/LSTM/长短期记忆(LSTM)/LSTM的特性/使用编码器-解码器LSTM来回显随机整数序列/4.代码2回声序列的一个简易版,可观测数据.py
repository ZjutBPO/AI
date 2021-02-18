from random import randint
from numpy import array
from numpy import argmax
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
import pandas
import numpy
"""
��ʾ�����ñ��볤��Ϊ10�����ڹ۲�
"""
pandas.set_option('display.width', 1000)  # �����ַ���ʾ���
pandas.set_option('display.max_rows', None)  # ������ʾ�����
pandas.set_option('display.max_columns', None)  # ������ʾ�����
## generate a sequence of random numbers in [0, 99]
#def generate_sequence(length=25):
#    return [randint(0, 99) for _ in range(length)]
#
## ����һ��one hot encode ����
#def one_hot_encode(sequence, n_unique=100):
#    encoding = list()
#    for value in sequence:
#        vector = [0 for _ in range(n_unique)]
#        vector[value] = 1
#        encoding.append(vector)
#    return array(encoding)


def generate_sequence(length=5):
    return [randint(0, 9) for _ in range(length)]

# ����һ��one hot encode ����
def one_hot_encode(sequence, n_unique=10):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)

# ����one hot encoded ���ַ���
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]

# ����������ת��Ϊ�ලѧϰ
def to_supervised(sequence, n_in, n_out):
    # �������е��ͺ󸱱�
    df = DataFrame(sequence)
    #
    df = concat([df.shift(n_in-i-1) for i in range(n_in)], axis=1)
    print(df)
    # ɾ��ȱʧ���ݵ���
    df.dropna(inplace=True)
    # ָ�����������Ե���
    values = df.values
    print(values)
    width = sequence.shape[1]
    X = values.reshape(len(values), n_in, width)
    y = values[:, 0:(n_out*width)].reshape(len(values), n_out, width)
    print(X)
    print(y)
    return X, y

# ΪLSTM׼������
def get_data(n_in, n_out):
    # �����������
    sequence = generate_sequence()
    # one hot encode
    encoded = one_hot_encode(sequence)
    # convert to X,y pairs
    X,y = to_supervised(encoded, n_in, n_out)
    return X,y

# ���� LSTM

n_in = 4
n_out = 4
encoded_length = 10
#encoded_length = 100
batch_size = 2
model = Sequential()
#����һ����״̬������
model.add(LSTM(20, batch_input_shape=(batch_size, n_in, encoded_length), return_sequences=True, stateful=True))
model.add(TimeDistributed(Dense(encoded_length, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# train LSTM
for epoch in range(500):
    # �����������
    X,y = get_data(n_in, n_out)
    # ����ģ��ִ��һ��ʱ������
    model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()
# �������� LSTM������һ��Ԥ�⿴��Ч��
X,y = get_data(n_in, n_out)
yhat = model.predict(X, batch_size=batch_size, verbose=0)
# ����one hot wncoder����
for i in range(len(X)):
    print('Expected:', one_hot_decode(y[i]), 'Predicted', one_hot_decode(yhat[i]))