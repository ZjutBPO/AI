from pandas import read_csv
from matplotlib import pyplot
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import numpy as np

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)

# load dataset
dataset = read_csv('预测总出行人数/date-num-COVID.csv')
data = dataset[14:]
data.fillna(0,inplace = True)
data["Date"] = data["Date"].map(lambda item:item[5:10])

values = data.values

one,two,three=[],[],[]
vone,vtwo,vthree=[],[],[]
index = 0

for item in values:
    if item[7] == 1:
        one.append(item[0])
        vone.append(item[1])
    elif item[7] == 2:
        two.append(item[0])
        vtwo.append(item[1])
    elif item[7] == 3:
        three.append(item[0])
        vthree.append(item[1])
    index += 1

pyplot.plot(values[:,0],values[:,1])
pyplot.scatter(one,vone,c='red')
pyplot.scatter(two,vtwo,c='blue')
pyplot.scatter(three,vthree,c='green')
pyplot.gca().xaxis.set_major_locator(MultipleLocator(10))
pyplot.show()