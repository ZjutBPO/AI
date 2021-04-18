import pandas as pd
import numpy as np
import time

# for file_index in range(1,27):
file_index = 26
st = time.time()

if file_index < 10:
    data = pd.read_csv("天池数据提取/15/1 minutes Extract data/2019-01-0{}.csv".format(file_index),dtype={
        "Time":np.str_,
        "Station":np.int32,
        "Status":np.int32,
        "num":np.int32})
else:
    data = pd.read_csv("天池数据提取/15/1 minutes Extract data/2019-01-{}.csv".format(file_index),dtype={
        "Time":np.str_,
        "Station":np.int32,
        "Status":np.int32,
        "num":np.int32})

data


# print(data)
# print(time.time()-st)

# flow_num = np.zeros((len(time_list),len(station_list),2,1))

# for item in data:
#     hour = item[0][0:2]
#     minute = item[0][2:]
#     if hour < 6:
#         continue
#     date_time = "2019-01-01 {}:{}:00".format(hour,minute)
#     print(date_time)
#     break

# # datetime.timestamp