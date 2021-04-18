import pandas as pd
import numpy as np
import time
import datetime

for file_index in range(1,27):
    # file_index = 1
    st = time.time()

    if file_index < 10:
        data = pd.read_csv("天池/Metro_train/record_2019-01-0{}.csv".format(file_index))
    else:
        data = pd.read_csv("天池/Metro_train/record_2019-01-{}.csv".format(file_index))

    data = data.to_numpy()

    time_list = {}
    station_list = {}

    for item in data:
        # ['2019-01-01 02:52:05' 'C' 42 1951 1 'Da5d01f88151727ff061e6f32258bdceb' 3]
        # print(item)
        # 0123456789
        # 2019-01-01 02:00:05
        date = item[0][8:10]
        hour = item[0][11:13]
        minute = item[0][14:16]
        time_list[hour+minute] = 1
        station = item[2]
        station_list[station] = 1
        status = item[4]


    inverse_time_list = []
    index = 0

    for item in time_list:
        time_list[item] = index
        inverse_time_list.append(item)
        index += 1

    inverse_station_list = []
    index = 0

    for item in station_list:
        station_list[item] = index
        inverse_station_list.append(item)
        index += 1

    # print(len(time_list))
    # print(len(station_list))

    inverse_time_list = np.array(inverse_time_list)
    inverse_station_list = np.array(inverse_station_list)
    inverse_status = np.array([0,1])

    # print(inverse_time_list)
    # print(inverse_station_list)

    print(time.time() - st)

    flow_num = np.zeros((len(time_list),len(station_list),2,1))

    for item in data:
        date = item[0][8:10]
        hour = item[0][11:13]
        minute = item[0][14:16]
        station = item[2]
        status = item[4]
        flow_num[time_list[hour+minute],station_list[station],status] += 1

    flow_num = flow_num.astype("int32")
    output_list = []

    Time_Dimension_index = 0
    Station_Dimension_index = 0
    Status_Dimension_index = 0

    if file_index < 10:
        date_time_format = "2019-01-0{} {}:{}:00"
    else:
        date_time_format = "2019-01-{} {}:{}:00"

    for Time_Dimension in flow_num:

        for Station_Dimension in Time_Dimension:

            for Status_Dimension in Station_Dimension:
                date_time = date_time_format.format(file_index,inverse_time_list[Time_Dimension_index][0:2],inverse_time_list[Time_Dimension_index][2:])
                date_time = datetime.datetime.strptime(date_time,"%Y-%m-%d %H:%M:%S")
                date_stamp = int(time.mktime(date_time.timetuple()))
                output_list.append([date_stamp,inverse_time_list[Time_Dimension_index],inverse_station_list[Station_Dimension_index],inverse_status[Status_Dimension_index],Status_Dimension[0]])
                Status_Dimension_index += 1

            Status_Dimension_index = 0
            Station_Dimension_index += 1

        Station_Dimension_index = 0
        Time_Dimension_index += 1

    output = pd.DataFrame(output_list,columns=["Timestamp","Time","Station","Status","num"])

    if file_index < 10:
        output.to_csv("天池数据提取/15/1 minutes Extract data/2019-01-0{}.csv".format(file_index),index = 0)
    else:
        output.to_csv("天池数据提取/15/1 minutes Extract data/2019-01-{}.csv".format(file_index),index = 0)

    # print(output)
    print(time.time() - st)