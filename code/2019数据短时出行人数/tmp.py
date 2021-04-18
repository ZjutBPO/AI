import pandas as pd

output = []

for index in range(81):
    if index == 54:
        continue
    data = pd.read_csv("by{}minutes/sta{}.csv".format(15,index))
    output.append([index,data.max()["Num"]])

output = pd.DataFrame(output,columns=["站点","人数"])
output.to_csv("tmp.csv",index = 0)