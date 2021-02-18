import pandas as pd
import numpy as np

COVID = pd.read_csv("data/COVID-19.csv")
date_num = pd.read_csv("data/date-num.csv")
date_type = pd.read_csv("data/date.csv")
ans = pd.merge(date_num, COVID, on="Date", how='left', sort=False)
print(ans)
ans = pd.merge(ans, date_type, on="Date", how='left', sort=False)
print(ans)
ans.to_csv('预测总出行人数/date-num-COVID.csv',index = 0)