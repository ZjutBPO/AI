import pandas as pd
import numpy as np

AllTimeCovid = pd.read_csv("用户画像/AllTime-Covid.csv")
UserTrips = pd.read_csv("用户画像/UserTrips.csv")

UserTrips = UserTrips.drop(["DiffConfirmedCount","DateType"],axis = 1)

SupplementaryTrips = pd.merge(AllTimeCovid,UserTrips,left_on="Time",right_on="InTime",how="left")
print(SupplementaryTrips)

SupplementaryTrips.to_csv("用户画像/SupplementaryTrips.csv",index = 0)