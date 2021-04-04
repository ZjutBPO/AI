import requests
import numpy as np
import json
import os

param = {"instances": [[[0.8655992150306702, 0.5, 0.0833333432674408, 0.0], 
                [0.4750121235847473, 0.5, 0.0833333432674408, 0.0],
                [0.4444444477558136, 0.5, 0.0833333432674408, 0.0],
                [0.2531645596027374, 0.5, 0.0833333432674408, 0.0]]]}
param = json.dumps(param)
res = requests.post('http://115.200.1.183:8502/v1/models/LSTM:predict', data=param)
print(res.text)
os.system("pause");