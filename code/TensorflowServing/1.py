import json
import requests
 
url = 'http://192.168.99.110:8501/v1/models/half_plus_two:predict'
data = {"instances": [1.0, 2.0, 5.0]}
r =requests.post(url,json.dumps(data))
print(r)
print(r.text)
print(r.content)