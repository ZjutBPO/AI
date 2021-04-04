import requests

res = requests.get("http://115.200.1.183:5000/forcast_15/forcast.do?Station=1")
print(res)