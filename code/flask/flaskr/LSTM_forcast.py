from flask import (Blueprint,flash,g,redirect,render_template,request,url_for)
from werkzeug.exceptions import abort

from flask.json import jsonify
import joblib
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import requests
import json

TimeInterval = 15
bp = Blueprint('forcast_{}'.format(TimeInterval),__name__,url_prefix='/forcast_{}'.format(TimeInterval))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def GetParameter(request,ParameterName):
    try:
        parameter = request.form[ParameterName]
    except KeyError:
        parameter = request.args.get(ParameterName)
    return parameter

@bp.route('/forcast.do',methods=('GET','Post'))
def forcast():
    st = time.time()
    # 需要预测的站点
    Station = GetParameter(request,"Station")
    if Station is None:
        Station = 46
    Station = int(Station)

    # 需要预测的时间
    PredictTime = GetParameter(request,"PredictTime")
    if PredictTime is None:
        PredictTime = "2019-01-08 18:17:44"
    PredictTime = datetime.datetime.strptime(PredictTime,"%Y-%m-%d %H:%M:%S")

    # 时间间隔
    TimeInterval = GetParameter(request,"TimeInterval")
    if TimeInterval is None:
        TimeInterval = 15
    TimeInterval = int(TimeInterval)
    TimeStep = 3
    
    Y_scaler = joblib.load("Predict_Data_{}/Station{}_Y_scaler".format(TimeInterval,Station))
    data = pd.read_csv("Predict_Data_{}/Station{}.csv".format(TimeInterval,Station))
    PredictTime = PredictTime.replace(minute = int(PredictTime.minute / TimeInterval) * TimeInterval,second=0)
    PredictTime = PredictTime.strftime("%Y-%m-%d %H:%M:%S")
    record = data.loc[data["PredictTime"] == PredictTime].values.reshape(-1)[1:].reshape(1,4,4)
    
    param = {"instances": record}
    param = json.dumps(param, cls=NumpyEncoder)
    res = requests.post('http://localhost:8502/v1/models/LSTM:predict', data=param)

    predictions = json.loads(res.text)
    PredictiveValue = Y_scaler.inverse_transform(predictions["predictions"])[0][0]

    return {"PredictiveValue":PredictiveValue}