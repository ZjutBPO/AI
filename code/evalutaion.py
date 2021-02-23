from math import sqrt
import pandas as pd
import numpy as np
import keras.backend as K
import keras.backend as K
from sklearn.metrics import mean_squared_error,mean_absolute_error

def r2(y_true, y_pred):
    SS_reg = K.sum(K.square(y_pred - y_true))
    mean_y = K.mean(y_true)
    SS_tot = K.sum(K.square(y_true - mean_y))
    f = 1 - SS_reg/SS_tot
    return f

def R2(y_true, y_pred):
    SS_reg = np.sum((y_pred - y_true)**2)
    mean_y = y_true.mean()
    SS_tot = np.sum((y_true - mean_y)**2)
    f = 1 - SS_reg/SS_tot
    return f

def r2(y_true, y_pred):
    SS_reg = K.sum(K.square(y_pred - y_true))
    mean_y = K.mean(y_true)
    SS_tot = K.sum(K.square(y_true - mean_y))
    f = 1 - SS_reg/SS_tot
    return f

def MAPE(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)


def get_evaluation(inv_y, inv_yhat):
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('|RMSE|%.3f|' % rmse)
    Test_R2 = R2(inv_y, inv_yhat)
    print('|R2|%.3f|' % Test_R2)
    mae = mean_absolute_error(inv_y, inv_yhat)
    print('|MAE|%.3f|' % mae)
    mape = MAPE(inv_y, inv_yhat)
    print('|MAPE|%.3f|' % mape)