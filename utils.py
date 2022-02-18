import time
import datetime
import numpy as np

# 날짜, 시간 관련 문자열 형식
FORMAT_DATE = "%Y%m%d"
FORMAT_DATETIME = "%Y%m%d%H%M%S"

def get_today_str():
    a = datetime.datetime.today()
    b = datetime.datetime.min.time()
    today = datetime.datetime.combine(a, b)
    today_str = today.strftime(FORMAT_DATE)

    # 이렇게 해도 되지 않나?
    # today = datetime.datetime.today()
    # today_str = today.strftime(FORMAT_DATE)
    return today_str

def get_time_str():
    a = int(time.time())
    b = datetime.datetime.fromtimestamp(a)
    time_str = b.strftime(FORMAT_DATETIME)
    return time_str

def sigmoid(x):
    return 1. / (1. + np.exp(-x))