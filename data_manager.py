#데이터 관리 모듈 data_manager
#두가지 버전으로 자질 벡터 구성

import numpy as np
import pandas as pd

COLUMNS_CHART_DATA = ["date", "open", "high", "low", "close", "volume"]

#last: 전일
#차트 데이터만으로 구성
COLUMNS_TRAINING_DATA_V1 = [
    "open_lastclose_ratio", "high_close_ratio", "low_close_ratio",
    "close_lastclose_ratio", "volume_lastvolume_ratio",
    "close_ma5_ratio", "volume_ma5_ratio",
    "close_ma10_ratio", "volume_ma10_ratio",
    "close_ma20_ratio", "volume_ma20_ratio",
    "close_ma60_ratio", "volume_ma60_ratio",
    "close_ma120_ratio", "volume_ma120_ratio"]

#차트 데이터와 분석 지표로 구성
COLUMNS_TRAINING_DATA_V2 = [
    "per", "pbr", "roe",
    "open_lastclose_ratio", "high_close_ratio", "low_close_ratio",
    "close_lastclose_ratio", "volume_lastvolume_ratio",
    "close_ma5_ratio", "volume_ma5_ratio",
    "close_ma10_ratio", "volume_ma10_ratio",
    "close_ma20_ratio", "volume_ma20_ratio",
    "close_ma60_ratio", "volume_ma60_ratio",
    "close_ma120_ratio", "volume_ma120_ratio",
    "market_kospi_ma5_ratio", "market_kospi_ma20_ratio",
    "market_kospi_ma60_ratio", "market_kospi_ma120_ratio",
    "bond_k3y_ma5_ratio", "bond_k3y_ma20_ratio",
    "bond_k3y_ma60_ratio", "bond_k3y_ma120_ratio"]

""" 데이터 전처리 함수 """
def preprocess(data):
    windows = [5, 10, 20, 60, 120]

    for window in windows:
        #종가 이동평균
        data["close_ma{}".format(window)] = data["close"].rolling(window).mean()
        #거래량 이동평균
        data["volume_ma{}".format(window)] = data["volume"].rolling(window).mean()
        #이동평균 종가 비율
        data["close_ma{}_ratio".format(window)] = \
            (data["close"] - data["close_ma{}".format(window)] / data["close_ma{}".format(window)])
        #이동평균 거래량 비율
        data["volume_ma{}_ratio".format(window)] = \
            (data["volume"] - data["volume_ma{}".format(window)] / data["volume_ma{}".format(window)])
    #시가/전일종가 비율
    data["open_lastclose_ratio"] = np.zeros(len(data))
    data.loc[1:, "open_lastclose_ratio"] = \
        (data["open"][1:].values - data["close"][:-1].values) / data["close"][:-1].values
    #고가/종가 비율
    data["high_close_ratio"] = \
        (data["high"].values - data["close"].values) / data["close"].values
    #저가/종가 비율
    data["low_close_ratio"] = \
        (data["low"].values - data["close"].values) / data["close"].values
    #종가/전일 종가 비율
    data["close_lastclose_ratio"] = np.zeros(len(data))
    data.loc[1:, "close_lastclose_ratio"] = \
        (data["close"][1:].values - data["close"][:-1].values) / data["close"][:-1].values
    #거래량/전일 거래량 비율
    data["volume_lastvolume_ratio"] = np.zeros(len(data))
    data.loc[1:, "volume_lastvolume_ratio"] = \
        (data["volume"][1:].values - data["volume"][:-1].values) \
        /data["volume"][:-1]\
            .replace(to_replace=0, method="ffill")\
            .replace(to_replace=0, method="bfill").values

    return data

""" 차트 데이터 및 학습 데이터 로드 함수 """
""" 강화학습 실행 모듈에서 호출 """
def load_data(fpath, date_from, date_to, ver="v2"):
    # header = None if ver == "v1" else 0
    data = pd.read_csv(fpath, thousands=",", header=0, converters={"date":lambda x:str(x)})

    #데이터 전처리
    data = preprocess(data)
    #기간 필터링
    data["date"] = data["date"].str.replace("-", "")
    data = data[(data["date"] >= date_from) & (data["date"] <= date_to)]
    data = data.dropna()
    #차트 데이터 분리
    chart_data = data[COLUMNS_CHART_DATA]
    #학습 데이터 분리
    training_data = None
    if ver == "v1":
        training_data = data[COLUMNS_TRAINING_DATA_V1]
    elif ver == "v2":
        data.loc[:, ["per", "pbr", "roe"]] = data[["per", "pbr", "roe"]].apply(lambda x:x/100)
        training_data = data[COLUMNS_TRAINING_DATA_V2]
        training_data = training_data.apply(np.tanh)
    else:
        raise Exception("Invalid version.")

    return chart_data, training_data












