import time
import win32com.client
import pandas as pd

class Creon:
    def __init__(self):
        #각종 코드 정보 및 코드 리스트를 얻을 수 있음
        self.obj_CpCodeMgr = win32com.client.Dispatch("CpUtil.CpCodeMgr")
        #CYBOS의 각종 상태를 알 수 있음
        self.obj_CpCybos = win32com.client.Dispatch("CpUtil.CpCybos")
        #주식의 차트데이터를 수신함
        self.obj_StockChart = win32com.client.Dispatch("CpSysDib.StockChart")

    """ 데이터 요청 함수 """
    def creon_7400_주식차트조회(self, code, date_from, date_to):
        #통신연결상태 확인 0:연결끊김, 1:연결정상
        b_connected = self.obj_CpCybos.IsConnect
        if b_connected == 0:
            print("연결 실패")
            return None

        list_field_key = [0, 1, 2, 3, 4, 5, 8] #날짜(0), 시간(1), 시가(2), 고가(3), 저가(4), 종가(5), 거래량(8)
        list_field_name =["date", "time", "open", "high", "low", "close", "volume"]
        dict_chart = {name: [] for name in list_field_name}

        #obj_StockChart.SetInputValue(type, value): type에 해당하는 입력값을 value로 설정
        self.obj_StockChart.SetInputValue(0, "A"+code) #0: 종목코드 타입, 주식(A003540)
        self.obj_StockChart.SetInputValue(1, ord("1")) #1: 요청구분 타입, "1": 기간
        self.obj_StockChart.SetInputValue(2, date_to) #2: 요청종료일 타입
        self.obj_StockChart.SetInputValue(3, date_from) #3: 요청시작일 타입
        self.obj_StockChart.SetInputValue(5, list_field_key) #5:필드값 타입
        self.obj_StockChart.SetInputValue(6, ord("D")) #차트구분 타입, "D": 일
        self.obj_StockChart.BlockRequest() #데이터 요청

        """ 요청 결과 및 출력물 가져오기 """
        status = self.obj_StockChart.GetDibStatus() #요청결과 상태: 0이면 비정상
        msg = self.obj_StockChart.GetDibMsg1()
        print("통신상태: {} {}".format(status, msg))
        if status != 0:
            return None

        cnt = self.obj_StockChart.GetHeaderValue(3) #3: 수신개수
        for i in range(cnt):
            dict_item = ( {name:self.obj_StockChart.GetDataValue(pos, i)
                           for pos, name in zip(range(len(list_field_name)), list_field_name)} )

            for k, v in dict_item.items():
                dict_chart[k].append(v)

        print("차트: {} {}".format(cnt, dict_chart))
        return pd.DataFrame(dict_chart, columns=list_field_name)

if __name__ == "__main__":
    creon = Creon()
    print(creon.creon_7400_주식차트조회("035420", 20150101, 20171231))

