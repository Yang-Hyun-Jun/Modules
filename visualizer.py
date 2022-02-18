#가시화 모듈: 보유 주식 수, 가치 신경망, 투자 행동, 포트폴리오 가치 등 시각화

#prepare(): Figure를 초기화하고 일봉 차트 출력
#plot(): Figure를 그림 파일로 저장
#clear(): 일봉차트를 제외한 나머지 차트를 초기화

"Figure는 다음과 같은 5행 1열 구조"
#Figure 제목: 파라미터, 에포크 및 탐험률
#Axes 1: 종목의 일봉 차트
#Axes 2: 보유 주식 수 및 에이전트 행동 차트
#Axes 3: 가치 신경망 출력
#Axes 4: 정책 신경망 출력 및 탐험 차트
#Axes 5: 포트폴리오 가치 및 학습 지점 차트

#맷플롯립 컬러: 검정(k), 하양(w), 빨강(r), 파랑(b), 초록(g), 노랑(y)

import threading
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")

from mplfinance.original_flavor import candlestick_ohlc
from agent import Agent

lock = threading.Lock()

class Visualizer:
    COLORS = ["r", "b", "g"]

    def __init__(self, vnet=False):
        #Matplotlib의 Figure 클래스의 객체인 캔버스
        self.canvas = None
        #차트를 그리기 위한 Matplotlib의 Axes 클래스 객체
        self.fig = None
        self.axes = None
        self.title = ""

    """ 가시화 준비 함수 """
    def prepare(self, chart_data, title): #전체 과정에서 한번만 호출되는 함수
        self.title = title
        with lock:
            #5개 차트를 그릴 캔버스 세팅
            self.fig, self.axes = plt.subplots(nrows=5, ncols=1, facecolor="w", sharex=True)
            for ax in self.axes:
                #눈금이 지수형태(1e-3)와 같은 과학적 표기로 나타나는 것 비활성화
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)
                #y축 위치 오른쪽으로 변경
                ax.yaxis.tick_right()

            # ----- 차트1: 일봉 차트 -----
            #y축 레이블 작성
            self.axes[0].set_ylabel("Env.")
            x = np.arange(len(chart_data))

            #ohlc: open, high, low, close 순서로 된 2차원 배열
            #열 방향으로 인덱스 배열 x와 데이터 일부 np.array(chart_data)를 붙임
            ohlc = np.hstack( (x.reshape(-1, 1), np.array(chart_data)[:, 1:-1]) )
            #양봉은 빨간색, 음봉은 파란색으로 표시
            candlestick_ohlc(self.axes[0], ohlc, colorup="r", colordown="b")
            #거래량 가시화(bar plot으로 그리기)
            ax = self.axes[0].twinx() #x축 공유
            volume = np.array(chart_data)[:,-1].tolist()
            ax.bar(x, volume, color="b", alpha=0.3) #alpha: bar plot 투명도

    """ 가시화 함수 """
    def plot(self,
             epoch_str=None, num_epoches=None, epsilon=None,
             action_list=None, actions=None, num_stocks=None,
             outvals_value=[], outvals_policy=[], exps=None,
             learning_idxes=None, initial_balance=None, pvs=None):

        #epoch_str: Figure 제목으로 표시할 에포크/ num_epoches: 총 수행할 에포크 수
        #epsilon: 탐험률/ action_list: 에이전트 가능 행동 리스트/ actions: 에이전트가 수행한 행동 배열
        #outvals_value: 가치 신경망의 출력 배열/ outvals_policy: 정책 신경망의 출력 배열
        #exps: 탐험 여부 배열/ learning_idxes: 학습 위치 배열/ initial_balance: 초기 자본금
        #pvs: 포트폴리오 가치 배열 / num_stocks: 보유 주식 수 배열

        with lock:
            x = np.arange(len(actions)) #모든 차트가 공유할 x축 값
            #맷플롯립은 넘파이 배열을 입력으로 받음
            actions = np.array(actions) #에이전트가 한 행동 배열
            outvals_value = np.array(outvals_value) #가치 신경망 출력 배열
            outvals_policy = np.array(outvals_policy) #정책 신경망 출력 배열
            pvs_base = np.zeros(len(actions)) + initial_balance #초기 자본금 배열

            # ----- 차트2: 에이전트 상태 차트(행동, 보유 주식 수) -----
            for action, color in zip(actions, self.COLORS):
                for i in x[actions == action]:
                    #배경색으로 행동 표시(세로선 긋는 함수)
                    self.axes[1].axvline(i, color=color, alpha=0.1)
            #보유 주식 수 그리기
            self.axes[1].plot(x, num_stocks, "-k") # "-k" 검정색 실선

            # ----- 차트3: 가치 신경망 -----
            #outvals_value는 매수 가치값 열, 매도 가치값 열을 갖는 넘파이배열
            if len(outvals_value) > 0:
                max_actions = np.argmax(outvals_value, axis=1)
                for action, color in zip(actions, self.COLORS):
                    #배경 그리기
                    for idx in x:
                        if max_actions[idx] == action: #최대 가치를 갖는 행동이 실제 한 행동이면
                            self.axes[2].axvline(idx, color=color, alpha=0.1)
                    #가치 신경망 tanh 출력값 그리기
                    self.axes[2].plot(x, outvals_value[:, action], color=color, linestyle="-")

            # ----- 차트4: 정책 신경망, 탐험 여부 ------
            #탐험을 노란색 배경으로 그리기
            for exp_idx in exps:
                self.axes[3].axvline(exp_idx, color="y")

            #행동을 배경으로 그리기
            if len(outvals_policy) > 0:
                _outvals = outvals_policy
            else:
                _outvals = outvals_value

            for idx, outval in zip(x, _outvals):
                color = "white" #홀딩 하얀색
                if np.isnan(outval.max()):
                    continue
                if outval.max() == Agent.ACTION_BUY:
                    color = "r" #매수 빨간색
                elif outval.max() == Agent.ACTION_SELL:
                    color = "b" #매도 파란색
                self.axes[3].axvline(idx, color=color, alpha=0.1)

            #정책 신경망의 출력 그리기
            if len(outvals_policy) > 0:
                for action, color in zip(action_list, self.COLORS):
                    self.axes[3].plot(x, outvals_policy[:,action], color=color, linestyle="-")

            # ----- 차트5: 포트폴리오 가치 -----
            self.axes[4].axhline(initial_balance, linestyle="-", color="gray") #가로선 긋기
            self.axes[4].fill_between(x, pvs, pvs_base, where=pvs>pvs_base, facecolor="r", alpha=0.1) #곡선과 가로선 사이 빨간색
            self.axes[4].fill_between(x, pvs, pvs_base, where=pvs<pvs_base, facecolor="b", alpha=0.1) #곡선과 가로선 사이 파란색
            self.axes[4].plot(x, pvs, "-k")

            #학습 위치 표시
            for learning_idxes in learning_idxes:
                self.axes[4].axvline(learning_idxes, color="y")

            #에포크 및 탐험 비율
            self.fig.suptitle("{} \nEpoch:{}/{} e={:.2f}".format(self.title, epoch_str, num_epoches, epsilon))
            #캔버스 레이아웃 조정
            self.fig.tight_layout() #figure 크기 자동 조절
            self.fig.subplots_adjust(top=0.85) #sub title 위치 조절

    """ 가시화 정보 초기화 및 결과 저장 함수 """
    def clear(self, xlim):
        with lock:
            _axes = self.axes.tolist()

            for ax in _axes[1:]: #차트1 제외하고 초기화
                ax.cla() #그려진 차트 지우기
                ax.relim() #limit 초기화
                ax.autoscale() #스케일 재설정

            #y축 레이블 재설정
            self.axes[1].set_ylabel("Agent")
            self.axes[2].set_ylabel("V")
            self.axes[3].set_ylabel("P")
            self.axes[4].set_ylabel("PV")

            for ax in _axes:
                ax.set_xlim(xlim) #x축 limit 재설정
                ax.get_xaxis().get_major_formatter().set_scientific(False) #x축 과학적 표기 비활성화
                ax.get_yaxis().get_major_formatter().set_scientific(False) #y축 과학적 표기 비활성화
                ax.ticklabel_format(useOffset=False) #x축 간격을 일정하게 설정

    def save(self, path):
        with lock:
            self.fig.savefig(path)

















