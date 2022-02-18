import numpy as np
import utils

class Agent:
    """ 에이전트 상수 정의 """
    #에이전트 상태가 구성하는 값 개수 (주식 보유 비율, 포트폴리오 가치 비율)
    STATE_DIM = 2
    #수수료 (일반적으로 0.015%)
    TRADING_CHARGE = 0.00015
    #거래세 (실제 0.25%)
    TRADING_TEX = 0.0025

    # TRADING_CHARGE = 0 #수수료 미적용
    # TRADING_TEX = 0 #거래세 미적용

    #행동
    ACTION_BUY = 0 #매수
    ACTION_SELL = 1 #매도
    ACTION_HOLD = 2 #홀딩

    #인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL]
    #인공 신경망에서 고려할 출력값의 개수
    NUM_ACTIONS = len(ACTIONS)

    def __init__(self,
                 environment, min_trading_unit = 1, max_trading_unit = 2,
                 delayed_reward_threshold = 0.05):
        #환경 객체
        self.environment = environment
        #최소, 최대 단일 거래 단위
        self.min_trading_unit = min_trading_unit
        self.max_trading_unit = max_trading_unit
        #지연보상 임계치
        self.delayed_reward_threshold = delayed_reward_threshold

        """ 에이전트 속성들 """
        self.initial_balance = 0 #초기 자본금
        self.balance = 0 #현재 현금 잔고
        self.num_stocks = 0 #보유 주식 수

        #포트폴리오 가치 = balance + num_stocks * {현재 주식 가격}
        self.portfolio_value = 0
        self.base_portfolio_value = 0 #직전 학습 시점의 pv

        self.num_buy = 0 #매도 횟수
        self.num_sell = 0 #매수 횟수
        self.num_hold  = 0 #홀딩 횟수

        self.immediate_reward = 0 #즉시 보상
        self.profitloss = 0 #현재 손익
        self.base_profitloss = 0 #직전 지연 보상 이후 손익

        self.exploration_base = 0 #탐험 행동 결정 기준

        """ Agent 클래스의 상태 """
        self.ratio_hold = 0 #주식 보유 비율
        self.ratio_portfolio_value = 0 #포트폴리오 가치 비율


    """ Agent 획득(Get) / 설정(Set) 메소드 """
    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0

        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def reset_exploration(self):
        self.exploration_base = 0.5 + np.random.rand()/2

    # 변동 된 초기자본금을 현재 잔고로 재설정
    # ex) 초기 자본금 1000에서 현재 잔고 800이면
    # 다음 학습에서 초기 잔고 = 800으로 재설정하고 시작
    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        #주식 보유 비율 = 보유 주식 수 / 보유 가능 최대 주식 수
        #           = 보유 주식 수 / PV / 현재 가격
        #           = 보유 주식 수 / (현재 현금 + 보유 주식 수 * 현재 가격) / 현재 가격
        #           = 보유 주식 수 / (현재 현금/현재 가격 + 보유 주식 수)
        self.ratio_hold = self.num_stocks / int(self.portfolio_value / self.environment.get_price())

        #포트폴리오 가치 비율 = 포트폴리오 가치/기준 포트폴리오 가치
        #               = 포트폴리오 가치/직전 표 달성 시 가치
        self.ratio_portfolio_value = self.portfolio_value / self.base_portfolio_value

        #상태값 리턴
        return (self.ratio_hold, self.ratio_portfolio_value)


    """ 행동 결정 메소드 """
    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0.0

        pred = pred_policy
        if pred is None:
            pred = pred_value

        if pred is None:
            #예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            #값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if all( pred==maxpred ):
                epsilon = 1

        #epsilon-greedy
        # 1) 엡실론이 더 크면 탐험
        if np.random.rand() < epsilon:
            exploration = True
            if np.random.rand() < self.exploration_base:
                action = Agent.ACTION_BUY #매수 기조로 탐험
            else:
                action = np.random.randint(Agent.NUM_ACTIONS - 1) + 1 #매수 액션: 0을 제외한 나머지 중 랜덤하게 추출
        # 2) 엡실론이 더 작으면 탐욕적으로 액션 선택
        else:
            exploration = False
            action = np.argmax(pred)

        confidence = 0.5
        if pred_policy is not None:
            confidence = pred[action]
        if pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration

    """ 행동 유효성 검사 메소드 """
    def validate_action(self, action):
        if action == Agent.ACTION_BUY:
            #적어도 1주를 살 수 있는지 확인
            #매수 가격 = 주당 가격 * (1 + 수수료) * 최소 거래 단위
            charged_price = self.environment.get_price() * (1 + Agent.TRADING_CHARGE) * self.min_trading_unit
            if self.balance < charged_price:
                return False
        elif action == Agent.ACTION_SELL:
            #주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                return False
        return True

    """ 신뢰(Confidence)에 따라 매수/매도 단위 결정 메소드 """
    # 정책 신경망이 결정한 행동의 신뢰도가 높을수록 매수 또는 매도 단위를 크게 정함
    # confidence = 1이면 return = self.max_trading_unit
    # confidence != 1이면 int(self.min_trading_unit * (1-confidence) + confidence * self.max_trading_unit)
    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_unit

        added_trading_unit = max(min(int(confidence * (self.max_trading_unit - self.min_trading_unit)),
                                     self.max_trading_unit - self.min_trading_unit), 0)
        return self.min_trading_unit + added_trading_unit


    """ 투자 행동 수행 함수(1) """
    def act(self, action, confidence):
        if not self.validate_action(action): #유효성 False이면 action을 관망으로 변경
            action = Agent.ACTION_HOLD

        #환경에서 가격 얻기
        curr_price = self.environment.get_price()
        #즉시 보상 초기화 (에이전트가 act할 때마다 결정되므로 초기화)
        self.immediate_reward = 0

        #매수 행동
        if action == Agent.ACTION_BUY:
            #매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            #수수료 적용해서 총 매수 금액
            invest_amount = curr_price*(1+self.TRADING_CHARGE)*trading_unit
            #매수 했을때 남을 현금
            balance_left = self.balance - invest_amount

            #보유 현금이 부족할 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance_left < 0:
                trading_unit = max(min(int(self.balance/(curr_price*(1+self.TRADING_CHARGE))), self.max_trading_unit),
                                   self.min_trading_unit)

            if invest_amount > 0:
                self.balance -= invest_amount #보유 현금 갱신
                self.num_stocks += trading_unit #보유 주식수 갱신
                self.num_buy += 1 #매수 횟수 증가

        #매도 행동
        elif action == Agent.ACTION_SELL:
            #매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            #거래세, 수수 적용해서 총 매도 금액
            invest_amount = curr_price*(1-(self.TRADING_CHARGE+self.TRADING_TEX))*trading_unit

            if invest_amount > 0:
                self.balance += invest_amount #보유 현금을 갱신
                self.num_stocks -= trading_unit #보유 주식수를 갱신
                self.num_sell += 1 #매도 횟수 증가

        #홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1 #홀딩 횟수 증가

        #포트폴리오 가치 갱신
        self.portfolio_value = self.balance + self.num_stocks*curr_price
        self.profitloss = (self.portfolio_value - self.initial_balance)/self.initial_balance
        self.base_profitloss = (self.portfolio_value - self.base_portfolio_value)/self.base_portfolio_value
        #즉시보상 - 수익률
        self.immediate_reward = self.profitloss
        #지연 보상 - 익절, 손절 기준
        #지연 보상 임계치(delayed_reward_threshold)를 초과하는 경우 즉시 보상값으로 하고 그 외 0
        #RLTRADER는 지연 보상이 0이 아닌 경우 학습을 수행한다.
        delayed_reward = 0

        if self.base_profitloss > self.delayed_reward_threshold or \
            self.base_profitloss < -self.delayed_reward_threshold:
            #목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
            delayed_reward = self.profitloss

        else:
            delayed_reward = 0

        return self.immediate_reward, delayed_reward




















