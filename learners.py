#학습기 모듈 (learner)은 다양한 강화학습 방식을 수행하기 위한 학습기 클래스들을 가진다.

import os
import logging
import abc
import collections
import threading
import time
import numpy as np

from utils import sigmoid
from environment import Environment
from agent import Agent
from networks import Network, DNN, LSTMNetwork, CNN
from visualizer import Visualizer

class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta #상속을 위한 추상 클래스, 인스턴스 생성은 안된다.
    lock = threading.Lock()

    """ 초기화 """
    def __init__(self,
                 rl_method="rl", stock_code=None, chart_data=None,
                 training_data=None, min_trading_unit=1, max_trading_unit=2,
                 delayed_reward_threshold=0.05, net="dnn", num_steps=1,
                 lr=0.001, value_network=None, policy_network=None,
                 output_path="", reuse_models=True):

        #rl_method: 강화학습 기법/ chart_data: 주식 일봉 차트/ training_data: 전처리된 학습용 데이터
        #net: 신경망 종류/ n_steps: LSTM, CNN에서의 스텝 수/ outpath_path: 저장 경로

        #assert 가정 설정문으로 인자 조건 확인 (만족하지 않을 경우 Assertion error 발생)
        assert min_trading_unit > 0
        assert max_trading_unit > 0
        assert max_trading_unit >= min_trading_unit
        assert num_steps > 0
        assert lr > 0

        #강화학습 기법 설정
        self.rl_method = rl_method
        #환경 설정
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data=chart_data)
        #에이전트 설정
        self.agent = Agent(self.environment,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)
        #학습 데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        #벡터 크기(28) = 학습 데이터 벡터 크기(26) + 에이전트 상태 크기(2)
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        #신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        #가시화 모듈
        self.visualizer = Visualizer()
        #메모리
        self.memory_sample = [] #학습 데이터 샘플
        self.memory_action = [] #수행한 행동
        self.memory_reward = [] #획득한 보상
        self.memory_value = [] #수행한 행동의 예측 가치
        self.memory_policy = [] #수행한 행동의 예측 확률
        self.memory_pv = [] #포트폴리오 가치
        self.memory_num_stocks = [] #보유 주식 수
        self.memory_exp_idx = [] #탐험 위치
        self.memory_learning_idx = [] #학습 위치
        #에포크 관련 정보
        self.loss = 0.0
        self.itr_cnt = 0 #수행한 에포크 수
        self.exploration_cnt = 0 #무작위 투자 수행한 수
        self.batch_size = 0 #미니 배치 크기
        self.learning_cnt = 0 #한 에포크 동안 수행한 미니 배치 학습 횟수
        #로그 등 출력 경로
        self.output_path = output_path


    """ 가치 신경망 생성 함수 """
    def init_value_network(self, shared_network=None, activation="linear", loss="mse"):
        if self.net == "dnn":
            self.value_network = DNN(input_dim=self.num_features,
                                     output_dim=self.agent.NUM_ACTIONS, lr = self.lr,
                                     shared_network=shared_network,
                                     activation=activation, loss=loss)
        elif self.net == "lstm":
            self.value_network = LSTMNetwork(input_dim=self.num_features, num_steps=self.num_steps,
                                             output_dim = self.agent.NUM_ACTIONS, lr=self.lr,
                                             shared_network=shared_network,
                                             activation=activation, loss=loss)
        elif self.net == "cnn":
            self.value_network = CNN(input_dim=self.num_features, num_steps=self.num_steps,
                                     output_dim=self.agent.NUM_ACTIONS, lr=self.lr,
                                     shared_network=shared_network,
                                     activation=activation, loss=loss)

        #모델 reuse하고 지정된 경로에 파일이 존재하면 불러오기
        if self.reuse_models and os.path.exists(self.value_network_path):
            self.value_network.load_model(model_path=self.value_network_path)


    """ 정책 신경망 생성 함수 """
    def init_policy_network(self, shared_network=None, activation="sigmoid", loss="mse"):
        if self.net == "dnn":
            self.policy_network = DNN(input_dim=self.num_features,
                                     output_dim=self.agent.NUM_ACTIONS, lr = self.lr,
                                     shared_network=shared_network,
                                     activation=activation, loss=loss)
        elif self.net == "lstm":
            self.policy_network = LSTMNetwork(input_dim=self.num_features, num_steps=self.num_steps,
                                             output_dim = self.agent.NUM_ACTIONS, lr=self.lr,
                                             shared_network=shared_network,
                                             activation=activation, loss=loss)
        elif self.net == "cnn":
            self.policy_network = CNN(input_dim=self.num_features, num_steps=self.num_steps,
                                     output_dim=self.agent.NUM_ACTIONS, lr=self.lr,
                                     shared_network=shared_network,
                                     activation=activation, loss=loss)

        #모델 reuse하고 지정된 경로에 파일이 존재하면 불러오기
        if self.reuse_models and os.path.exists(self.policy_network_path):
            self.policy_network.load_model(model_path=self.policy_network_path)


    """ 에포크 초기화 함수 """
    def reset(self):
        self.sample = None
        self.training_data_idx = -1 #학습 데이터를 읽어가면서 1씩 증가
        #환경 초기화
        self.environment.reset()
        #에이전트 초기화
        self.agent.reset()
        #가시화 초기화
        self.visualizer.clear(xlim=[0, len(self.chart_data)])
        #메모리 초기화
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        #에포크 관련 정보 초기화
        self.loss = 0.0
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0


    """ 가치 신경망 및 정책 신경망 학습 """
    """ (1) 샘플 하나 생성 함수 """
    def build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist() #26 피쳐
            self.sample.extend(self.agent.get_states()) #2피쳐 추가: 에이전트 상태 튜플 iterable을 풀어서 각 원소를 리스트 추가
            return self.sample
        return None


    """ (2) 배치 학습 데이터 생성 함수 """
    @abc.abstractmethod #자식 클래스에서 구현할 메소드
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        pass


    """ (3) 신경망 학습 함수 """
    def update_networks(self, batch_size, delayed_reward, discount_factor):
        #배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch(batch_size, delayed_reward, discount_factor)
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                #가치 신경망 학습
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                #정책 신경망 학습
                loss += self.policy_network.train_on_batch(x, y_policy)
            return loss
        return None


    """ (4) 신경망 학습 요청 함수 """
    def fit(self, delayed_reward, discount_factor):
        #배치 학습 데이터 생성 및 신경망 갱신
        if self.batch_size > 0:
            _loss = self.update_networks(self.batch_size, delayed_reward, discount_factor)
            if _loss is not None:
                self.loss += abs(_loss)
                self.learning_cnt += 1
                self.memory_learning_idx.append(self.training_data_idx)
            self.batch_size = 0


    """ (5) 하나의 에포크 결과 가시화 함수 """
    def visualize(self, epoch_str, num_epoches, epsilon):
        #가시화 대상들은 환경의 일봉 수보다 (num_steps-1)만큼 부족하기 때문에
        #(num_steps-1)만큼 의미 없는 값을 첫 부분에 채워준다.
        self.memory_action = [Agent.ACTION_HOLD]*(self.num_steps-1)+self.memory_action
        self.memory_num_stocks = [0]*(self.num_steps-1)+self.memory_num_stocks

        if self.value_network is not None:
            self.memory_value = [np.array( [np.nan]*len(Agent.ACTIONS) )]*(self.num_steps-1)+self.memory_value
        if self.policy_network is not None:
            self.memory_policy = [np.array( [np.nan]*len(Agent.ACTIONS) )]*(self.num_steps-1)+self.memory_policy

        self.memory_pv = [self.agent.initial_balance] * (self.num_steps-1) + self.memory_pv

        #가시화 실행
        self.visualizer.plot(
            epoch_str=epoch_str, num_epoches=num_epoches, epsilon=epsilon,
            action_list=Agent.ACTIONS, actions=self.memory_action, num_stocks=self.memory_num_stocks,
            outvals_value=self.memory_value, outvals_policy=self.memory_policy, exps=self.memory_exp_idx,
            learning_idxes=self.memory_learning_idx, initial_balance=self.agent.initial_balance, pvs=self.memory_pv)

        self.visualizer.save(os.path.join(self.epoch_summary_dir, "epoch_summary_{}.png".format(epoch_str)))


    """ (6) 강화학습 실행 함수 """
    def run(self, num_epoches=100, balance=10000000, discount_factor=0.9, start_epsilon=0.5, learning=True):

        #start_epsilon:초기 탐험 비율
        #learning:학습 유무(True이면 신경망 학습, False이면 학습된 모델로 투자 시뮬레이션)

        info = "[{code}] RL:{rl} Net:{net} LR:{lr} DF:{discount_factor}"\
                "TU:[{min_trading_unit}, {max_trading_unit}]"\
                "DRT:{delayed_reward_threshold}".format(
            code=self.stock_code, rl=self.rl_method, net=self.net,
            lr=self.lr, discount_factor=discount_factor,
            min_trading_unit=self.agent.min_trading_unit,
            max_trading_unit=self.agent.max_trading_unit,
            delayed_reward_threshold=self.agent.delayed_reward_threshold)

        with self.lock:
            logging.info(info)

        #시작 시간
        time_start = time.time()

        "--------------------------가시화 준비-----------------------------"
        #차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(chart_data=self.environment.chart_data, title=info)
        #가시화 결과 저장할 폴더 준비, 가시화 결과는 output_path 경로 하위의 epoch_summary_{}폴더에 저장
        self.epoch_summary_dir = os.path.join(self.output_path, "epoch_summary_{}".format(self.stock_code))

        #하위폴더 없으면 만들기
        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)
        #하위폴더 있으면 폴더 내 파일 모두 삭제
        else:
            for f in os.listdir(self.epoch_summary_dir):
                os.remove(os.path.join(self.epoch_summary_dir, f))

        #에이전트 초기 자본금 설정
        self.agent.set_balance(balance)
        #학습에 대한 정보 초기화
        max_portfolio_value = 0 #수행한 에포크 중 가장 높은 pv
        epoch_win_cnt = 0 #수행 중 수익이 발생한 횟수

        "--------------------------학습 반복------------------------------"
        for epoch in range(num_epoches):
            time_start_epoch = time.time()

            #step 샘플을 만들기 위한 큐
            q_sample = collections.deque(maxlen=self.num_steps)
            #환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()
            #학습을 진행할수록 탐험 비율 감소: 초기 탐험률 * (1-학습 진행률)
            if learning:
                epsilon = start_epsilon * ( 1.0-(float(epoch) / (num_epoches-1)) )
                self.agent.reset_exploration() #exploration_base 새로 정하기
            else:
                epsilon = start_epsilon


            while True:
                #샘플 생성
                next_sample = self.build_sample()
                if next_sample is None: #마지막 데이터까지 다 읽은 경우
                    break
                #num_steps만큼 샘플 저장
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue

                #가치, 정책 신경망 예측
                pred_value = None
                pred_policy = None
                if self.value_network is not None:
                    pred_value = self.value_network.predict(list(q_sample))
                if self.policy_network is not None:
                    pred_policy = self.policy_network.predict(list(q_sample))

                #신경망 또는 탐험에 의한 행동 결정
                action, confidence, exploration = self.agent.decide_action(pred_value, pred_policy, epsilon)
                #결정한 행동을 수행하고 즉시 보상과 지연 보상 획득
                immediate_reward, delayed_reward = self.agent.act(action, confidence)

                #행동 및 행동에 대한 결과를 기억(학습에서 배치 학습 데이터용, 가시화 데이터용)
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(immediate_reward)
                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)

                #반복에 대한 정보 갱신
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0

                #지연 보상 발생된 경우 미니 배치 학습
                if learning and (delayed_reward != 0):
                    self.fit(delayed_reward, discount_factor)

            #에포크 종료 후 학습
            if learning:
                self.fit(self.agent.profitloss, discount_factor)


            #에포크 관련 정보 로그 기록
            num_epoches_digit = len(str(num_epoches)) #문자열 자릿수
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, "0") #문자열 자릿수만큼 오른쪽 정렬

            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch #에포크 소요 시간

            if self.learning_cnt > 0:
                self.loss = self.loss/self.learning_cnt #loss를 손실 총합에서 평균 손실로 갱신

            logging.info("[{}][Epoch {}/{}] Epsilon:{:.4f}"
                         "#Expl.:{}/{} #Buy:{} #Sell:{} #Hold:{}"
                         "#Stocks:{} PV:{:,.0f}"
                         "LC:{} Loss:{:.6f} ET:{:.4f}".format(
                self.stock_code, epoch_str, num_epoches, epsilon,
                self.exploration_cnt, self.itr_cnt,
                self.agent.num_buy, self.agent.num_sell,
                self.agent.num_hold, self.agent.num_stocks,
                self.agent.portfolio_value, self.learning_cnt,
                self.loss, elapsed_time_epoch))

            #에포크 관련 정보 가시화
            self.visualize(epoch_str, num_epoches, epsilon)
            #학습 관련 정보 갱신
            max_portfolio_value = max(max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        #종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start

        #학습 관련 정보 로그 기록
        with self.lock:
            logging.info("[{code}] Elapsed Time:{elapsed_time:.4f}"
                         "Max PV:{max_pv:,.0f} #Win:{cnt_win}".format(
                code=self.stock_code, elapsed_time=elapsed_time,
                max_pv=max_portfolio_value, cnt_win=epoch_win_cnt))


    """ (7) 신경망 모델 저장 함수 """
    def save_model(self):
        if self.value_network is not None and self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)


class DQNLearner(ReinforcementLearner):
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network() #가치 신경망 지정

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_value[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]))

        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        value_max_next = 0 #다음 상태에서 최대 q값
        reward_next = self.memory_reward[-1]

        #get_batch로 x(sample), 가치 타깃값(y_value) 얻기
        #x는 train_on_batch의 input으로 들어가서 output 가치 예측 값을 내고 y_value로 td_err 오차함수 구성

        for i, (sample, action, value, reward) in enumerate(memory):
            x[i] = sample
            y_value[i] = value
            r = ( (delayed_reward -reward) + (reward_next - reward) )*100
            y_value[i, action] = r + discount_factor*value_max_next #가치 타깃값 업데이트
            value_max_next = value.max()
            reward_next = reward #다음 행동 시점에서의 손익률을 next_reward변수에 저장
        return x, y_value, None

class PolicyGradientLearner(ReinforcementLearner):
    def __init__(self, *args, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_network_path = policy_network_path
        self.init_policy_network()

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_policy[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]))

        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), 0.5)
        reward_next = self.memory_reward[-1]

        for i, (sample, action, policy, reward) in enumerate(memory):
            x[i] = sample
            y_policy[i] = policy
            r = ( (delayed_reward -reward) + (reward_next - reward) )*100
            y_policy[i, action] = sigmoid(r)
            reward_next = reward
        return x, None, y_policy









































