#강화학습 주식투자 실행 모듈
#학습기 클래스를 이용해 강화학습을 수행하고 학습한 신경망들을 저장하는 메인 모듈

import os
import sys
import logging
import argparse
import json

import settings
import utils
import data_manager

""" 프로그램 인자 설정 """
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_code", nargs = "+") #nargs: 여러개의 인자를 리스트로 저장하는 옵션
    parser.add_argument("--ver", choices=["v1", "v2"], default="v1")
    parser.add_argument("--rl_method", choices=["dqn", "pg", "ac", "a2c", "a3c"])
    parser.add_argument("--net", choices=["dnn", "lstm", "cnn"], default="dnn")
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--discount_factor", type=float, default=0.9)
    parser.add_argument("--start_epsilon", type=float, default=0)
    parser.add_argument("--balance", type=int, default=10000000)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--delayed_reward_threshold", type=float, default=0.05)
    parser.add_argument("--backend", choices=["tensorflow", "plaidml"], default="tensorflow")
    parser.add_argument("--output_name", default=utils.get_time_str()) #출력 파일 저장 폴더 이름
    parser.add_argument("--value_network_name") #가치 신경망 모델 파일명
    parser.add_argument("--policy_network_name") #정책 신경망 모델 파일명
    parser.add_argument("--reuse_models", action="store_true") #신경망 모델 재사용 유무
    parser.add_argument("--learning", action="store_true") #강화학습 유무
    parser.add_argument("--start_date", default="20170101")
    parser.add_argument("--end_date", default="20170131")
    args = parser.parse_args()

""" 인자에 맞추어 강화학습 설정 """
#keras 백엔드 설정
if args.backend == "tensorflow":
    os.environ["KERAS_BACKEND"] = "tensorflow"
elif args.backend == "plaidml":
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

#출력 경로 설정
output_path = os.path.join(settings.BASE_DIR, "output_{}_{}_{}".format(args.output_name, args.rl_method, args.net))
if not os.path.isdir(output_path):
    os.makedirs(output_path)

#파라미터 기록
#json.dumps()는 딕셔너리에서 json 문자열로 변환
#json.loads()는 json 문자열에서 딕셔너리로 변환
with open(os.path.join(output_path, "params.json"), "w") as f:
    f.write(json.dumps(vars(args)))

#로그 기록 설정
file_handler = logging.FileHandler(filename=os.path.join(output_path, "{}.log".format(args.output_name)), encoding="utf-8") #파일 로그 출
stream_handler = logging.StreamHandler(sys.stdout) #콘솔 로그 출력
file_handler.setLevel(logging.DEBUG)
stream_handler.setLevel(logging.INFO)
logging.basicConfig(format="%(message)s", handlers=[file_handler, stream_handler], level=logging.DEBUG)

""" 에이전트, 학습기 클래스 임포트, 신경망 파일 경로 설정 """
#로그, keras 백엔드 설정을 먼저하고 RLTrader 모듈을 이후에 임포트해야함
from agent import Agent
from learners import DQNLearner, PolicyGradientLearner

#모델 경로 준비
value_network_path = ""
policy_network_path = ""

if args.value_network_name is not None:
    value_network_path = os.path.join(settings.BASE_DIR, "models_{}.h5".format(args.value_network_name))
else:
    value_network_path = os.path.join(output_path, "{}_{}_value_{}.h5".format(args.rl_method, args.net, args.output_name))
if args.policy_network_name is not None:
    policy_network_path = os.path.join(settings.BASE_DIR, "models_{}.h5".format(args.policy_network_name))
else:
    policy_network_path = os.path.join(output_path, "{}_{}_policy_{}.h5".format(args.rl_method, args.net, args.output_name))

""" 강화학습 준비와 실행 """
common_params = []
list_stock_code = []
list_chart_data = []
list_training_data = []
list_min_trading_unit = []
list_max_trading_unit = []

for stock_code in args.stock_code:
    #차트 데이터, 학습 데이터 준비
    # DataFrame = creon_api.creon_7400_주식차트조회(stock_code, args.start_data, args.end_data) #
    # DataFrame.to_csv(os.path.join(settings.BASE_DIR, "data/{}/{}.csv".format(args.ver, stock_code)), header=True) #

    chart_data, training_data = data_manager.load_data(
        os.path.join(settings.BASE_DIR, "data_{}_{}.csv".format(args.ver, stock_code)),
        args.start_date, args.end_date, ver=args.ver)

    #최소/최대 투자 단위 설정
    min_trading_unit = max(int(100000 / chart_data.iloc[-1]["close"]), 1) #10만원 안에 매수가능 수
    max_trading_unit = max(int(1000000 / chart_data.iloc[-1]["close"]), 1) #100만원 안에 매수가능 수

    #공통 파라미터 설정
    common_params = {"rl_method": args.rl_method, "delayed_reward_threshold":args.delayed_reward_threshold,
                     "net":args.net, "num_steps":args.num_steps, "lr":args.lr,
                     "output_path":output_path, "reuse_models":args.reuse_models}


    """ 강화학습 수행 """
    #강화학습 시작
    learner = None
    if args.rl_method != "a3c":
        common_params.update({"stock_code":stock_code, "chart_data":chart_data, "training_data":training_data,
                              "min_trading_unit":min_trading_unit, "max_trading_unit":max_trading_unit})

        if args.rl_method == "dqn":
            learner = DQNLearner(**{**common_params, "value_network_path":value_network_path})
        elif args.rl_method == "pg":
            learner = PolicyGradientLearner(**{**common_params, "policy_network_path":policy_network_path})
        # elif args.rl_method == "ac":
        #     learner = ActorCriticLearner(**{**common_params, "value_network_path":value_network_path, "policy_network_path":policy_network_path})
        # elif args.rl_method == "a2c":
        #     learner = A2CLearner(**{**common_params, "value_network_path":value_network_path, "policy_network_path":policy_network_path})

        if learner is not None:
            learner.run(balance=args.balance, num_epoches=args.num_epoches,
                        discount_factor=args.discount_factor,
                        start_epsilon=args.start_epsilon,
                        learning=args.learning)
            learner.save_model()

    else: #a3c를 위한
        list_stock_code.append(stock_code)
        list_chart_data.append(chart_data)
        list_training_data.append(training_data)
        list_min_trading_unit.append(min_trading_unit)
        list_max_trading_unit.append(max_trading_unit)

# if args.rl_method == "a3c":
#     learner = A3CLearner(**{**common_params,
#                             "list_stock_code":list_stock_code,
#                             "list_chart_data":list_chart_data,
#                             "list_training_data":list_training_data,
#                             "list_min_trading_unit":list_min_trading_unit,
#                             "list_max_trading_unit":list_max_trading_unit,
#                             "value_network_path":value_network_path,
#                             "policy_network_path":policy_network_path})
#
#     learner.run(balance=args.balance, num_epoches=args.num_epoches,
#                 discount_factor=args.discount_factor,
#                 start_epsilon=args.start_epsilon,
#                 learning=args.learning)
#     learner.save_model()

