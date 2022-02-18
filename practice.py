import data_manager
import os
import settings
import pandas as pd
import environment
import agent

path = os.path.join(settings.BASE_DIR, "data_{}_{}.csv".format("v1", "005930"))

chart_data, training_data = data_manager.load_data(
    path, "20170101", "20170131", ver="v1")

env = environment.Environment(chart_data=chart_data)
env.observe()
observation = env.observation

agent = agent.Agent(environment=env)
agent.set_balance(1000000)


print(agent.environment.get_price())
print(agent.num_stocks)
print(agent.portfolio_value)
# print(agent.get_states())
# print(agent.num_stocks/agent.portfolio_value/agent.get_states())
# print(env.get_price())
# # print(observation[4])

x = int(4)
print(min(x))