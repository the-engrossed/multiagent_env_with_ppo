import numpy as np
from multi_agent_env import WumpusWorldEnvironment

env = WumpusWorldEnvironment(environment_type='deterministic')

n_observation_spaces = env.observation_space.n
n_actions = env.action_space.n

#Initialise a Q-table
Q_table = np.zeros((n_observation_spaces,n_actions))
print(Q_table)

n_episodes = 1000
max_iterations = 100
p_exploration = 1
exp_dec_decay = 0.001
min_exploration_proba = 0.01
gamma = 0.99
lr = 0.1

total_rewards_episode = list()

