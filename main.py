import numpy as np

from maxQ0 import Agent
from maxQ0 import Manager

import gym

ENVIRONMENT = 'Taxi-v2'

LEARNING_RATE = 0.1
LEARNING_RATE_DECREMENT = 0.01
DISCOUNT_FACTOR = 0.999
EXPLORATION_RATE = 0.1

EPISODES = 5000

env = gym.make(ENVIRONMENT).env

taxi = Agent.Taxi(env, LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_RATE)
manager = Manager.TaxiCallCentre(ENVIRONMENT, taxi)

manager.run(EPISODES)
