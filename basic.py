import time
import random
import matplotlib.pyplot as plt
import math

import numpy as np
from maxQ0 import Agent

import gym

ENVIRONMENT = 'Taxi-v2'

LEARNING_RATE = 0.1
LEARNING_RATE_DECREMENT = 0.01
DISCOUNT_FACTOR = 0.999
EXPLORATION_RATE = 0.1

EPISODES = 5000

ROOT = 12

env = gym.make(ENVIRONMENT).env

epochsSeries = []
scoresSeries = []

taxi = Agent.Taxi(env, LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_RATE)
for _ in range(EPISODES):
    env.reset()
    taxi.reset(env)
    epochs = taxi.maxQ0(taxi.skillset.root, env.s)
    epochsSeries.append(epochs)
    scoresSeries.append(taxi.score)
print("Results after {} episodes:".format(EPISODES))
print("\tAverage timesteps per episode: {}".format(sum(epochsSeries) / EPISODES))
print("\tAverage score per episode: {}".format(sum(scoresSeries) / EPISODES))

plt.plot(range(EPISODES), scoresSeries)
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.show()