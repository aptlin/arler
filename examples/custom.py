import gym
import os
from arler.Taxi import Agent, Manager
from arler.Taxi.settings import Configuration
from arler.Taxi.Routing import Instructions

ENVIRONMENT = "Taxi-v2"

LEARNING_RATE = 0.1
LEARNING_RATE_DECREMENT = 0.01
DISCOUNT_FACTOR = 0.999
EXPLORATION_RATE = 0.1

EPISODES = 5000

config = Configuration()
env = gym.make(ENVIRONMENT).env
instructions = Instructions(env, config.customHierarchyBlueprint)

taxi = Agent.Taxi(env, instructions, LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_RATE)
manager = Manager.TaxiCallCentre(ENVIRONMENT, taxi)

exampleDir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
modelsDir = os.path.join(exampleDir, "assets/models")
imagesDir = os.path.join(exampleDir, "assets/images")
manager.run(modelsDir, imagesDir, EPISODES)
