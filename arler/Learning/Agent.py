from abc import ABC, abstractmethod
import numpy as np
import math


class Learner(ABC):
    def __init__(self, nTasks, nStates, learningRate, discountFactor, explorationRate):
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.explorationRate = explorationRate

        self.score = 0
        self.done = False
        self.completionCost = np.zeros((nTasks, nStates))
        self.discountedCompletionCost = np.zeros((nTasks, nStates, nTasks))

    @abstractmethod
    def step(self, task):
        pass

    @abstractmethod
    def nextTask(self, task, state, expectedRewards, explorationRate):
        pass

    @abstractmethod
    def subtasks(self, task):
        pass

    @abstractmethod
    def isTerminal(self, task):
        pass

    @abstractmethod
    def isBeneficial(self, task):
        pass

    @abstractmethod
    def getState(self):
        pass

    def computeReward(self, task, state):
        if task.isPrimitive:
            return self.completionCost[task.identity][state]
        else:
            return max(self.taskCompositeRewards(task, state))

    def computeCompositeReward(self, task, state, nextTask):
        return (
            self.computeReward(nextTask, state)
            + self.discountedCompletionCost[task.identity][state][nextTask.identity]
        )

    def alphaUpdate(self, value, alpha, reward):
        return value + alpha * (reward - value)

    def updateCompletionCostsWith(self, reward, task, state):
        self.completionCost[task.identity][state] = self.alphaUpdate(
            self.completionCost[task.identity][state], self.learningRate, reward
        )

    def updateDiscountedCompletionCostsWith(self, reward, task, state, nextTask):
        self.discountedCompletionCost[task.identity][state][
            nextTask.identity
        ] = self.alphaUpdate(
            self.discountedCompletionCost[task.identity][state][nextTask.identity],
            self.learningRate,
            reward,
        )

    def taskCompositeRewards(self, task, state):
        subtasks = self.subtasks(task)
        series = np.full((len(subtasks), 1), -math.inf)
        for idx in range(len(subtasks)):
            if self.isBeneficial(subtasks[idx]):
                series[idx] = self.computeCompositeReward(task, state, subtasks[idx])
        return series

    def maxQ0(self, task, state):
        if task.isPrimitive:
            reward = self.step(task)
            self.updateCompletionCostsWith(reward, task, state)
            return 1
        else:
            effort = 0
            while not self.isTerminal(task):
                nextTask = self.nextTask(
                    task,
                    state,
                    self.taskCompositeRewards(task, state),
                    self.explorationRate,
                )
                nextTaskEffort = self.maxQ0(nextTask, state)
                nextState = self.getState()

                discountFactor = self.discountFactor ** nextTaskEffort
                discountedReward = self.computeReward(task, nextState) * discountFactor

                self.updateDiscountedCompletionCostsWith(
                    discountedReward, task, state, nextTask
                )
                effort += nextTaskEffort
                state = nextState
            return effort

