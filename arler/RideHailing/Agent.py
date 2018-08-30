import numpy as np
import math


class Taxi:
    def __init__(self, env, agenda, learningRate, discountFactor, explorationRate):
        self.domain = env
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.explorationRate = explorationRate
        self.agenda = agenda

        self.score = 0
        self.done = False
        numberOfTasks = self.agenda.size
        self.completionCost = np.zeros((numberOfTasks, self.domain.observation_space.n))
        self.discountedCompletionCost = np.zeros(
            (numberOfTasks, self.domain.observation_space.n, numberOfTasks)
        )

    @property
    def loaded(self):
        # The third argument in the decoded state is 4 iff the passenger is inside
        # https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
        return list(self.domain.decode(self.domain.s))[2] == 4

    def computeReward(self, task, state):
        if task.isPrimitive:
            return self.completionCost[task.id][state]
        else:
            return max(self.taskCompositeRewards(task, state))

    def computeCompositeReward(self, task, state, nextTask):
        return (
            self.computeReward(nextTask, state)
            + self.discountedCompletionCost[task.id][state][nextTask.id]
        )

    def alphaUpdate(self, value, alpha, reward):
        return value + alpha * (reward - value)

    def updateCompletionCostsWith(self, reward, task, state):
        self.completionCost[task.id][state] = self.alphaUpdate(
            self.completionCost[task.id][state], self.learningRate, reward
        )

    def updateDiscountedCompletionCostsWith(self, reward, task, state, nextTask):
        self.discountedCompletionCost[task.id][state][nextTask.id] = self.alphaUpdate(
            self.discountedCompletionCost[task.id][state][nextTask.id],
            self.learningRate,
            reward,
        )

    def reset(self):
        self.domain.reset()
        self.score = 0
        self.done = False

    def step(self, task):
        if not task.isPrimitive:
            raise ValueError("Not a primitive action, aborting.")
        _, reward, self.done, _ = self.domain.step(task.id)
        self.score += reward
        return reward

    def isTerminal(self, task):
        if self.done or task.isPrimitive:
            return True
        elif task.id == self.agenda.root.id:
            return self.done
        elif task.isGetting():
            return self.loaded
        elif task.isDelivering():
            return not self.loaded
        else:
            return self.agenda.hasArrived(task)

    def isBeneficial(self, task):
        if task.id == self.agenda.root.id:
            return not self.done
        elif task.isPrimitive:
            return True
        elif task.isGetting():
            return not self.loaded
        elif task.isDelivering():
            return self.loaded
        else:
            return not self.agenda.hasArrived(task)

    def taskCompositeRewards(self, task, state):
        subtasks = self.agenda.subtasks(task)
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
                nextTask = self.agenda.next(
                    task, self.taskCompositeRewards(task, state), self.explorationRate
                )
                nextTaskEffort = self.maxQ0(nextTask, state)
                nextState = self.domain.s

                discountFactor = self.discountFactor ** nextTaskEffort
                discountedReward = self.computeReward(task, nextState) * discountFactor

                self.updateDiscountedCompletionCostsWith(
                    discountedReward, task, state, nextTask
                )
                effort += nextTaskEffort
                state = nextState
            return effort

    def run(self):
        return self.maxQ0(self.agenda.root, self.domain.s)
