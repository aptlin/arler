import numpy as np
import math
from .Hierarchy import SkillTree


class Taxi:
    def __init__(
        self,
        env,
        learningRate,
        discountFactor,
        explorationRate,
        hierarchy=None,
    ):
        self.domain = env
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.explorationRate = explorationRate
        self.skillset = self.inferSkillsFrom(hierarchy)

        self.score = 0
        self.done = False

        self.completionCost = np.zeros(
            (
                self.skillset.size,
                self.domain.observation_space.n
            )
        )
        self.discountedCompletionCost = np.zeros(
            (
                self.skillset.size,
                self.domain.observation_space.n,
                self.skillset.size
            )
        )

    @property
    def loaded(self):
        return list(self.domain.decode(self.domain.s))[2] == 4

    def inferSkillsFrom(
        self,
        hierarchy
    ):
        if hierarchy is None:
            return SkillTree(self.domain)
        else:
            return hierarchy

    def computeReward(
        self,
        task,
        state
    ):
        if task.isPrimitive():
            return self.completionCost[task.id][state]
        else:
            return max(self.taskCompositeRewards(task, state))

    def computeCompositeReward(
        self,
        task,
        state,
        nextTask
    ):
        return self.computeReward(nextTask, state) \
            + self.discountedCompletionCost[task.id][state][nextTask.id]

    def alphaUpdate(
        self,
        value,
        alpha,
        reward
    ):
        return value + alpha * (reward - value)

    def updateCompletionCostsWith(
        self,
        reward,
        task,
        state
    ):
        self.completionCost[task.id][state] = self.alphaUpdate(
            self.completionCost[task.id][state],
            self.learningRate,
            reward
        )

    def updateDiscountedCompletionCostsWith(
        self,
        reward,
        task,
        state,
        nextTask
    ):
        self.discountedCompletionCost[task.id][state][nextTask.id] \
            = self.alphaUpdate(
            self.discountedCompletionCost[task.id][state][nextTask.id],
            self.learningRate,
            reward
        )

    def reset(
        self
    ):
        self.domain.reset()
        self.score = 0
        self.done = False

    def step(
        self,
        task
    ):
        if not task.isPrimitive():
            raise ValueError("Not a primitive action, aborting.")
        _, reward, self.done, _ = self.domain.step(task.id)
        self.score += reward
        return reward

    def isTerminal(
        self,
        task
    ):
        if self.done or task.isPrimitive():
            return True
        elif task.isRoot():
            return self.done
        elif task.isGetting():
            return self.loaded
        elif task.isDelivering():
            return not self.loaded
        else:
            return task.isInPosition()

    def isBeneficial(
        self,
        task
    ):
        if task.isRoot():
            return not self.done
        elif task.isPrimitive():
            return True
        elif task.isGetting():
            return not self.loaded
        elif task.isDelivering():
            return self.loaded
        else:
            return task.isNotInPosition()

    def taskCompositeRewards(
        self,
        task,
        state
    ):
        subtasks = task.subtasks()
        series = np.full((len(subtasks), 1), -math.inf)
        for idx in range(len(subtasks)):
            if self.isBeneficial(subtasks[idx]):
                series[idx] = self.computeCompositeReward(
                    task, state, subtasks[idx])
        return series

    def maxQ0(
        self,
        task,
        state
    ):
        if task.isPrimitive():
            reward = self.step(task)
            self.updateCompletionCostsWith(reward, task, state)
            return 1
        else:
            effort = 0
            while not self.isTerminal(task):
                nextTask = task.next(
                    state,
                    self.taskCompositeRewards(task, state),
                    self.explorationRate
                )
                nextTaskEffort = self.maxQ0(nextTask, state)
                nextState = self.domain.s

                discountFactor = self.discountFactor ** nextTaskEffort
                discountedReward = self.computeReward(
                    task,
                    nextState
                ) * discountFactor

                self.updateDiscountedCompletionCostsWith(
                    discountedReward,
                    task,
                    state,
                    nextTask
                )
                effort += nextTaskEffort
                state = nextState
            return effort

    def run(self):
        return self.maxQ0(self.skillset.root, self.domain.s)
