from arler.Learning.Agent import Learner

import numpy as np
import math


class Taxi(Learner):
    def __init__(self, env, agenda, learningRate, discountFactor, explorationRate):
        self.domain = env
        self.agenda = agenda

        super().__init__(
            self.agenda.size,
            self.domain.observation_space.n,
            learningRate,
            discountFactor,
            explorationRate,
        )

    @property
    def loaded(self):
        # The third argument in the decoded state is 4 iff the passenger is inside
        # https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
        return list(self.domain.decode(self.domain.s))[2] == 4

    def reset(self):
        self.domain.reset()
        self.score = 0
        self.done = False

    def step(self, task):
        if not task.isPrimitive:
            raise ValueError("Not a primitive action, aborting.")
        _, reward, self.done, _ = self.domain.step(task.identity)
        self.score += reward
        return reward

    def nextTask(self, task, state, expectedRewards, explorationRate):
        return self.agenda.next(
            task, self.taskCompositeRewards(task, state), self.explorationRate
        )

    def subtasks(self, task):
        return self.agenda.subtasks(task)

    def isTerminal(self, task):
        if self.done or task.isPrimitive:
            return True
        elif task.identity == self.agenda.root.identity:
            return self.done
        elif task.isGetting():
            return self.loaded
        elif task.isDelivering():
            return not self.loaded
        else:
            return self.agenda.hasArrived(task)

    def isBeneficial(self, task):
        if task.identity == self.agenda.root.identity:
            return not self.done
        elif task.isPrimitive:
            return True
        elif task.isGetting():
            return not self.loaded
        elif task.isDelivering():
            return self.loaded
        else:
            return not self.agenda.hasArrived(task)

    def getState(self):
        return self.domain.s

    def run(self):
        return self.maxQ0(self.agenda.root, self.domain.s)
