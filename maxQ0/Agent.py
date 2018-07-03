import numpy as np
import random

SOUTH = 0
NORTH = 1
EAST = 2
WEST = 3
PICKUP = 4
DROPOFF = 5

GET = 6
PUT = 7
GOTO_R = 8
GOTO_G = 9
GOTO_Y = 10
GOTO_B = 11

ROOT = 12

R, G, Y, B = (0, 0), (0, 4), (4, 0), (4, 3)

DEFAULT_PRIMITIVE_ACTIONS = {SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF}
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_LEARNING_RATE_DECREMENT = 0.001

DEFAULT_HIERARCHY_STRUCTURE = {
    ROOT: [GET, PUT],
    GET: [PICKUP, GOTO_R, GOTO_G, GOTO_Y, GOTO_B],
    PUT: [DROPOFF, GOTO_R, GOTO_G, GOTO_Y, GOTO_B],
    GOTO_R: [SOUTH, NORTH, EAST, WEST],
    GOTO_G: [SOUTH, NORTH, EAST, WEST],
    GOTO_Y: [SOUTH, NORTH, EAST, WEST],
    GOTO_B: [SOUTH, NORTH, EAST, WEST]
}

REWARD_FLOOR = -1000000


class Task:
    def __init__(
        self,
        identity
    ):
        self.id = identity

    def isPrimitive(
        self,
        primitives=DEFAULT_PRIMITIVE_ACTIONS
    ):
        return self.id in primitives


class TaskHierarchy:
    def __init__(
        self,
        graph=DEFAULT_HIERARCHY_STRUCTURE,
        primitives=DEFAULT_PRIMITIVE_ACTIONS
    ):
        self.structure = graph
        self.size = len(graph) + len(primitives)

    def __getitem__(
        self,
        taskId
    ):
        return self.structure[taskId]


class Taxi:
    def __init__(
        self,
        env,
        learningRate,
        discountFactor,
        explorationRate,
        hierarchy=None,
        primitives=DEFAULT_PRIMITIVE_ACTIONS
    ):
        self.env = env

        self.score = 0
        self.epochs = 0

        self.done = False

        self.loaded = False

        self.primitives = primitives

        if hierarchy is not None:
            self.hierarchy = hierarchy
        else:
            self.hierarchy = TaskHierarchy(primitives=primitives)

        self.discountFactor = discountFactor

        self.learningRate = learningRate

        self.explorationRate = explorationRate

        eventSpaceDim = env.observation_space.n

        self.cTable = np.zeros(
            (self.hierarchy.size, eventSpaceDim, self.hierarchy.size))
        self.vTable = np.zeros((self.hierarchy.size, eventSpaceDim))

    def trackProgress(
        self,
        reward
    ):
        self.score += reward * (self.discountFactor ** self.epochs)
        self.epochs += 1

    def run(
        self,
        action
    ):
        if action not in self.primitives:
            raise ValueError("Not a primitive action, aborting.")
        _, reward, self.done, _ = self.env.step(action)
        if action == PICKUP and reward == 0:
            self.loaded = True
        return reward

    def alphaUpdate(
        self,
        value,
        alpha,
        reward
    ):
        return value + alpha * (reward - value)

    def decreaseLearningRateBy(self, decrement):
        if (self.learningRate >= decrement):
            self.learningRate -= decrement
        else:
            self.learningRate = 0

    def updateRewardsWith(
        self,
        reward,
        task,
        state
    ):
        v = self.vTable[task.id][state]
        self.vTable[task.id][state] = self.alphaUpdate(
            v, self.learningRate, reward)

    def updateCompositeRewardsWith(
        self,
        reward,
        previousTask,
        state,
        currentTask
    ):
        c = self.cTable[previousTask.id][state][currentTask.id]
        self.cTable[previousTask.id][state][currentTask.id] = self.alphaUpdate(
            c, self.learningRate, reward)

    def computeReward(
        self,
        task,
        state
    ):
        if task.isPrimitive():
            return self.vTable[task.id][state]
        else:
            return self.computeCompositeReward(task, state, self.pickNextTask(task, state))

    def computeCompositeReward(
        self,
        task,
        state,
        nextTask
    ):
        return self.computeReward(nextTask, state) + self.cTable[task.id][state][nextTask.id]

    def pickNextTask(
        self,
        task,
        state
    ):
        numberOfSubtasks = len(self.hierarchy[task.id])
        memory = np.full((numberOfSubtasks,), REWARD_FLOOR)
        for subtaskId in range(numberOfSubtasks):
            subtask = Task(self.hierarchy[task.id][subtaskId])
            if self.isBeneficial(subtask):
                memory[subtaskId] = self.computeCompositeReward(
                    task, state, subtask)
        optimalSubtaskId = np.argmax(memory)
        return Task(self.hierarchy[task.id][optimalSubtaskId])

    def explore(
        self,
        task,
        state
    ):
        if random.uniform(0, 1) < self.explorationRate:
            return Task(self.env.action_space.sample())
        else:
            return self.pickNextTask(task, state)

    def position(self):
        data = list(self.env.decode(self.env.s))
        return data[0], data[1]

    def isBeneficial(
        self,
        task
    ):
        if task.id == ROOT:
            return not self.done
        elif task.id == GET:
            return not self.loaded
        elif task.id == PUT:
            return self.loaded
        elif task.id == GOTO_R:
            return self.position() != R
        elif task.id == GOTO_G:
            return self.position() != G
        elif task.id == GOTO_Y:
            return self.position() != Y
        elif task.id == GOTO_B:
            return self.position() != B
        else:
            return True

    def isTerminal(
        self,
        task
    ):
        if self.done:
            return True
        elif task.id == ROOT:
            return self.done
        elif task.id == GET:
            return self.loaded
        elif task.id == PUT:
            return not self.loaded
        elif task.id == GOTO_R:
            return self.position() == R
        elif task.id == GOTO_G:
            return self.position() == G
        elif task.id == GOTO_Y:
            return self.position() == Y
        elif task.id == GOTO_B:
            return self.position() == B
        else:
            return True

    def reset(self, env):
        self.env = env
        self.score = 0
        self.epochs = 0
        self.done = False
        self.loaded = False

    def maxQ0(
        self,
        task,
        state,
        learningRateDecrement=DEFAULT_LEARNING_RATE_DECREMENT
    ):
        self.decreaseLearningRateBy(learningRateDecrement)
        if task.isPrimitive():
            reward = self.run(task.id)
            self.updateRewardsWith(reward, task, state)
            self.trackProgress(reward)
            return 1

        else:
            effort = 0
            while (not self.isTerminal(task)):
                currentTask = self.explore(task, state)
                taskEffort = self.maxQ0(currentTask, state)
                currentState = self.env.s
                discountFactor = self.discountFactor ** taskEffort
                discountedReward = discountFactor * \
                    self.computeReward(task, currentState)
                self.updateCompositeRewardsWith(
                    discountedReward, task, state, currentTask)
                effort += taskEffort
                state = currentState
            return effort
