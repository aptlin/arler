import math
import random
from random import choice
from arler.Utilities.operate import findRoot

import networkx as nx
import numpy as np


class Agenda:
    def __init__(self, priorities):
        self.priorities = priorities
        self.root = findRoot(self.priorities)

    @property
    def size(self):
        return len(self.priorities.nodes())

    def info(self, task):
        return self.priorities.nodes()[task]

    def subtasks(self, task):
        return list(self.priorities.successors(task))

    def next(self, task, distribution, explorationRate):
        if random.uniform(0, 1) < explorationRate:
            return choice(self.subtasks(task))
        else:
            assert len(distribution) == len(self.subtasks(task))
            return self.subtasks(task)[np.argmax(distribution)]


class Task:
    def __init__(self, name, identity=None, isPrimitive=False):
        self.name = name
        self.id = identity
        self.isPrimitive = isPrimitive

