import math
import random
from random import choice
from arler.Utilities.operate import findRoot

import networkx as nx
import numpy as np


class Agenda:
    def __init__(self, priorities):
        self.__priorities__ = priorities
        self.root = findRoot(self.__priorities__)

    @property
    def size(self):
        return len(self.__priorities__.nodes())

    def info(self, task):
        return self.__priorities__.nodes()[task]

    def subtasks(self, task):
        return list(self.__priorities__.successors(task))

    def addTo(self, parent, *args):
        edges = [(parent, child) for child in args]
        self.__priorities__.add_edges_from(edges)

    def next(self, task, distribution, explorationRate):
        if random.uniform(0, 1) < explorationRate:
            return choice(self.subtasks(task))
        else:
            assert len(distribution) == len(self.subtasks(task))
            return self.subtasks(task)[np.argmax(distribution)]
