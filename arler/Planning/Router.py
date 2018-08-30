import networkx as nx
from arler.Planning.Environment import Variable
from arler.Utilities.build import buildScaffold


class Trajectory:
    def __init__(self, blueprint):
        self.__scaffold__ = buildScaffold(blueprint, Variable)

    @property
    def size(self):
        return len(self.__scaffold__) - 2

    def getSharedContextBetween(self, parent, child):
        return self.__scaffold__[parent][child]

    def isRelevanceIdentical(self, *args)
        pass