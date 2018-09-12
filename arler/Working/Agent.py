import networkx as nx
from arler.Thinking.Agent import Condition
from arler.Utilities.operate import mergeLists


class ActionModel:
    def __init__(self, network=nx.DiGraph()):
        self.network = network

    def visualise(self):
        pass

    @staticmethod
    def mergeEdges(*args):
        model = nx.DiGraph()
        edges = map(lambda actionModel: list(actionModel.network.edges()), args)
        edges = filter(None, edges)
        model.add_edges_from(mergeLists(edges))
        return model


class Action:
    def __init__(self, name, identity=None):
        self.name = name
        self.identity = identity
        self.terminationCondition = Condition()
        self.benefitCondition = Condition()

    @property
    def isPrimitive(self):
        return isinstance(self, PrimitiveAction)

    @property
    def isTerminal(self):
        return bool(self.terminationCondition)

    @property
    def isBeneficial(self):
        return bool(self.benefitCondition)

    def setTerminationCondition(self, variables=set()):
        self.terminationCondition = Condition(variables)

    def setBenefitCondition(self, variables=set()):
        self.benefitCondition = Condition(variables)


class PrimitiveAction(Action):
    def __init__(self, name, identity=None, model=ActionModel()):
        super().__init__(name, identity)
        self.model = model


class CompositeAction(Action):
    def __init__(self, name, identity=None, models=ActionModel()):
        super().__init__(name, identity)
        self.model = ActionModel.mergeEdges(models)
