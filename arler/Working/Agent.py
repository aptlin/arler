import networkx as nx
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
        self.id = identity
        self.terminationCondition = set()
        self.benefitCondition = set()

    @property
    def isPrimitive(self):
        return isinstance(self, PrimitiveAction)


class PrimitiveAction(Action):
    def __init__(self, name, identity=None, model=ActionModel()):
        super().__init__(name, identity)
        self.model = model


class CompositeAction(Action):
    def __init__(self, name, identity=None, models=ActionModel()):
        super().__init__(name, identity)
        self.model = ActionModel.mergeEdges(models)
