import networkx as nx
from arler.Utilities.operate import mergeLists


class ActionModel:
    def __init__(self, network):
        self.network = network

    def visualise(self):
        pass

    @staticmethod
    def merge(*args):
        model = nx.DiGraph()
        edges = map(lambda actionModel: actionModel.edges(), args)
        model.add_edges_from(mergeLists(edges))
        return model


class Action:
    def __init__(self, name, identity=None):
        self.name = name
        self.id = identity


class PrimitiveAction(Action):
    def __init__(self, name, identity, model):
        super().__init__(name, identity)
        self.model = model


class CompositeAction(Action):
    def __init__(self, name, identity, models):
        super().__init__(name, identity)
        self.model = ActionModel.merge(models)
