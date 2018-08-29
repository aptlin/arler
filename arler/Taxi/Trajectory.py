import networkx as nx
from arler.Planner.Blueprint import Trajectory


class TaxiRoute(Trajectory):
    def __init__(self):
        route = nx.DiGraph()
        Trajectory.__init__(self)

