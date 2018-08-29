import networkx as nx


class Trajectory:
    def __init__(self, path=nx.DiGraph()):
        self.path = path

    def addEdge(self, leftNode, rightNode, relevantVariables=set()):
        self.path.add_edge(leftNode, rightNode, relevantVariables=relevantVariables)

