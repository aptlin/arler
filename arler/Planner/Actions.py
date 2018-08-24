from arler.Planner.Environment import Variable


class Action:
    def __init__(self, actionIdentity, parents=dict()):
        self.id = actionIdentity
        self.parents = parents

