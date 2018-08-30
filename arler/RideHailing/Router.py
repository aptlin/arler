from arler.Planning.Scheduler import Agenda
from arler.Working.Actor import Action
from arler.RideHailing.Director import Configuration
from arler.Utilities.build import buildPriorities


class TaxiOrder(Action):
    def isGetting(self):
        return self.name == "GET"

    def isDelivering(self):
        return self.name == "PUT"


class Instructions(Agenda):
    def __init__(self, env, hierarchy):
        self.domain = env
        config = Configuration()
        self.corners = config.constants["corners"]
        super().__init__(buildPriorities(hierarchy, TaxiOrder))

    def position(self):
        data = list(self.domain.decode(self.domain.s))
        return data[:2]

    def hasArrived(self, task):
        return task.name in self.corners and self.position() == self.corners[task.name]

