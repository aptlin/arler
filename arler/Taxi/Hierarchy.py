import numpy as np
import math
import random
from random import choice

# Hard-coded ids of primitive actions

SOUTH = 0
NORTH = 1
EAST = 2
WEST = 3

DIRECTIONS = [SOUTH, NORTH, EAST, WEST]

PICKUP = 4
DROPOFF = 5

# Hard-coded ids of complex actions

GET = 6
PUT = 7

GOTO_R = 8
GOTO_G = 9
GOTO_Y = 10
GOTO_B = 11

ROOT = 12

# Hard-coded ids of corners

COLOR_R = (0, 0)
COLOR_G = (0, 4)
COLOR_Y = (4, 0)
COLOR_B = (4, 3)

# Hard-coded collections

CORNERS = [COLOR_R, COLOR_G, COLOR_Y, COLOR_B]
GOTO_ACTIONS = [GOTO_R, GOTO_G, GOTO_Y, GOTO_B]

DEFAULT_PRIMITIVE_ACTIONS = {SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF}

DO_MOVE = dict(zip(GOTO_ACTIONS, CORNERS))

DEFAULT_HIERARCHY_STRUCTURE = {
    ROOT: [GET, PUT],
    GET: [PICKUP, *GOTO_ACTIONS],
    PUT: [DROPOFF, *GOTO_ACTIONS],
    **dict(zip(GOTO_ACTIONS, DIRECTIONS * 4)),
}


class Skill:
    def __init__(self, identity, skillset):
        self.id = identity
        self.skillset = skillset

    def isPrimitive(self):
        return self.id in self.skillset.primitives

    def isRoot(self):
        return self.id == self.skillset.root.id

    def isGetting(self):
        return self.id == GET

    def isDelivering(self):
        return self.id == PUT

    def position(self):
        env = self.skillset.domain
        data = list(env.decode(env.s))
        return data[0], data[1]

    def isInPosition(self):
        return self.id in DO_MOVE and self.position() == DO_MOVE[self.id]

    def isNotInPosition(self):
        return self.id in DO_MOVE and self.position() != DO_MOVE[self.id]

    def subtasks(self):
        if self.isPrimitive():
            return []
        else:
            return [
                Skill(subtaskId, self.skillset) for subtaskId in self.skillset[self.id]
            ]

    def next(self, state, distribution, explorationRate):
        if random.uniform(0, 1) < explorationRate:
            return choice(self.subtasks())
        else:
            assert len(distribution) == len(self.skillset[self.id])
            return Skill(self.skillset[self.id][np.argmax(distribution)], self.skillset)


class SkillTree:
    def __init__(
        self,
        env,
        structure=DEFAULT_HIERARCHY_STRUCTURE,
        root=ROOT,
        primitives=DEFAULT_PRIMITIVE_ACTIONS,
    ):
        self.domain = env
        self.root = Skill(root, self)
        self.structure = structure
        self.primitives = primitives

    @property
    def size(self):
        return len(self.structure) + len(self.primitives)

    def __getitem__(self, taskId):
        return self.structure[taskId]
