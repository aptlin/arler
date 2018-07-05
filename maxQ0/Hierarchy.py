import numpy as np
import math
import random
from random import choice

SOUTH = 0
NORTH = 1
EAST = 2
WEST = 3
PICKUP = 4
DROPOFF = 5

GET = 6
PUT = 7
GOTO_R = 8
GOTO_G = 9
GOTO_Y = 10
GOTO_B = 11

ROOT = 12

R, G, Y, B = (0, 0), (0, 4), (4, 0), (4, 3)

DEFAULT_PRIMITIVE_ACTIONS = {SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF}
GOTO = {GOTO_R, GOTO_G, GOTO_Y, GOTO_B}

DEFAULT_HIERARCHY_STRUCTURE = {
    ROOT: [GET, PUT],
    GET: [PICKUP, GOTO_R, GOTO_G, GOTO_Y, GOTO_B],
    PUT: [DROPOFF, GOTO_R, GOTO_G, GOTO_Y, GOTO_B],
    GOTO_R: [SOUTH, NORTH, EAST, WEST],
    GOTO_G: [SOUTH, NORTH, EAST, WEST],
    GOTO_Y: [SOUTH, NORTH, EAST, WEST],
    GOTO_B: [SOUTH, NORTH, EAST, WEST]
}


class Skill:
    def __init__(
        self,
        identity,
        skillset
    ):
        self.id = identity
        self.skillset = skillset

    def isPrimitive(
        self
    ):
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
        if self.id == GOTO_R:
            return self.position() == R
        elif self.id == GOTO_G:
            return self.position() == G
        elif self.id == GOTO_Y:
            return self.position() == Y
        elif self.id == GOTO_B:
            return self.position() == B
        else:
            return False

    def isNotInPosition(self):
        if self.id == GOTO_R:
            return self.position() != R
        elif self.id == GOTO_G:
            return self.position() != G
        elif self.id == GOTO_Y:
            return self.position() != Y
        elif self.id == GOTO_B:
            return self.position() != B
        else:
            return False

    def subtasks(
        self
    ):
        if self.isPrimitive():
            return []
        else:
            return [
                Skill(subtaskId, self.skillset)
                for subtaskId in self.skillset[self.id]
            ]

    def next(self, state, distribution, explorationRate):
        if random.uniform(0, 1) < explorationRate:
            return choice(self.subtasks())
        else:
            assert len(distribution) == len(self.skillset[self.id])
            return Skill(self.skillset[self.id][np.argmax(distribution)],
                         self.skillset)


class SkillTree:
    def __init__(
        self,
        env,
        structure=DEFAULT_HIERARCHY_STRUCTURE,
        root=ROOT,
        primitives=DEFAULT_PRIMITIVE_ACTIONS
    ):
        self.domain = env
        self.root = Skill(root, self)
        self.structure = structure
        self.primitives = primitives
        self.size = len(self.structure) + len(primitives)

    def __getitem__(
        self,
        taskId
    ):
        return self.structure[taskId]
