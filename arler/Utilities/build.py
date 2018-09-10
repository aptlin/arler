from collections.abc import Mapping
from collections import defaultdict
import networkx as nx


def __saveNamedObject__(ObjectClass, memory, objectName):
    if objectName not in memory:
        memory[objectName] = ObjectClass(objectName)


def buildPriorities(blueprint, PrimitiveTask, CompositeTask):
    if len(blueprint) != 1:
        raise ValueError("The hierarchy given is not a rooted tree.")

    hierarchy = nx.DiGraph()
    flattenedBlueprint = list(blueprint.items())
    taskMemory = dict()
    primitiveIds = set()
    while flattenedBlueprint:
        parent, childLayer = flattenedBlueprint.pop()
        __saveNamedObject__(CompositeTask, taskMemory, parent)
        for child, grandchildLayer in childLayer.items():
            if isinstance(grandchildLayer, Mapping):
                __saveNamedObject__(CompositeTask, taskMemory, child)
                flattenedBlueprint.append((child, grandchildLayer))
            else:
                __saveNamedObject__(PrimitiveTask, taskMemory, child)
                taskMemory[child].id = grandchildLayer
                primitiveIds.add(grandchildLayer)
            hierarchy.add_edge(taskMemory[parent], taskMemory[child])
    availableIds = set(range(len(hierarchy))) - primitiveIds
    for node in hierarchy.nodes():
        if isinstance(node, CompositeTask):
            node.id = availableIds.pop()
    return hierarchy


def buildTrajectoryScaffold(blueprint, primitiveAction):
    taskMemory = dict()
    scaffold = nx.DiGraph()
    for parent, childLayer in blueprint.items():
        __saveNamedObject__(primitiveAction, taskMemory, parent)
        for child, childAttrs in childLayer.items():
            __saveNamedObject__(primitiveAction, taskMemory, child)
            scaffold.add_edge(taskMemory[parent], taskMemory[child], **childAttrs)

    return scaffold
