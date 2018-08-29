from collections.abc import Mapping
import networkx as nx


def __saveTask(TaskClass, memory, taskName):
    if taskName not in memory:
        memory[taskName] = TaskClass(taskName)


def buildPriorities(blueprint, Task):
    if len(blueprint) != 1:
        raise ValueError("The hierarchy given is not a rooted tree.")

    hierarchy = nx.DiGraph()
    flattenedBlueprint = list(blueprint.items())
    taskMemory = dict()
    primitiveIds = set()
    while flattenedBlueprint:
        parent, childLayer = flattenedBlueprint.pop()
        __saveTask(Task, taskMemory, parent)
        for child, grandchildLayer in childLayer.items():
            __saveTask(Task, taskMemory, child)
            if isinstance(grandchildLayer, Mapping):
                flattenedBlueprint.append((child, grandchildLayer))
            else:
                taskMemory[child].id = grandchildLayer
                taskMemory[child].isPrimitive = True
                primitiveIds.add(grandchildLayer)
            hierarchy.add_edge(taskMemory[parent], taskMemory[child])
    availableIds = set(range(len(hierarchy))) - primitiveIds
    for node in hierarchy.nodes():
        if node.id is None:
            node.id = availableIds.pop()
    return hierarchy


def buildTrajectory(blueprint):
    return nx.DiGraph(blueprint)
