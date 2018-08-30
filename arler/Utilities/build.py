from collections.abc import Mapping
from collections import defaultdict
import networkx as nx


def __saveNamedObject(ObjectClass, memory, objectName):
    if objectName not in memory:
        memory[objectName] = ObjectClass(objectName)


def buildPriorities(blueprint, Task):
    if len(blueprint) != 1:
        raise ValueError("The hierarchy given is not a rooted tree.")

    hierarchy = nx.DiGraph()
    flattenedBlueprint = list(blueprint.items())
    taskMemory = dict()
    primitiveIds = set()
    while flattenedBlueprint:
        parent, childLayer = flattenedBlueprint.pop()
        __saveNamedObject(Task, taskMemory, parent)
        for child, grandchildLayer in childLayer.items():
            __saveNamedObject(Task, taskMemory, child)
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


def buildScaffold(blueprint, componentClass):
    hasUsed = dict()
    scaffold = nx.DiGraph()
    for parent, childLayer in blueprint.items():
        __saveNamedObject(componentClass, hasUsed, parent)
        for child, childAttrs in childLayer.items():
            __saveNamedObject(componentClass, hasUsed, child)
            scaffold.add_edge(hasUsed[parent], hasUsed[child], **childAttrs)

    return scaffold
