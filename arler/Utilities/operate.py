import itertools


def findRoot(digraph):
    candidates = [node for node in digraph if digraph.in_degree(node) == 0]

    if not candidates:
        raise ValueError("No root found, aborting.")

    if len(candidates) > 1:
        raise ValueError("More than one tree in the hierarchy, aborting.")

    return candidates.pop()


def mergeLists(*args):
    return itertools.chain.from_iterable(args)
