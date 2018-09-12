class Variable:
    def __init__(self, name, value=None, data=None):
        self.name = name
        self.value = value

    def equals(self, value):
        self.value == value


class Term:
    def __init__(self, variable, sentinel):
        self.variable = variable
        self.sentinel = sentinel

    def __bool__(self):
        return self.variable.equals(self.sentinel)

    def __nonzero__(self):
        return self.__bool__()


class Condition:
    def __init__(self, terms=set()):
        self.terms = terms

    def __bool__(self):
        return all(map(lambda term: bool(term), self.data))

    def __nonzero__(self):
        return self.__bool__()

