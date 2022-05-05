# TODO: finish functions


class CoreMicroCluster:
    # lambda != lambda (careful how I name my variables)
    def __init__(self, cf2x, cf1x, weight, t0, lastEdit, lmbda, tfactor):
        self.cf2x = cf2x
        self.cf1x = cf1x
        self.weight = weight
        self.t0 = t0
        self.lastEdit = lastEdit
        self.lmbda = lmbda
        self.tfactor = tfactor

    def getCentroid(self):
        pass

    def getRMSD(self):
        pass

    # maybe there is already a copy() func for python class objects instead of defining it
    def copy(self):
        pass

    def insert(self, point, n):
        pass

    def insert(self, point, time, n):
        pass

    def mergeWithOtherMC(self, otherMC):
        pass

    # TODO: add time dependent functions like getWeight(t) etc.
