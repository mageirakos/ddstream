# general
import time, math, copy
import numpy as np


class MacroCluster:
    def __init__(self, cf2x, cf1x, weight, num_labels=3, label=None):
        self.cf2x = cf2x
        self.cf1x = cf1x
        self.weight = weight
        self.pts = 1
        self.num_labels = num_labels
        self.lbl_counts = [0] * self.num_labels
        self.lbl_counts[label] += 1
        self.correctPts = 1
        self.label = self.lbl_counts.index(max(self.lbl_counts))
        self.purity = None

    def getCentroid(self):
        if self.weight > 0:
            return self.cf1x / self.weight
        else:
            return self.cf1x

    # TODO: Calculate Purity by treating the micro clusters as points
    def calcPurity(self):
        lbl = self.getLabel()
        self.purity = self.correctPts / self.lbl_counts[lbl]
        return self.purity

    def getLabel(self):
        self.correctPts = max(self.lbl_counts)
        self.label = self.lbl_counts.index(self.correctPts)
        return self.label


class DDStreamOfflineModel:
    def __init__(self, epsilon=16, mu=10):
        self.epsilon = epsilon
        self.mu = mu

        self.macroClusters = []
        self.tag = []

    def calcAvgPurity(self, mcs):
        """
        Calculate Average Purity by treating the micro clusters as points
        """
        num_of_clusters, purity_sum = len(mcs), 0
        if num_of_clusters == 0:
            return None
        for mc in mcs:
            purity_sum += mc.calcPurity()
        return purity_sum / num_of_clusters

    def offlineDBSCAN(self, coreMicroClusters):
        print("In OfflineDBSCAN")
        self.tag = [0] * len(coreMicroClusters)
        for i, mc in enumerate(coreMicroClusters):
            if self.tag[i] != 1:
                self.tag[i] = 1
                print(
                    f"mc_{i} | weight >= mu => {mc.weight} >= {self.mu:}\t{mc.weight >= self.mu:}"
                )
                if mc.weight >= self.mu:
                    neighborHoodList = self.getNeighborHood(i, coreMicroClusters)
                    print(f"#neighbors in mc_{i} = {len(neighborHoodList)}")
                    if len(neighborHoodList) > 0:
                        newMacro = MacroCluster(
                            cf2x=mc.cf2x,
                            cf1x=mc.cf1x,
                            weight=mc.weight,
                            num_labels=mc.num_labels,
                            label=mc.getLabel(),
                        )
                        self.macroClusters.append(newMacro)
                        self.expandCluster(coreMicroClusters, neighborHoodList)

        # for i, mc in enumerate(self.macroClusters):
        #     print(f"Macro cluster weight = {mc.weight}")
        #     print(f"Macro cluster center = {mc.getCentroid()}")
        return self.macroClusters

    def getNeighborHood(self, pos, points):
        """
        Get neighboring microclusters by comparing centroids.
        (Density reachable Def 4.1 Cao et al.)

        :param pos       = if of microcluster we wish to find the neighborhood of
        :param points    = list of existing microclusters
        :return idBuffer = contains neighboring microcluster ids
        """
        idBuffer = []
        for i in range(len(points)):
            if i != pos and self.tag[i] != 1:
                dist = np.linalg.norm(
                    points[i].getCentroid() - points[pos].getCentroid()
                )
                # print(f"dist: {dist} \t centr1: {points[i].getCentroid()} centr2: {points[pos].getCentroid()}")
                # print(f"2epsilon : {2*self.epsilon}")
                if dist < 2 * self.epsilon:
                    idBuffer.append(i)
        return idBuffer

    # TODO: Recursion Depth Exceeded -> look into it/debug
    # TODO: Fix Infinite Recursion
    def expandCluster(self, points, neighborHoodList):
        """
        Creates MacroClusters based on extended neightborhood of MicroClusters (points)
        Extended neighborhood is found recursively

        :param points           = list of CoreMicroClusters
        :param neighborHoodList = list of neighboring CoreMicroClusters
        """
        last_mc = len(self.macroClusters) - 1
        for neighbor in neighborHoodList:
            self.tag[neighbor] = 1
            self.macroClusters[last_mc].cf1x = (
                self.macroClusters[last_mc].cf1x + points[neighbor].cf1x
            )
            self.macroClusters[last_mc].cf2x = (
                self.macroClusters[last_mc].cf2x + points[neighbor].cf2x
            )
            self.macroClusters[last_mc].weight = (
                self.macroClusters[last_mc].weight + points[neighbor].weight
            )
            self.macroClusters[last_mc].weight = (
                self.macroClusters[last_mc].weight + points[neighbor].weight
            )
            assert self.macroClusters[last_mc].num_labels == points[neighbor].num_labels
            self.macroClusters[last_mc].lbl_counts = [
                a + b
                for a, b in zip(
                    self.macroClusters[last_mc].lbl_counts, points[neighbor].lbl_counts
                )
            ]
            self.macroClusters[last_mc].pts = points[neighbor].pts
            self.macroClusters[last_mc].label = self.macroClusters[last_mc].getLabel()

        for neighbor in neighborHoodList:
            neighborHoodList2 = self.getNeighborHood(neighbor, points)
            if len(neighborHoodList2) > 0:
                self.expandCluster(points, neighborHoodList2)

    # TODO: Fix
    def getFinalClusters(self, coreMicroClusters):
        self.macroClusters = self.offlineDBSCAN(coreMicroClusters)
        return self.macroClusters
        # a, n, a2 = [], [], []
        # if len(self.macroClusters) > 0:
        #     a = self.macroClusters.map(lambda x : (list(x.cf1x), list(x.cf2x)))
        #     n = self.macroClusters.map(lambda x : x.weight)
        #     r = zip(a, n)

    # TODO: Fix
    def getFinalMicroClusters(self, coreMicroClusters):
        pass

    # TODO: Fix
    def updateClusters(self, coreMicroClusters, maxIterations):
        pass

    # TODO: Fix
    def assign(self, input, cls):
        pass
