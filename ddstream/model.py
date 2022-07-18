from microcluster import CoreMicroCluster

# general
import time, math
import numpy as np


class DDStreamModel:
    def __init__(
        self,
        broadcasted_var,
        # TODO: set epsilon = 16
        epsilon=16.0,
        # TODO: set minPoints = 10.0
        minPoints=5.0,
        beta=0.2,
        mu=10.0,
        lmbda=0.25,
        Tp=2,
    ):
        self.broadcasted_var = broadcasted_var

        # TODO: check
        self.eps = 0.01

        self.time = 0
        self.N = 0
        self.currentN = 0

        # list of microcluster objects
        self.pMicroClusters = []
        self.oMicroClusters = []

        self.broadcastPMic = None
        self.broadcastOMic = None

        self.initialized = False
        self.recursiveOutliersRMSDCheck = 1

        # only used during initilization (initDBSCAN)
        self.initArr = np.asarray([])
        self.tag = []

        self.epsilon = epsilon
        self.minPoints = minPoints
        self.beta = beta
        self.mu = mu
        self.Tp = Tp
        self.lmbda = lmbda

        self.modelDetectPeriod = 0

        self.lastEdit = 0
        self.tfactor = 1.0

        # idk
        self.AlldriverTime = 0.0
        self.AlldetectTime = 0.0
        self.AllprocessTime = 0.0

    def setCheck(self, t):
        self.tfactor = t

    def setTp(self):
        # this.Tp = Math.round(1 / this.lambda * Math.log((this.beta * this.mu) / (this.beta * this.mu - 1))) + 1
        pass

    # TODO: Test (test with initLabels and calculations)
    def initDBSCAN(
        self, ssc, initialEpsilon: float, path="./data/init_toy_dataset.csv"
    ):
        """
        Initialize DBSCAN microcluster.

        :param ssc              = spark context (needed to broadcast variable)
        :param initialEpsilon   =
        :param path             = must be path to container volume that cointaines initialization data
        """

        # 1. Read init data file
        with open(path) as f:
            lines, initLabels = f.readlines(), []
            for i, line in enumerate(lines):
                tmp = line.split(",")
                initLabels.append(int(tmp[-1]))  # assume label at end
                self.initArr = np.append(
                    self.initArr, np.asarray(list(map(lambda x: float(x), tmp[:-1])))
                )

        num_of_datapts, num_of_dimensions = i + 1, len(tmp) - 1
        self.initArr = self.initArr.reshape((-1, num_of_dimensions))

        print(f"Number of dimensions: {num_of_dimensions}")
        print(f"Number of initialization data points: {num_of_datapts}")
        assert num_of_datapts == len(self.initArr)  # float64

        # 2. Create core micro clusters
        self.tag = [0] * num_of_datapts  # 1 added, 0 not added to core micro cluster
        for i in range(num_of_datapts):
            # get neighborhood of this specific data point which has X dimensions
            neighborHoodList = self.getNeighborHood(i, initialEpsilon)
            # print(
            #     f"\nCHECK 1 : len(neighborHoodList) > self.minPoints = {len(neighborHoodList)} > {self.minPoints}"
            # )
            if len(neighborHoodList) > self.minPoints:
                self.tag[0] = 1
                # new microcluster for a core data point ( since neighborhood > minpts)
                # X num of dimensions -> len(cf2x) = len(cf1x) = X
                newMC = CoreMicroCluster(
                    cf2x=self.initArr[i] * self.initArr[i],  # element wise mult
                    cf1x=self.initArr[i],
                    weight=1.0,
                    t0=0,
                    lastEdit=0,
                    lmbda=self.lmbda,
                    tfactor=self.tfactor,
                )
                # expandCluster adds all neighborhood points to the newMC
                self.expandCluster(newMC, neighborHoodList, initialEpsilon)
                self.pMicroClusters.append(newMC)

        # TODO: Check if we need to .collect() some rdd first before creating the broadcast variable as they are doing
        self.broadcastPMic = ssc.sparkContext.broadcast(
            list(zip(self.pMicroClusters, range(len(self.pMicroClusters))))
        )
        print(
            f"broadcastPMic (after) = {self.broadcastPMic} \n {self.broadcastPMic.value}"
        )

        # TODO: Check if we need to initialize the outlierMC as well
        # self.broadcastOMic = ssc.sparkContext.broadcast(
        #     list(zip(self.oMicroClusters, range(len(self.oMicroClusters))))
        # )

        print(f"number of pMicroClusters = {len(self.pMicroClusters)}")
        print(f"number of oMicroClusters = {len(self.oMicroClusters)}")
        not_in_mc = len(list(filter(lambda x: x == 0, self.tag)))
        print(f"Points not added to MicroClusters = {not_in_mc}")

        self.initialized = True
        print("The initialization is complete\n")

    # TODO: Is this only used in initialization?
    # - I think yes
    # TODO: Test
    def getNeighborHood(self, pos: int, epsilon: float):
        """
        Get the indices of the points in the neighborhood of the point = self.initArr[pos]

        :param epsilon   = DBSCAN parameter (minimum radius of points in neighborhood)
        :param pos       = index of point we wish to calculate the neighborhood of (point = self.initArr[pos])
        :output idBuffer = list of indices (i) of points in self.initArr[] within the neighborhood of point = self.initArr[pos]
        """
        # print("\n\tIn getNeighborHood")
        idBuffer = []
        # for all datapoints
        # print(f"self.initArr = {self.initArr}\nlen(self.initArr) = {len(self.initArr)}")
        total_dist, pts = 0, 0
        for i in range(len(self.initArr)):
            # if not the one we're calculating the neighborhood for
            # also tag!=1 to check if it already exists in a micro cluster or in future (from expandCluster recursion)
            # print(f"\ni:{i} != pos:{pos} and self.tag[i]:{self.tag[i]} != 1", end="")
            if i != pos and self.tag[i] != 1:
                pts += 1
                # print("-> YES")

                # TODO: Figure out appropriate epsilon based on distance. ( Need to normalize init data )
                
                # Euclidean distance is calculated correctly: https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
                # print("Calculate distance: ")
                # print(f"A:self.initArr[pos] - self.initArr[i] = {self.initArr[pos]} - {self.initArr[i]} = {self.initArr[pos] - self.initArr[i]}")
                dist = np.linalg.norm(self.initArr[pos] - self.initArr[i])
                total_dist += dist
                # print(f"dist: np.linalg.norm(A) = {dist} < epsilon = {epsilon}", end="")
                # print(f"\tAVG distance = {total_dist/pts}", end="")
                if dist < epsilon:
                    # print("->YES")
                    # add the point to the neighborhood of the initArr[pos] point
                    # print(f"idBuffer = {idBuffer}")
                    idBuffer.append(i)
                    # print(f"(new) idBuffer = {idBuffer}")
        return idBuffer

    # TODO: Test
    def expandCluster(self, newMC, neighborHoodList, initialEpsilon):
        """
        Recursively add neighborhood points to micro cluster.

        :param newMC            = CoreMicroCluster to be expanded
        :param neighborHoodList = List of neighbor indices (for self.initArr) to be added to the CoreMicroCluster
        """
        # print("\n\tIn expandCluster:")
        for neighbor in neighborHoodList:
            # print(f"self.tag[neighbor] = {self.tag[neighbor]}")
            self.tag[neighbor] = 1
            # print(f" (after) self.tag[neighbor] = {self.tag[neighbor]}")

        # recursively expand the cluster based on newly added points
        for neighbor in neighborHoodList:
            newMC.insert(point=self.initArr[neighbor], time=0, n=1.0)

            # print(f"neighbor = {neighbor}\nneighborHoodList = {neighborHoodList}")
            neighborHoodList2 = self.getNeighborHood(neighbor, initialEpsilon)
            if len(neighborHoodList2) > self.minPoints:
                self.expandCluster(newMC, neighborHoodList2, initialEpsilon)

    def run(self, df, batch_id):
        """Run .foreachBatch()"""
        print(f"BATCH: {batch_id}", df, end="\n")
        rdd = df.rdd.map(tuple)  # after this it is like for each rdd.... Is it?
        # this does not update the broadcasted_var but rather the local copy
        # TODO: Should self.broad_var be global?

        # TODO: It is possible broadcasted_var does not work correctly because the data has
        #       not yet been .collected() to the "main" in order to do the update?
        print(f"BEFORE UPDATE: {self.broadcasted_var} {self.broadcasted_var.value}")
        self.broadcasted_var = rdd.context.broadcast((1, 2, 3, batch_id))
        print(f"AFTER UPDATE: {self.broadcasted_var} {self.broadcasted_var.value}")
        print()

        lastEdit = 0
        if not rdd.isEmpty() and self.initialized:
            t0 = time.time()
            print(f"The time is now {lastEdit}")
            assignations = assignToMicroCluster(rdd, self.eps)

    def assignToMicroCluster(self, row):
        if len(self.broadcasted_var.value) > 5:
            print("\n\nHERE\n\n")
        print(f"row assigned to micro cluster {row}")
        return (0, row)

    def assignToOutlierCluster(self):
        pass

    def computeDelta(self):
        pass

    def updateMicroClusters(self, assignations):
        print(f"update_mc {assignations}")
        pass

    def ModelDetect(self):
        pass

    def FinalDetect(self):
        pass

    # TODO: add setters/getters anything I missed that might be used
