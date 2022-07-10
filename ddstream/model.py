from microcluster import CoreMicroCluster

# general
import time


class DDStreamModel:
    def __init__(
        self,
        broadcasted_var,
        epsilon=16.0,
        minPoints=10.0,
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

        self.pMicroClusters = []  # array of microcluster objects
        self.oMicroClusters = []

        self.broadcastPMic = None
        self.broadcastOMic = None

        self.initialized = False
        self.recursiveOutliersRMSDCheck = 1

        self.initArr = []
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

    def initDBSCAN(self):
        # initialize and broadcast pmc/omc
        self.initialized = True

    def getNeighborHood(self, pos, epsilon):
        pass

    def expandCluster(self):
        pass

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
