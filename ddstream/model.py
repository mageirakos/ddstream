from microcluster import CoreMicroCluster

# general
import time, math, copy
import numpy as np


class DDStreamModel:
    def __init__(
        self,
        numDimensions,
        # broadcasted_var,
        # TODO: original: epsilon=16.0,
        #TODO: change self.epsilon to smaller as most distances <1.0 (not even 16) -> because we standardized
        epsilon=0.2,
        # TODO: set minPoints = 10.0
        minPoints=10.0,
        beta=0.2,
        mu=10.0,
        lmbda=0.25,
        Tp=2,
    ):
        # self.broadcasted_var = broadcasted_var
        self.numDimensions = numDimensions
        # TODO: check correct initialization etc.
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
        print("START of Initialization")
        # 1. Read init data file
        with open(path) as f:
            lines, initLabels = f.readlines(), []
            for i, line in enumerate(lines):
                tmp = line.split(",")
                # print(f"adding {tmp} to initArr")
                initLabels.append(int(tmp[-1]))  # assume label at end
                self.initArr = np.append(
                    self.initArr, np.asarray(list(map(lambda x: float(x), tmp[:-1])))
                )
        # print(f"Final initArr {self.initArr}")
        num_of_datapts, num_of_dimensions = i + 1, len(tmp) - 1
        # init dims must be same as training data stream dims
        assert num_of_dimensions == self.numDimensions
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
                # print(f"Point {self.initArr[i]} has neighborhood > minPts = {neighborHoodList}")
                # print(f"Creating newMC {newMC} with cf1x = {newMC.cf1x}")
                # expandCluster adds all neighborhood points to the newMC
                self.expandCluster(newMC, neighborHoodList, initialEpsilon)
                self.pMicroClusters.append(newMC)

        # TODO: Check if we need to .collect() some rdd first before creating the broadcast variable as they are doing
        self.broadcastPMic = ssc.sparkContext.broadcast(
            list(zip(self.pMicroClusters, range(len(self.pMicroClusters))))
        )
        #TODO: Fix initDBSCAN to create outlier clusters
        # self.broadcastOMic = ssc.sparkContext.broadcast(
        #     list(zip(self.oMicroClusters, range(len(self.oMicroClusters))))
        # )
        print(
            f"broadcastPMic (after) = {self.broadcastPMic} \n {self.broadcastPMic.value}"
        )
        # print(
        #     f"broadcastOMic (after) = {self.broadcastOMic} \n {self.broadcastOMic.value}"
        # )
        # for mc in self.broadcastPMic.value:
        #     print(f"MC: {mc[0]}\n  weight: {mc[0].weight}")

        # TODO: Check if we need to initialize the outlierMC as well
        # self.broadcastOMic = ssc.sparkContext.broadcast(
        #     list(zip(self.oMicroClusters, range(len(self.oMicroClusters))))
        # )

        print(f"number of pMicroClusters = {len(self.pMicroClusters)}")
        # print centroids
        for mc in self.broadcastPMic.value:
            print(f"mc =  {mc[1]} with centroid =  {mc[0].getCentroid()}")
        print(f"number of oMicroClusters = {len(self.oMicroClusters)}")
        not_in_mc = len(list(filter(lambda x: x == 0, self.tag)))
        print(f"Points not added to MicroClusters = {not_in_mc}")

        self.initialized = True
        print("END of Initialization\n")

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
                # TODO: Possible problem look into https://stackoverflow.com/questions/66806583/np-linalg-norm-ord-2-not-giving-euclidean-norm
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
            newMC.insertAtT(point=self.initArr[neighbor], time=0, n=1.0)

            # print(f"neighbor = {neighbor}\nneighborHoodList = {neighborHoodList}")
            neighborHoodList2 = self.getNeighborHood(neighbor, initialEpsilon)
            if len(neighborHoodList2) > self.minPoints:
                print(f"in expandCluster for {newMC} with neighborhood2 {neighborHoodList2}")
                self.expandCluster(newMC, neighborHoodList2, initialEpsilon)

    # TODO: Make sure that createing multiple rdd.contect.broadcast() objects is not a problem
    # and that all the workers use the same one
    # TODO: It is possible broadcasted_var does not work correctly because the data has
    # not yet been .collected() to the "main" in order to do the update?

    def run(self, streaming_df, batch_id):
        """Run .foreachBatch()"""
        print(f"BATCH: {batch_id}", streaming_df, end="\n")
        # Step 0: Initializatino of Micro Clusters must have already been done
        # Step 1: Split to each rdd
        #: why the tuple? -> to get rid of Row(..) https://intellipaat.com/community/7578/how-to-convert-a-dataframe-back-to-normal-rdd-in-pyspark
        rdd = streaming_df.rdd.map(tuple)
        # # only for printing:
        # to_be_printed = rdd.collect()
        # for row in to_be_printed:
        #     print(f"rdd: _1 = {str(row[0])} , _2 = {str(row[1])}, all = {str(row)}")

        # Step 2: Make sure batch is not empty and p-mc have been initialized
        if not rdd.isEmpty() and self.initialized:
            t0 = time.time()
            print(f"The time is now: {self.lastEdit}")

            # Step 3: Assign each point in the rdd to closest micro cluster (if radius<epsilon)
            assignations = self.assignToMicroCluster(rdd)
            # # only for printing:
            # to_be_printed = assignations.collect()
            # for row in to_be_printed:
            #     print(f"assignations : minIndex={str(row[0])} , a={str(row[1])}")
            # TODO: understand updateMicroClusters()
            self.updateMicroClusters(assignations)

    def assignToMicroCluster(self, rdd):
        """
        Assign each point in the batch to a pMicroCluster.
        :param rdd       : (key, <features>)
        :return rdd      : (minIndex, (key, <features>))
        :return minIndex : index of closest microcluster ( -1 indicates no assignment)
        """
        print("In assignToMicroCluster")
        # STEP 1: For each element in the RDD
        # print(f"broadcastPMic: {self.broadcastPMic} {self.broadcastPMic.value} {len(self.broadcastPMic.value)}")
        # print(f"{self.broadcastPMic.value[-1][0]}")
        # TODO: Why print not work in assign() -> because in map??
        def assign(a):
            """:param a : point/row in batch format=(None, DenseVector([<features>]))"""
            # TODO: check if correct:
            # TODO: Maybe we need to turn it back to DenseVector at the end of the function
            a = a[0], a[1].toArray()

            minDist, minIndex = float("inf"), -1
            # TODO: init pcopy
            # Step 2: if there are p microclusters
            tmp = []
            if len(self.broadcastPMic.value) > 0:
                for mc in self.broadcastPMic.value:
                    # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.linalg.DenseVector.html#pyspark.mllib.linalg.DenseVector.toArray
                    # squaredDistance equivalent
                    # TODO: Possible problem look into https://stackoverflow.com/questions/66806583/np-linalg-norm-ord-2-not-giving-euclidean-norm
                    # Step 3: Calculate squared distances of point from centroids of each microcluster
                    dist = np.linalg.norm(a[1] - mc[0].getCentroid())  # ** 2
                    tmp.append(round(dist, 1))
                    if dist < minDist:
                        minDist, minIndex = dist, mc[1]

                    # mc = {mc_location, index}
                    # get the distance between the current point and the center of the microcluster
                    # TODO: what is a[1] here? -> features?
                    # dist = self.squaredDistance(a[1], mc[0].getCentroid)

                # TODO: Understand this
                # Step 4: Create a copy of the closest p microcluster to the point and insert it into it
                pcopy = copy.deepcopy(self.broadcastPMic.value[minIndex][0])

                # pcopy = self.broadcastPMic.value[minIndex][0].copy()
                n = 0
                # with open('blah.txt','a') as f:
                #     n += 1
                #     f.write(f"n={n} bPMic.value[{minIndex}][0] = {self.broadcastPMic.value[minIndex][0]} weight = {self.broadcastPMic.value[minIndex][0].weight}\n")
                #     f.write(f"n={n} pcopy={pcopy}, weight = {pcopy.weight}\n")
                #     f.write(f"n={n} inserting...\n")

                pcopy.insert(a, 1)
                # pcopy.insertAtT(a[1], a[0], 1)
                # with open('blah.txt','a') as f:
                #     n += 1
                #     f.write(f"n={n} after insert after RMSD\n")
                #     f.write(f"n={n}pcopy={pcopy}, weight = {pcopy.weight}\n")
                #     f.write(f"n={n}pcopy RMSD={pcopy.getRMSD()}\n")

                # Step 5: If the radius of this microcluster is larger than the epsilon then reset the minIndex (i.e the point is not inserted)
                # - we still need to insert the point to the microcluster in the future
                # - this only returns a tuple of (index_of_closest_mc, point) === (minIndex, a)
                #TODO: change self.epsilon to smaller as most distances <1.0 (not even 16) -> because we standardized
                if pcopy.getRMSD() > self.epsilon:
                    minIndex = -1
            return minIndex, a#, pcopy.getRMSD(), self.epsilon

        return rdd.map(lambda a: assign(a))

    def assignToOutlierCluster(self, rdd):
        """
        For the points not assigned to a primary microcluster, assing them to an outlier microcluster.
        :param rdd: RDD of outlier points (timestamp, <features>)
        """
        if not self.broadcastOMic:
            return None
        def assign(a):
            # if we have no outlier microclusters
            minIndex = -1
            #TODO: We don't seem to have created any outlier micro clusters anywhere.
            if len(self.broadcastOMic.value) > 0:
                minDist = float("inf")
                for mc in self.broadcastOMic.value:
                    dist = np.linalg.norm(a[1] - mc[0].getCentroid())
                    if dist < minDist:
                        minDist, minIndex = dist, mc[1]

                ocopy = copy.deepcopy(self.broadcastOMic.value[minIndex][0])
                ocopy.insert(a, 1)
                if ocopy.getRMSD > self.epsilon:
                    minIndex = -1
                return minIndex, a
        return rdd.map(lambda a: assign(a))

    # TODO: Fix 
    def computeDelta(self, sortedRDD):
        """
        Compute the Delta for Order-Aware Local Update Step.
        The Delta represents the effect to each microcluster of adding the points in order to the mc they belong to.
        
        At the end of the function we collect all the Deltas from the worker to the driver node for the global update step
        to be computed. The global update step actually applies the deltas/effects of insertion to the microcluters,
        preparing them for the next batch.

        :param sortedRDD: RDD grouped and sorted by microcluster id that contains list of numpy arrays with features of each point.
                          Example: (0, [(1, array([ 0.60528983, -1.5069482 ])), (2, array([ 1.11312873, -1.06434713]))])
                          Format (<microcluster_id>, list[ np.array(<features_point1>), ..., np.array(<features_pointN>) ])
        :output : (microcluster_id, <(delta_cf1x, delta_cf2x, delta_n, delta_t)>)
        """
        print("In computeDelta")
        # print(f"computeDelta input = {sortedRDD.collect()}")
        # TODO: Do some python/spark magic for it to be in parallel
        # basically a for loop to calculate deltas over entire rdd
        #TODO: How do we aggregate the Deltas once we .collect them? Since a different Delta has been computed at each node.
        def calcDelta(x):
            """
            :param x: list of points :(<timestamp>, list<feature values>)
            :output (delta_cf1x, delta_cf2x, delta_n, delta_t)
            """
            #TODO: Give a proper definition of n. Basically n is used in weight where we +1 when we add a point
            # but because here we add eg. 3 poitns do n = (l1*1)+(l2*1)+(l3*1) < 3 (lambda affect of each point)
            delta_cf1x, delta_cf2x, delta_n, delta_t = np.zeros(self.numDimensions), np.zeros(self.numDimensions), 0.0, 0
            # with open('blah.txt','a+') as f:
            #         f.write(f"type = {type(x)}\t")
            #         f.write(f"x = {x}\n")
            # for each point in order apply decay and compute the delta "effect" to mc
            for row in x:     
                arrivalT, featVals = row[0], row[1]
                assert type(featVals) == type(delta_cf1x) == type(delta_cf1x)
                lmbda = math.pow(2, -self.lmbda * (arrivalT - delta_t))
                delta_cf1x = (delta_cf1x * lmbda) + featVals
                delta_cf2x = (delta_cf1x * lmbda) + featVals * featVals 
                delta_n = (delta_n * lmbda) + 1
                delta_t = max(arrivalT, delta_t)
            return delta_cf1x, delta_cf2x, delta_n, delta_t
        return sortedRDD.mapValues(lambda x: calcDelta(x)).collect()

        

    # TODO: Understand and code up function.
    def updateMicroClusters(self, assignations):
        ##### ---- Global Update Step
        print("In updateMicroClusters")
        # all are RDD[(Int, (<key>Long, Vector[<features(Double)>]))]:
        # dataInPmic = None
        # dataInAndOut = None
        # dataOut = None
        # dataInOmic = None
        # outliers = None

        # why persist: https://stackoverflow.com/questions/31002975/how-to-use-rdd-persist-and-cache
        assignations.persist()
        print(f"assignations: {assignations.collect()}")
        # # for printing:
        # to_be_printed = assignations.collect()
        # for row in to_be_printed:
        #     print(f"assignations: {str(row)}")

        # print("updateMicroClusters: Step 1")
        # Step 1: filter out the points that were not assigned to any microcluster
        dataInPmic = assignations.filter(lambda x: x[0] != -1)
        print(f"dataInPmic: {dataInPmic.collect()}")
        # to_be_printed = dataInPmic.collect()
        # for row in to_be_printed:
        #     print(f"dataInPmic: {str(row)}")
        # works ok until here

        # TODO: write aggregateFunction if needed:
        # aggregateFunction = lambda x: pass

        # Step 2: Sort the data assigned to Pmic based on arrival order
        # print("updateMicroClusters: Step 2")
        # if you want to see dataInPmic.groupByKey() before sorting: https://stackoverflow.com/questions/29717257/pyspark-groupbykey-returning-pyspark-resultiterable-resultiterable
        # note: the data seems to be process in order by default but we follow the sorting logic. To test it use prints bellow
        # sortedRDD = dataInPmic.groupByKey().map(lambda x : (x[0], list(x[1])))
        # # to_be_printed here will be a ResultIterable so we need a double loop for contents
        # to_be_printed = sortedRDD.collect()
        # for row in to_be_printed:
        #     print(f"pre_sortedRDD:  {str(row)}")
        # x.toList.sortBy(key=x[0])
        sortedRDD = dataInPmic.groupByKey().mapValues(lambda x: sorted(list(x), key=lambda y : y[0]))
        # to_be_printed = sortedRDD.collect()
        # for row in to_be_printed:
        #     print(f"sortedRDD:  {str(row)}")

        # Step 3: 
        dataInPmicSS = self.computeDelta(sortedRDD)
        # print(f"dataInPmicSS: {dataInPmicSS}")
        
        # datapoints are data not assigned to pmic that might belong to outlier mc or are noise
        dataInAndOut = assignations.filter(lambda x: x[0] == -1).map(lambda x: x[1])
        print(f"dataInAndOut: {dataInAndOut.collect()}")
        
        #TODO: dataOut seems to not be needed --> same as 'outliers'
        # dataOut: data not assigned to primary microclusters that have not been assigned to outlier microcluster
        # or not assigned to any cluster -> index == -1
        dataOut = self.assignToOutlierCluster(dataInAndOut)
        # data in outlier microclusters
        #TODO: Adjust code to handle:
        if dataOut:
            print(f"dataOut = {dataOut.collect()}")
            dataOut.persist()
            
            dataInOmic = dataOut.filter(lambda x: x[0] != -1)
            # outliers : data not assigned to outlier microclusters -> complete outliers
            outliers = dataOut.filter(lambda x: x[0] == -1).map(lambda x: x[1])
            # group by omc id and sort by arrivel and compute delta
            omicSortedRDD = dataInOmic.groupByKey().mapValues(lambda x: sorted(list(x), key=lambda y : y[0]))
            dataInOmicSS = self.computeDelta(omicSortedRDD)
            
            realOutliers = outliers.collect()

        else:
            dataInOmic, outliers, omicSortedRDD, dataInOmicSS  = None, None, None, None
            realOutliers = None
            
            
            
        totalIn = 0 #TODO: Remove as it might be useless

        # optimization not needed necessarily
        if realOutliers and len(realOutliers) > 35_000:
            realOutliers = realOutliers.sortByKey().collect()
        
        assignations.unpersist()
        if dataOut: dataOut.unpersist()
        #TODO: What is this
        DriverTime = time.time()

        ##### ---- Global Update Step
        #TODO: Continue code
        print("\t----Global Update Step:----")
        print(f"dataInPmicSS= {dataInPmicSS}")
        if len(dataInPmicSS) > 0:
            for ss in dataInPmicSS:
                i, ts = ss[0], ss[1][3]
                print(f"i = {i}, time = {ts}")

        



    def ModelDetect(self):
        pass

    def FinalDetect(self):
        pass

    # TODO: add setters/getters anything I missed that might be used
