from microcluster import CoreMicroCluster
from utils import *

# general
import time, math, copy
import numpy as np

# TODO: Check that weight etc. is calculated/decaied based on cluster t0 ( which is when cluster was created etc. )


class DDStreamModel:
    def __init__(
        self,
        numDimensions,
        batchTime,  # TODO: See where this is used and use it ()
        # TODO: change self.epsilon to smaller as most distances <1.0 (not even 16 (original)) -> because we standardized
        epsilon,
        beta,
        mu,
        lmbda,
        num_labels,
        Tp=4,
    ):
        self.numDimensions = numDimensions
        self.batchTime = batchTime
        # TODO: check correct initialization etc.
        self.eps = 0.01
        self.NUM_LABELS = num_labels

        self.time = 0
        self.N = 0
        self.currentN = 0
        self.batch_id = None

        # list of microcluster objects
        self.pMicroClusters = []
        self.oMicroClusters = []

        self.broadcastPMic = None
        self.broadcastOMic = None

        self.initialized = False
        self.recursiveOutliersRMSDCheck = 1

        # only used during initilization (initDBSCAN)
        self.initArr = np.asarray([])
        self.initLabels = []
        self.tag = []

        self.epsilon = epsilon
        self.beta = beta
        self.mu = mu
        self.Tp = Tp
        self.lmbda = lmbda

        self.modelDetectPeriod = 0

        self.lastEdit = 0

        # idk
        self.AlldriverTime = 0.0
        self.AlldetectTime = 0.0
        self.AllprocessTime = 0.0

    def getMicroClusters(self):
        #TODO: what to return here?
        print(f"\tin getMicroClusters")
        print(f"self.pMicroClusters = {self.pMicroClusters}\n")
        print(f"self.broadcastPMic = {self.broadcastPMic.value}\n")

        return self.pMicroClusters

    # TODO: Test (test with initLabels and calculations)
    # TODO: Handle .lbl_counts etc. in code below (probably in insert?)
    def initDBSCAN(self, ssc, initialEpsilon=0.5, path="./data/init_toy_dataset.csv"):
        """
        Initialize DBSCAN microcluster.

        :param ssc              = spark context (needed to broadcast variable)
        :param initialEpsilon   =
        :param path             = must be path to container volume that cointaines initialization data
        """
        print("START of Initialization")
        self.batch_id = -1
        # 1. Read init data file
        with open(path) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                tmp = line.split(",")
                # expecting int label
                self.initLabels.append(int(tmp[-1]))
                self.initArr = np.append(
                    self.initArr, np.asarray(list(map(lambda x: float(x), tmp[:-1])))
                )
        # print(f"Final initArr {self.initArr}")
        num_of_datapts, num_of_dimensions = i + 1, len(tmp) - 1
        # init dims must be same as training data stream dims
        # print(f"assert {num_of_dimensions} == {self.numDimensions}")
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
            #     f"\n{i}) CHECK 1 : len(neighborHoodList) > self.mu = {len(neighborHoodList)} > {self.mu}"
            # )
            if len(neighborHoodList) > self.mu:
                self.tag[0] = 1
                # new microcluster for a core data point ( since neighborhood > mu)
                # X num of dimensions -> len(cf2x) = len(cf1x) = X

                newMC = CoreMicroCluster(
                    cf2x=self.initArr[i] * self.initArr[i],  # element wise mult
                    cf1x=self.initArr[i],
                    weight=1.0,
                    t0=0,  # timestamp of CoreMicroCluster creation
                    lastEdit=0,  # when initializing same as timestamp of CoreMicroCluster creation
                    lmbda=self.lmbda,
                    num_labels=self.NUM_LABELS,
                    label=self.initLabels[i],
                )
                # print(f"Point {self.initArr[i]} has neighborhood > mu = {neighborHoodList}")
                # print(f"Creating newMC {newMC} with cf1x = {newMC.cf1x}")
                # expandCluster adds all neighborhood points to the newMC
                self.expandCluster(newMC, neighborHoodList, initialEpsilon)
                self.pMicroClusters.append(newMC)

        self.broadcastPMic = ssc.sparkContext.broadcast(
            list(zip(self.pMicroClusters, range(len(self.pMicroClusters))))
        )
        # TODO: Fix initDBSCAN to create outlier clusters
        self.broadcastOMic = ssc.sparkContext.broadcast(
            list(zip(self.oMicroClusters, range(len(self.oMicroClusters))))
        )
        print(
            f"broadcastPMic (-1) = {self.broadcastPMic} \n {self.broadcastPMic.value}"
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
        # for mc in self.broadcastPMic.value:
        #     print(f"mc =  {mc[1]} center =  {mc[0].getCentroid()} w = {mc[0].weight} lbl_counts = {mc[0].lbl_counts} pts = {mc[0].pts} label = {mc[0].getLabel()} correctPts = {mc[0].correctPts}")
        # print(f"number of oMicroClusters = {len(self.oMicroClusters)}")
        not_in_mc = len(list(filter(lambda x: x == 0, self.tag)))
        print(f"Points not added to MicroClusters = {not_in_mc}")

        self.initialized = True
        print("END of Initialization\n")

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
                # Euclidean distance : https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
                # print("Calculate distance: ")
                # print(f"A:self.initArr[pos] - self.initArr[i] = {self.initArr[pos]} - {self.initArr[i]} = {self.initArr[pos] - self.initArr[i]}")
                # TODO: Possible problem look into https://stackoverflow.com/questions/66806583/np-linalg-norm-ord-2-not-giving-euclidean-norm
                dist = np.linalg.norm(self.initArr[pos] - self.initArr[i])
                total_dist += dist
                # print(f"dist: np.linalg.norm(A) = {dist} < epsilon = {epsilon}", end="")
                # print(f"\tAVG distance = {total_dist/pts}", end="")
                if dist < epsilon:
                    # print("-> 2 YES")
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
            newMC.insertAtT(
                point=self.initArr[neighbor],
                time=0,
                n=1,
                label=self.initLabels[neighbor],
            )

            # print(f"neighbor = {neighbor}\nneighborHoodList = {neighborHoodList}")
            neighborHoodList2 = self.getNeighborHood(neighbor, initialEpsilon)
            if len(neighborHoodList2) > self.mu:
                # print(f"in expandCluster for {newMC} with neighborhood2 {neighborHoodList2}")
                self.expandCluster(newMC, neighborHoodList2, initialEpsilon)

    # TODO: Make sure that createing multiple rdd.contect.broadcast() objects is not a problem
    # and that all the workers use the same one

    def calcAvgPurity(self, mcs):
        """
        Calculate the average purity as defined in Cao et al.
        where avg_pur = nominator / denominator
                nominator = Σ (dominant_lbl_pts / total_points) ( Σ over all K clusters )
                demoninator = K (num_of_clusters)
        :param mcs: broadcasted object (either PMIC or OMIC)
        :return average_purity_metric  (float)
        """
        num_of_clusters, purity_sum = len(mcs), 0
        if num_of_clusters == 0:
            return None
        for mc in mcs:
            actual_mc = mc
            if type(mc) == tuple:
                actual_mc, _ = mc[0], mc[1]
            purity_sum += actual_mc.calcPurity()
        return purity_sum / num_of_clusters

    def run(self, streaming_df, batch_id):
        global DETAILS, MICRO_CLUSTERS
        """Run .foreachBatch()"""
        self.batch_id = batch_id
        print(f"\nBATCH: {batch_id}", streaming_df, end="\n")
        # Step 0: Initializatino of Micro Clusters must have already been done
        # Step 1: Split to each rdd
        #: why the tuple? -> to get rid of Row(..) https://intellipaat.com/community/7578/how-to-convert-a-dataframe-back-to-normal-rdd-in-pyspark
        rdd = streaming_df.rdd.map(tuple)
        # # only for printing:
        # to_be_printed = rdd.collect()
        # print(f"RDD: {to_be_printed}")
        # for row in to_be_printed:
        #     print(f"rdd: _1 = {str(row[0])} , _2 = {str(row[1])}, all = {str(row)}")

        # Step 2: Make sure batch is not empty and p-mc have been initialized
        if not rdd.isEmpty() and self.initialized:
            batch_start_t = time.time()
            print(f"The time is now: {self.lastEdit}")
            print()
            # Step 3: Assign each point in the rdd to closest micro cluster (if radius<epsilon)
            assignations = self.assignToMicroCluster(rdd)
            # # only for printing:
            # to_be_printed = assignations.collect()
            # print(f"ASSIGNATIONS: {to_be_printed}")
            # for row in to_be_printed:
            #     print(f"assignations : minIndex={str(row[0])} , feats={str(row[1])}, label={str(row[2])}")

            self.updateMicroClusters(assignations)
            self.broadcastPMic = rdd.context.broadcast(
                list(zip(self.pMicroClusters, range(len(self.pMicroClusters))))
            )
            
            self.broadcastOMic = rdd.context.broadcast(
                list(zip(self.oMicroClusters, range(len(self.oMicroClusters))))
            )
            detectTime = time.time()
            # every 4 batches apply remove decayied pMicroClusters
            if self.modelDetectPeriod != 0 and self.modelDetectPeriod % 4 == 0:
                # print(f"Start Model Checking - Batch{batch_id}")
                self.ModelDetect()
                self.broadcastPMic = rdd.context.broadcast(
                    list(zip(self.pMicroClusters, range(len(self.pMicroClusters))))
                )
                self.broadcastOMic = rdd.context.broadcast(
                    list(zip(self.oMicroClusters, range(len(self.oMicroClusters))))
                )

            self.AllprocessTime += time.time() - batch_start_t
            detectTime = time.time() - detectTime
            self.AlldetectTime += detectTime
            self.modelDetectPeriod += 1
            # print(f"Model checking completed... detect time taken (ms) = {detectTime}")
            # ------------------ STATUS PRINTS ---------------------
            # primary
            print("Primary mc:")
            print(f"After batch {batch_id} number of p microclusters: {len(self.broadcastPMic.value)}")
            for i, mc in enumerate(self.broadcastPMic.value):
                print(f"pmic_{i} last_t={mc[0].lastEdit} w = {mc[0].weight} center = {mc[0].getCentroid()} lbl_counts = {mc[0].lbl_counts} pts = {mc[0].pts} label = {mc[0].getLabel()} correctPts = {mc[0].correctPts} purity = {mc[0].calcPurity()}")
            # primary purity
            pmic_avg_purity = self.calcAvgPurity(self.broadcastPMic.value)
            print(f"After batch {batch_id} number of p microclusters: {len(self.broadcastPMic.value)}")
            print(f"AVERAGE PURITY (pmic) = {pmic_avg_purity}")
            # outlier
            print(f"Outlier mc :")
            for i, mc in enumerate(self.broadcastOMic.value):
                print(f"omic_{i} last_t={mc[0].lastEdit} w ={mc[0].weight} center = {mc[0].getCentroid()} lbl_counts = {mc[0].lbl_counts} pts = {mc[0].pts} label = {mc[0].getLabel()} correctPts = {mc[0].correctPts} purity = {mc[0].calcPurity()}")
            
            # print(f"batch_{batch_id} OutlierMC ({batch_id}) = {self.oMicroClusters} \n {self.oMicroClusters.value}")
            # omic_avg_purity = self.calcAvgPurity(self.broadcastOMic.value)
            # print(
            #     f"After batch {batch_id} number of o microclusters: {len(self.broadcastOMic.value)}"
            # )
            # print(f"AVERAGE PURITY (omic) = {omic_avg_purity}")
            # ------------------ EXPERIMENT DATA ---------------------
            # TODO NOW: Save MICRO_CLUSTERS, MICRO_METRICS '/data/experiments/<dataset_d>
            
            for i, mc in enumerate(self.broadcastPMic.value):
                microcl, _ = mc[0], mc[1]
                append_to_MICRO_CLUSTERS(
                    batch_id=self.batch_id,
                    microcluster_id=id(microcl),
                    centroid=microcl.getCentroid().tolist(),
                    cf1x=microcl.cf1x.tolist(),
                    cf2x=microcl.cf2x.tolist(),
                    weight=microcl.weight,
                    t0=microcl.t0,
                    lastEdit=microcl.lastEdit,
                    pts=microcl.pts,
                    lbl_counts=microcl.lbl_counts,
                    correctPts=microcl.correctPts,
                    label=microcl.getLabel(),
                    purity=microcl.calcPurity())
            # append avg purity from above microclusters in current batch
            avg_purity = self.calcAvgPurity(self.broadcastPMic.value)
            append_to_MICRO_METRICS(batch_id=self.batch_id, name="PURITY(avg)", value=avg_purity)

    def assignToMicroCluster(self, rdd):
        """
        Assign each point in the batch to a pMicroCluster.
        :param rdd       : (key, <features>)
        :return rdd      : (minIndex, (key, <features>))
        :return minIndex : index of closest microcluster ( -1 indicates no assignment)
        """
        # print("In assignToMicroCluster")
        # STEP 1: For each element in the RDD
        # print(f"broadcastPMic: {self.broadcastPMic} {self.broadcastPMic.value} {len(self.broadcastPMic.value)}")
        # print(f"{self.broadcastPMic.value[-1][0]}")
        def assign(a):
            """:param a : point/row in batch format=(None, DenseVector([<features>]))"""
            # TODO: Maybe we need to turn it back to DenseVector at the end of the function
            # TODO: a[2] is label, should it be int? or just string
            a = a[0], a[1].toArray(), a[2]

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

                # Step 4: Create a copy of the closest p microcluster to the point and insert it into it
                pcopy = copy.deepcopy(self.broadcastPMic.value[minIndex][0])

                # pcopy = self.broadcastPMic.value[minIndex][0].copy()
                n = 0
                pcopy.insert(a, 1)

                # Step 5: If the radius of this microcluster is larger than the epsilon then reset the minIndex (i.e the point is not inserted)
                # - we still need to insert the point to the microcluster in the future
                # - this only returns a tuple of (index_of_closest_mc, point) === (minIndex, a)
                if pcopy.getRMSD() > self.epsilon:
                    minIndex = -1
            else:
                minIndex = -1
            return minIndex, a

        return rdd.map(lambda a: assign(a))

    def assignToOutlierCluster(self, rdd):
        """
        For the points not assigned to a primary microcluster, assing them to an outlier microcluster.
        :param rdd: RDD of outlier points (timestamp, <features>)
        """

        def assign(a):
            # if we have no outlier microclusters
            minIndex = -1
            # TODO: We don't seem to have created any outlier micro clusters anywhere.
            if len(self.broadcastOMic.value) > 0:
                minDist = float("inf")
                for mc in self.broadcastOMic.value:
                    dist = np.linalg.norm(a[1] - mc[0].getCentroid())
                    if dist < minDist:
                        minDist, minIndex = dist, mc[1]

                ocopy = copy.deepcopy(self.broadcastOMic.value[minIndex][0])
                ocopy.insert(a, 1)
                if ocopy.getRMSD() > self.epsilon:
                    minIndex = -1
            else:
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
        # print("In computeDelta")
        # print(f"computeDelta input = {sortedRDD.collect()}")
        # TODO: How do we aggregate the Deltas once we .collect them? Since a different Delta has been computed at each node.
        def calcDelta(x):
            """
            :param x: list of points :(<timestamp>, list<feature values>)
            :output (delta_cf1x, delta_cf2x, delta_n, delta_t)
            """
            # TODO: Pass label in this function
            # TODO: l must be int() so in KDD_99 we need to turn obj to int
            lbl_counts = [0] * self.NUM_LABELS

            # TODO: Give a proper definition of n. Basically n is used in weight where we +1 when we add a point
            # but because here we add eg. 3 poitns do n = (l1*1)+(l2*1)+(l3*1) < 3 (lambda affect of each point)
            delta_cf1x, delta_cf2x, delta_n, delta_t, delta_pts = (
                np.zeros(self.numDimensions),
                np.zeros(self.numDimensions),
                0.0,
                0,
                0,
            )
            # with open('blah.txt','a+') as f:
            #         f.write(f"type = {type(x)}\t")
            #         f.write(f"x = {x}\n")
            # for each point in order apply decay and compute the delta "effect" to mc
            for row in x:
                arrivalT, featVals, lbl = row[0], row[1], row[2]
                assert type(featVals) == type(delta_cf1x) == type(delta_cf1x)
                lmbda = math.pow(2, -self.lmbda * (arrivalT - delta_t))
                delta_cf1x = (delta_cf1x * lmbda) + featVals
                delta_cf2x = (delta_cf1x * lmbda) + featVals * featVals
                delta_n = (delta_n * lmbda) + 1
                delta_t = max(arrivalT, delta_t)
                # get total pts
                delta_pts = delta_pts + 1
                lbl_counts[lbl] = lbl_counts[lbl] + 1
            # TODO: get label -> do this later mc.getLabel()

            # TODO: label = max(sum(cl1), ... , sum(cln))
            # so I need to know the number of real clusters (n) and create a new counter for each
            # each time we calcDelta
            # TODO: We can get this if we pass label along with the training data?
            # TODO: get correctPts

            return delta_cf1x, delta_cf2x, delta_n, delta_t, delta_pts, lbl_counts

        return sortedRDD.mapValues(lambda x: calcDelta(x)).collect()

    def updateMicroClusters(self, assignations):
        """
        Algorithm 1 modified. (Cao et al. and Xu et al.)

        Go over the current batch of points and :
        1) Local Update Step:
            - Assign batch points to closest primary micro cluster
            - Compute deltas (effect of each point with decay) to the respective primary micro cluster
            - Assign rest of batch points to closest outlier micro cluster
            - Compute deltas (effect of each point with decay) to the respective outlier micro cluster
        2) Global Update Step:
            - Apply aggregate delta effects to primary micro clusters
            - Apply aggregate delta effects to outlier micro clusters
            - Create new outlier micro clusters from points not assigned to any cluster (pmc or omc)
            - Upgrade outlier micro clusters to primary micro clusters if .weight() > beta*mu
        """
        ##### ---- Global Update Step
        # print("In updateMicroClusters")
        # all are RDD[(Int, (<key>Long, Vector[<features(Double)>]))]:
        # dataInPmic = None
        # dataInAndOut = None
        # dataOut = None
        # dataInOmic = None
        # outliers = None

        # why persist: https://stackoverflow.com/questions/31002975/how-to-use-rdd-persist-and-cache
        assignations.persist()

        # aa = assignations.collect()
        # print(f"{len(aa)} assignations: {aa}")

        # # for printing:
        # to_be_printed = assignations.collect()
        # for row in to_be_printed:
        #     print(f"assignations: {str(row)}")

        # print("updateMicroClusters: Step 1")
        # Step 1: filter out the points that were not assigned to any microcluster
        dataInPmic = assignations.filter(lambda x: x[0] != -1)
        # print(f"dataInPmic: {dataInPmic.collect()}")
        # to_be_printed = dataInPmic.collect()
        # for row in to_be_printed:
        #     print(f"dataInPmic: {str(row)}")
        # works ok until here

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
        sortedRDD = dataInPmic.groupByKey().mapValues(
            lambda x: sorted(list(x), key=lambda y: y[0])
        )
        # to_be_printed = sortedRDD.collect()
        # for row in to_be_printed:
        #     print(f"sortedRDD:  {str(row)}")

        # Step 3:
        dataInPmicSS = self.computeDelta(sortedRDD)
        # print(f"dataInPmicSS: {dataInPmicSS}")

        # datapoints are data not assigned to pmic that might belong to outlier mc or are noise
        dataInAndOut = assignations.filter(lambda x: x[0] == -1).map(lambda x: x[1])
        # print(f"dataInAndOut: {dataInAndOut.collect()}")

        # dataOut: data not assigned to primary microclusters that have not been assigned to outlier microcluster
        # or not assigned to any cluster -> index == -1
        dataOut = self.assignToOutlierCluster(dataInAndOut)
        # data in outlier microclusters
        # print(f"dataOut = {dataOut.collect()}")
        dataOut.persist()

        dataInOmic = dataOut.filter(lambda x: x[0] != -1)
        # outliers : data not assigned to outlier microclusters -> complete outliers
        outliers = dataOut.filter(lambda x: x[0] == -1).map(lambda x: x[1])
        # group by omc id and sort by arrivel and compute delta
        omicSortedRDD = dataInOmic.groupByKey().mapValues(
            lambda x: sorted(list(x), key=lambda y: y[0])
        )
        dataInOmicSS = self.computeDelta(omicSortedRDD)

        realOutliers = outliers.collect()

        # print(f"realOutliers = {realOutliers}")
        assignations.unpersist()
        dataOut.unpersist()
        # TODO: What is this
        DriverTime = time.time()

        ##### ---- Global Update Step
        # print("\t----Global Update Step:----")
        # print(f"dataInPmicSS= {dataInPmicSS}")
        # TODO: Test exactly how deltas are applied, need to better understand effect.
        if len(dataInPmicSS) > 0:
            for ss in dataInPmicSS:
                i, delta_cf1x, delta_cf2x, n, ts, delta_pts, lbl_counts = (
                    ss[0],
                    ss[1][0],
                    ss[1][1],
                    ss[1][2],
                    ss[1][3],
                    ss[1][4],
                    ss[1][5],
                )
                # print(f"{i}) delta_pts = {delta_pts}", end='')
                # print(f"\t lbl_counts = {lbl_counts}")
                # print(f"i = {i}, time = {ts}")
                # the max(ts) out of the microclusters becomes the self.lastEdit
                if self.lastEdit < ts:
                    self.lastEdit = ts
                # TODO: Call mc.calcLabel() -> calculates correctPts
                # TODO: See if we can do this with .insert()
                # TODO: Should we be manipulating the broadcastedObjects? or are we in driver node
                #       so we can manipulate pMicroClusters?
                self.pMicroClusters[i].setWeight(n, ts)
                self.pMicroClusters[i].cf1x = self.pMicroClusters[i].cf1x + delta_cf1x
                self.pMicroClusters[i].cf2x = self.pMicroClusters[i].cf2x + delta_cf2x
                # overwrite the past lbl_counts, delta_pts with only the current batch ones
                # its a hassle to try and have window over previous ones + decay
                # TODO: look into windowed option later ( same with outliers )
                self.pMicroClusters[i].lbl_counts = lbl_counts
                self.pMicroClusters[i].pts = delta_pts

        # TODO: TEST
        upgradeToPMIC = []
        if len(dataInOmicSS) > 0:
            # print(f"Number of updated o-micro-clusters = {len(dataInOmicSS)}")
            # detectList = []
            for oo in dataInOmicSS:
                i, delta_cf1x, delta_cf2x, n, ts, delta_pts, lbl_counts = (
                    oo[0],
                    oo[1][0],
                    oo[1][1],
                    oo[1][2],
                    oo[1][3],
                    oo[1][4],
                    oo[1][5],
                )
                # detectList.insert(0, i)
                if self.lastEdit < ts:
                    self.lastEdit = ts
                # TODO: See if we can do this with .insert()
                self.oMicroClusters[i].setWeight(n, ts)
                self.oMicroClusters[i].cf1x = self.oMicroClusters[i].cf1x + delta_cf1x
                self.oMicroClusters[i].cf2x = self.oMicroClusters[i].cf2x + delta_cf2x
                self.oMicroClusters[i].lbl_counts = lbl_counts
                self.oMicroClusters[i].pts = delta_pts
                # print("\nfor o-mc")
                # print(f"self.beta * self.mu = {self.beta} * {self.mu} = {self.beta * self.mu}")
                # print(f"w = {w}")
                if self.oMicroClusters[i].weight >= self.beta * self.mu:
                    upgradeToPMIC.append(i)

        # TODO: TEST CODE
        # TODO: Handle .lbl_counts etc. in code below where we create new microcluster
        # tyoe(realOutliers) == list()
        if len(realOutliers) > 0:
            # print(f"Number of realOutliers =  {len(realOutliers)}")
            # print(f"realOutliers =  {realOutliers}")

            # TODO: Is this needed?
            # if len(realOutliers) > 35_000:
            #     realOutliers = realOutliers.sortByKey().collect()
            if len(realOutliers) < 50_000:
                realOutliers = sorted(realOutliers, key=lambda x: x[0])
            if self.lastEdit < realOutliers[len(realOutliers) - 1][0]:
                self.lastEdit = realOutliers[len(realOutliers) - 1][0]
            # newMCs -> keep track of newly created micro clusters
            j, newMCs = 0, []
            for point in realOutliers:
                ts, point_vals, point_labl = point[0], point[1], point[2]
                # print(f"ts {ts},point_vals {point_vals}, label {point_labl}")
                minDist, idMinDist, merged = float("inf"), 0, 0
                # TODO: What is recursiveOutliersRMSDCheck -> redundant?
                # TODO: Check if we can create oMicroClusters in InitDBSCAn rather than at this point (maybe start out with some oMicroClusters....)
                if (
                    len(self.oMicroClusters) > 0
                    and self.recursiveOutliersRMSDCheck == 1
                ):
                    # if we created a newMC on a previous point of the realOutliers (try to insert)
                    if len(newMCs) > 0:
                        for i in newMCs:
                            dist = np.linalg.norm(
                                self.oMicroClusters[i].getCentroid() - point[1]
                            )
                            if dist < minDist:
                                minDist, idMinDist = dist, i
                    # print(f"(realOloop) oMicroClusters == {self.oMicroClusters}")
                    # print(f"(realOloop) oMicroClusters[{idMinDist}] == {self.oMicroClusters[idMinDist]}")
                    ocopy = copy.deepcopy(self.oMicroClusters[idMinDist])
                    # TODO: Test correctness
                    ocopy.insertAtT(point=point_vals, time=ts, n=1, label=point_labl)
                    if ocopy.getRMSD() <= self.epsilon:
                        # TODO: create lbl
                        self.oMicroClusters[idMinDist].insert(
                            point, 1, label=point_labl
                        )
                        merged = 1
                        j += 1
                # Creation of outlier micro clusters
                if merged == 0:
                    # TODO: Fix
                    newOmic = CoreMicroCluster(
                        cf2x=point_vals * point_vals,
                        cf1x=point_vals,
                        weight=1.0,
                        t0=ts,
                        lastEdit=ts,
                        lmbda=self.lmbda,
                        num_labels=self.NUM_LABELS,
                        label=point_labl,
                    )
                    self.oMicroClusters.append(newOmic)
                    newMCs.append(len(self.oMicroClusters) - 1)
            # print(f"The number of newly generated microclusters: {len(newMCs)}")
            # print(f"Merged outliers: {j}")
            if self.recursiveOutliersRMSDCheck == 1:
                for k in newMCs:
                    # print("\nfor o-mc new")
                    # print(f"self.beta * self.mu = {self.beta} * {self.mu} = {self.beta * self.mu}")
                    # print(f"w = {w}")
                    if self.oMicroClusters[k].weight >= self.beta * self.mu:
                        upgradeToPMIC.append(k)
                        # print(f"upgradeToPMIC = {upgradeToPMIC}")

        # upgradeToPMIC..
        if len(upgradeToPMIC) > 0:
            # print(f"upgradeToPMIC = {upgradeToPMIC}")
            # we need descending order delete because del changes index
            for r in sorted(upgradeToPMIC, reverse=True):
                # print(f"r= {r} - deleting self.oMicroClusters[{r}] = {self.oMicroClusters[r]} from {self.oMicroClusters}")
                # print("\nupgrading..")
                # print(f"upgradeToPMIC = {upgradeToPMIC}")

                self.pMicroClusters.append(self.oMicroClusters[r])
                # print(f"self.pMicroClusters = {self.pMicroClusters}")
                del self.oMicroClusters[r]
                # print(f"self.oMicroClusters = {self.oMicroClusters}")
                # self.oMicroClusters -= self.oMicroClusters[r]

        DriverTime = time.time() - DriverTime
        self.AlldriverTime += DriverTime
        # print(f" Driver completed... driver time taken (ms) = {DriverTime}")

    def ModelDetect(self):
        """
        Algorithm 2 -> if t%Tp DenStream (Cat et al.)
        """
        # print(f"Time for model microcluster detection: {self.lastEdit}")
        to_be_deleted = []
        if len(self.pMicroClusters) > 0:
            for idx, mc in enumerate(self.pMicroClusters):
                # print(f"\n for pmic {idx} to_be_deleted")
                w = mc.getWeightAtT(self.lastEdit)
                # print(f"self.beta * self.mu = {self.beta} * {self.mu} = {self.beta * self.mu}")
                # print(f"w = {w}")
                if w < self.beta * self.mu:
                    # print(f"pmic {idx} will be deleted")
                    to_be_deleted.append(idx)
        # print(f"Number of P microclusters to be deleted: {len(to_be_deleted)}")
        if len(to_be_deleted) > 0:
            # we need descending order delete because del changes index
            for i in sorted(to_be_deleted, reverse=True):
                # print(f"i = {i}, to_be_deleted = {to_be_deleted}")
                # print(f"pMicroClusters = {self.pMicroClusters}")
                # print(f"len(pmc) = {len(self.pMicroClusters)}")
                del self.pMicroClusters[i]

        # Time for outlier microcluster deletion (slightly different)
        to_be_deleted = []
        if len(self.oMicroClusters) > 0:
            for idx, mc in enumerate(self.oMicroClusters):
                nomin = math.pow(2, -self.lmbda * (self.lastEdit - mc.t0 + self.Tp))
                denom = math.pow(2, -self.lmbda * self.Tp) - 1
                uthres = nomin / denom
                if mc.getWeightAtT(self.lastEdit) < uthres:
                    to_be_deleted.append(idx)

        # print(f"Number of O microclusters to be deleted: {len(to_be_deleted)}")
        if len(to_be_deleted) > 0:
            # we need descending order delete because del changes index
            for i in sorted(to_be_deleted, reverse=True):
                del self.oMicroClusters[i]

    def FinalDetect(self):
        # TODO: What is this?
        self.time = self.time - self.batchTime
        self.ModelDetect()
