from microcluster import CoreMicroCluster
from utils import *

# general
from collections import defaultdict
import time, math, copy
from pprint import pprint
import numpy as np


class DDStreamModel:
    def __init__(
        self,
        numDimensions,
        batchTime,
        epsilon,
        beta,
        mu,
        lmbda,
        num_labels,
        Tp=4,
    ):
        self.numDimensions = numDimensions
        self.batchTime = batchTime
        self.eps = 0.01
        self.NUM_LABELS = num_labels

        self.time = 0
        self.N = 0
        self.currentN = 0
        self.batch_id = None

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

        self.AlldriverTime = 0.0
        self.AlldetectTime = 0.0
        self.AllprocessTime = 0.0

    def getMicroClusters(self):
        print(f"\tin getMicroClusters")
        print(f"self.pMicroClusters = {self.pMicroClusters}\n")
        print(f"self.broadcastPMic = {self.broadcastPMic.value}\n")

        return self.pMicroClusters

    def initDBSCAN(self, ssc, initialEpsilon, path):
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
        num_of_datapts, num_of_dimensions = i + 1, len(tmp) - 1
        assert num_of_dimensions == self.numDimensions
        self.initArr = self.initArr.reshape((-1, num_of_dimensions))

        print(f"Number of dimensions: {num_of_dimensions}")
        print(f"Number of initialization data points: {num_of_datapts}")
        assert num_of_datapts == len(self.initArr)  # float64

        self.tag = [0] * num_of_datapts  # 1 added, 0 not added to core micro cluster
        for i in range(num_of_datapts):
            neighborHoodList = self.getNeighborHood(i, initialEpsilon)
            if len(neighborHoodList) > self.mu:
                self.tag[0] = 1
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
                self.expandCluster(newMC, neighborHoodList, initialEpsilon)
                self.pMicroClusters.append(newMC)

        for mc in self.pMicroClusters:
            mc.pts = 0

        self.broadcastPMic = ssc.sparkContext.broadcast(
            list(zip(self.pMicroClusters, range(len(self.pMicroClusters))))
        )
        self.broadcastOMic = ssc.sparkContext.broadcast(
            list(zip(self.oMicroClusters, range(len(self.oMicroClusters))))
        )
        print(
            f"broadcastPMic (-1) = {self.broadcastPMic} \n {self.broadcastPMic.value}"
        )

        print(f"number of pMicroClusters = {len(self.pMicroClusters)}")
        not_in_mc = len(list(filter(lambda x: x == 0, self.tag)))
        print(f"Points not added to MicroClusters = {not_in_mc}")

        self.initialized = True
        print("END of Initialization\n")

    def getNeighborHood(self, pos, epsilon):
        """
        Get the indices of the points in the neighborhood of the point = self.initArr[pos]

        :param epsilon   = DBSCAN parameter (minimum radius of points in neighborhood)
        :param pos       = index of point we wish to calculate the neighborhood of (point = self.initArr[pos])
        :output idBuffer = list of indices (i) of points in self.initArr[] within the neighborhood of point = self.initArr[pos]
        """
        idBuffer = []
        total_dist, pts = 0, 0
        for i in range(len(self.initArr)):
            if i != pos and self.tag[i] != 1:
                pts += 1
                # print("-> YES")
                # Euclidean distance : https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
                # print("Calculate distance: ")
                # print(f"A:self.initArr[pos] - self.initArr[i] = {self.initArr[pos]} - {self.initArr[i]} = {self.initArr[pos] - self.initArr[i]}")
                dist = np.linalg.norm(self.initArr[pos] - self.initArr[i])
                total_dist += dist
                # print(f"{i}) dist: np.linalg.norm(A) = {dist} < epsilon = {epsilon}", end="")
                # print(f"\tAVG distance = {total_dist/pts}")
                if dist < epsilon:
                    # print("-> 2 YES")
                    # add the point to the neighborhood of the initArr[pos] point
                    # print(f"idBuffer = {idBuffer}")
                    idBuffer.append(i)
                    # print(f"(new) idBuffer = {idBuffer}")
        return idBuffer

    def expandCluster(self, newMC, neighborHoodList, initialEpsilon):
        """
        Recursively add neighborhood points to micro cluster.

        :param newMC            = CoreMicroCluster to be expanded
        :param neighborHoodList = List of neighbor indices (for self.initArr) to be added to the CoreMicroCluster
        """
        for neighbor in neighborHoodList:
            self.tag[neighbor] = 1
        for neighbor in neighborHoodList:
            newMC.insertAtT(
                point=self.initArr[neighbor],
                time=0,
                n=1,
                label=self.initLabels[neighbor],
            )
            neighborHoodList2 = self.getNeighborHood(neighbor, initialEpsilon)
            if len(neighborHoodList2) > self.mu:
                self.expandCluster(newMC, neighborHoodList2, initialEpsilon)

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

        """Run .foreachBatch()"""
        self.batch_id = batch_id
        print(f"\nBATCH: {batch_id}", streaming_df, end="\n")
        points_in_batch = streaming_df.count()
        print(f"For batch_id:{batch_id}, rows in dataframe: {points_in_batch}\n")
        rdd = streaming_df.rdd.map(tuple)
        if not rdd.isEmpty() and self.initialized:
            batch_start_t = time.time()
            print(f"The time is now: {self.lastEdit}")
            print()
            assignations = self.assignToMicroCluster(rdd)

            self.updateMicroClusters(assignations)
            self.broadcastPMic = rdd.context.broadcast(
                list(zip(self.pMicroClusters, range(len(self.pMicroClusters))))
            )

            self.broadcastOMic = rdd.context.broadcast(
                list(zip(self.oMicroClusters, range(len(self.oMicroClusters))))
            )
            detectTime = time.time()
            if self.modelDetectPeriod != 0 and self.modelDetectPeriod % 4 == 0:
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

            pmic_avg_purity = self.calcAvgPurity(self.broadcastPMic.value)

            total_pts = 0
            # save primary
            for i, mc in enumerate(self.broadcastPMic.value):
                microcl, _ = mc[0], mc[1]
                total_pts += microcl.pts
                append_to_MICRO_CLUSTERS(
                    batch_id=self.batch_id,
                    mctype="primary",
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
                    purity=microcl.calcPurity(),
                    points_in_batch=points_in_batch,
                    radius=microcl.getRMSD(),
                )
            # save outliers
            for i, mc in enumerate(self.broadcastOMic.value):
                microcl, _ = mc[0], mc[1]
                total_pts += microcl.pts
                append_to_MICRO_CLUSTERS(
                    batch_id=self.batch_id,
                    mctype="outlier",
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
                    purity=microcl.calcPurity(),
                    points_in_batch=points_in_batch,
                    radius=microcl.getRMSD(),
                )
            # append avg purity from above microclusters in current batch
            avg_purity = self.calcAvgPurity(self.broadcastPMic.value)
            append_to_MICRO_METRICS(
                batch_id=self.batch_id, name="PURITY(avg)", value=avg_purity
            )

    def assignToMicroCluster(self, rdd):
        """
        Assign each point in the batch to a pMicroCluster.
        :param rdd       : (key, <features>)
        :return rdd      : (minIndex, (key, <features>))
        :return minIndex : index of closest microcluster ( -1 indicates no assignment)
        """

        def assign(a):
            """:param a : point/row in batch format=(None, DenseVector([<features>]))"""
            a = a[0], a[1].toArray(), a[2]

            minDist, minIndex = float("inf"), -1
            tmp = []
            if len(self.broadcastPMic.value) > 0:
                for mc in self.broadcastPMic.value:
                    dist = np.linalg.norm(a[1] - mc[0].getCentroid())  # ** 2
                    tmp.append(round(dist, 1))
                    if dist < minDist:
                        minDist, minIndex = dist, mc[1]

                pcopy = copy.deepcopy(self.broadcastPMic.value[minIndex][0])
                n = 0
                pcopy.insert(a, 1)
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
            minIndex = -1
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

        def calcDelta(x):
            """
            :param x: list of points :(<timestamp>, list<feature values>)
            :output (delta_cf1x, delta_cf2x, delta_n, delta_t)
            """
            lbl_counts = [0] * self.NUM_LABELS

            delta_cf1x, delta_cf2x, delta_n, delta_t, delta_pts = (
                np.zeros(self.numDimensions),
                np.zeros(self.numDimensions),
                0.0,
                0,
                0,
            )
            for row in x:
                arrivalT, featVals, lbl = row[0], row[1], row[2]
                assert type(featVals) == type(delta_cf1x) == type(delta_cf1x)
                lmbda = math.pow(2, -self.lmbda * (arrivalT - delta_t))
                delta_cf1x = (delta_cf1x * lmbda) + featVals
                delta_cf2x = (delta_cf2x * lmbda) + featVals * featVals
                delta_n = (delta_n * lmbda) + 1
                delta_t = max(arrivalT, delta_t)
                delta_pts = delta_pts + 1

                print(f"\nHERE len() = {len(lbl_counts)}")
                print(f"lbl_counts[lbl] = lbl_counts[{lbl}] = {lbl_counts[lbl]}\n")
                lbl_counts[lbl] = lbl_counts[lbl] + 1

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

        assignations.persist()

        dataInPmic = assignations.filter(lambda x: x[0] != -1)

        sortedRDD = dataInPmic.groupByKey().mapValues(
            lambda x: sorted(list(x), key=lambda y: y[0])
        )

        dataInPmicSS = self.computeDelta(sortedRDD)

        dataInAndOut = assignations.filter(lambda x: x[0] == -1).map(lambda x: x[1])

        dataOut = self.assignToOutlierCluster(dataInAndOut)

        dataOut.persist()

        dataInOmic = dataOut.filter(lambda x: x[0] != -1)
        outliers = dataOut.filter(lambda x: x[0] == -1).map(lambda x: x[1])
        omicSortedRDD = dataInOmic.groupByKey().mapValues(
            lambda x: sorted(list(x), key=lambda y: y[0])
        )
        dataInOmicSS = self.computeDelta(omicSortedRDD)

        realOutliers = outliers.collect()

        assignations.unpersist()
        dataOut.unpersist()
        DriverTime = time.time()

        for mc in self.pMicroClusters:
            mc.pts = 0
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
                if self.lastEdit < ts:
                    self.lastEdit = ts
                self.pMicroClusters[i].setWeight(n, ts)
                self.pMicroClusters[i].cf1x = self.pMicroClusters[i].cf1x + delta_cf1x
                self.pMicroClusters[i].cf2x = self.pMicroClusters[i].cf2x + delta_cf2x
                self.pMicroClusters[i].lbl_counts = lbl_counts
                self.pMicroClusters[i].pts = delta_pts

        for mc in self.oMicroClusters:
            mc.pts = 0
        upgradeToPMIC = []
        if len(dataInOmicSS) > 0:
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
                if self.lastEdit < ts:
                    self.lastEdit = ts
                self.oMicroClusters[i].setWeight(n, ts)
                self.oMicroClusters[i].cf1x = self.oMicroClusters[i].cf1x + delta_cf1x
                self.oMicroClusters[i].cf2x = self.oMicroClusters[i].cf2x + delta_cf2x
                self.oMicroClusters[i].lbl_counts = lbl_counts
                self.oMicroClusters[i].pts = delta_pts
                if self.oMicroClusters[i].weight >= self.beta * self.mu:
                    upgradeToPMIC.append(i)

        if len(realOutliers) > 0:
            if len(realOutliers) < 50_000:
                realOutliers = sorted(realOutliers, key=lambda x: x[0])
            if self.lastEdit < realOutliers[len(realOutliers) - 1][0]:
                self.lastEdit = realOutliers[len(realOutliers) - 1][0]
            j, newMCs = 0, []
            for point in realOutliers:
                ts, point_vals, point_labl = point[0], point[1], point[2]
                minDist, idMinDist, merged = float("inf"), 0, 0
                if (
                    len(self.oMicroClusters) > 0
                    and self.recursiveOutliersRMSDCheck == 1
                ):
                    if len(newMCs) > 0:
                        for i in newMCs:
                            dist = np.linalg.norm(
                                self.oMicroClusters[i].getCentroid() - point[1]
                            )
                            if dist < minDist:
                                minDist, idMinDist = dist, i
                    ocopy = copy.deepcopy(self.oMicroClusters[idMinDist])
                    ocopy.insertAtT(point=point_vals, time=ts, n=1, label=point_labl)
                    if ocopy.getRMSD() <= self.epsilon:
                        self.oMicroClusters[idMinDist].insert(
                            point, 1, label=point_labl
                        )
                        mc = self.oMicroClusters[idMinDist]
                        merged = 1
                        j += 1
                if merged == 0:
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
            if self.recursiveOutliersRMSDCheck == 1:
                for k in newMCs:
                    if self.oMicroClusters[k].weight >= self.beta * self.mu:
                        upgradeToPMIC.append(k)

        if len(upgradeToPMIC) > 0:
            for r in sorted(upgradeToPMIC, reverse=True):

                self.pMicroClusters.append(self.oMicroClusters[r])
                del self.oMicroClusters[r]

        DriverTime = time.time() - DriverTime
        self.AlldriverTime += DriverTime

    def ModelDetect(self):
        """
        Algorithm 2 -> if t%Tp DenStream (Cat et al.)
        """
        to_be_deleted = []
        if len(self.pMicroClusters) > 0:
            for idx, mc in enumerate(self.pMicroClusters):
                w = mc.getWeightAtT(self.lastEdit)
                if w < self.beta * self.mu:
                    to_be_deleted.append(idx)
        if len(to_be_deleted) > 0:
            for i in sorted(to_be_deleted, reverse=True):
                del self.pMicroClusters[i]

        to_be_deleted = []
        if len(self.oMicroClusters) > 0:
            for idx, mc in enumerate(self.oMicroClusters):
                nomin = math.pow(2, -self.lmbda * (self.lastEdit - mc.t0 + self.Tp))
                denom = math.pow(2, -self.lmbda * self.Tp) - 1
                uthres = nomin / denom
                if mc.getWeightAtT(self.lastEdit) < uthres:
                    to_be_deleted.append(idx)

        if len(to_be_deleted) > 0:
            for i in sorted(to_be_deleted, reverse=True):
                del self.oMicroClusters[i]

    def FinalDetect(self):
        self.time = self.time - self.batchTime
        self.ModelDetect()
