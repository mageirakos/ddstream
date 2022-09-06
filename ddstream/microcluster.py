import math
from typing import List


class CoreMicroCluster:
    def __init__(
        self,
        cf2x,
        cf1x,
        weight,
        t0,
        lastEdit,
        lmbda,
        num_labels,
        label=None,
    ):
        """
        Initialize CoreMicroCluster. (Definition 3.4 - Cao et al.)

        :param cf1x     = weighted linear sum of the points (numpy.ndarray of length dimensions)
        :param cf2x     = weighted squared sum of the points (numpy.ndarray of length dimensions)
        :param weight   = param to determine threshold of outlier relative to c-micro cluster (w > beta*mu)
        :param t0       = timestamp of CoreMicroCluster creation
        :param lastEdit = last time an edit (weight, cf1, cf2 recalculation) happened
        :param lmbda    =
        :param num_labels = number of unique clusters/labels in dataset (defalut 3)
        """
        self.cf2x = cf2x
        self.cf1x = cf1x
        self.weight = weight
        self.t0 = t0
        self.lastEdit = lastEdit
        self.lmbda = lmbda
        # During initialization only 1 point in mc so :
        self.pts = 1  # pts will keep the num of pts arrived in current batch
        self.num_labels = num_labels
        self.lbl_counts = [0] * self.num_labels
        self.lbl_counts[label] += 1
        self.correctPts = 1
        self.label = self.lbl_counts.index(max(self.lbl_counts))
        self.purity = None

    def calcPurity(self):
        lbl = self.getLabel()
        self.purity = self.correctPts / sum(self.lbl_counts)
        return self.purity

    def getLabel(self):
        self.correctPts = max(self.lbl_counts)
        self.label = self.lbl_counts.index(self.correctPts)
        return self.label

    def calcCf2x(self, dt):
        """
        Calculates new cf1x values based on passed dt time.
        Apply delta = 2^(-λ*δt) to every dimension.

        :param dt = time passed since lastEdit ( lastEdit - t where t is current time )
        """
        delta = math.pow(2, -self.lmbda * dt)
        self.cf2x = self.cf2x * delta

    def calcCf1x(self, dt):
        """
        Calculates new cf1x values based on passed dt time.
        Apply delta = 2^(-λ*δt) to every dimension.

        :param dt = time passed since lastEdit ( t - lastEdit; where t is current time )
        """
        delta = math.pow(2, -self.lmbda * dt)
        self.cf1x = self.cf1x * delta

    def setWeight(self, n, t):
        """Definition 3.4/Property 3.1 - Cao et al."""
        dt = t - self.lastEdit  # time passed since lastEdit
        self.lastEdit = t
        self.weight = self.weight * math.pow(2, -self.lmbda * dt) + n
        self.calcCf1x(dt)
        self.calcCf2x(dt)

    def setWeightWithoutDecaying(self, n):
        self.weight += n

    def getWeightAtT(self, t):
        if self.lastEdit == t:
            return self.weight
        else:
            self.setWeight(0, t)
            return self.weight

    def getCentroid(self):
        """Get center of CoreMicroCluster (Definition 3.3 - Cao et al.)"""
        if self.weight > 0:
            return self.cf1x / self.weight
        else:
            return self.cf1x

    def getRMSD(self):
        """Get radius of p-micro cluster (Definition 3.4 - Cao et al.)"""
        if self.weight > 1:
            sumi, maxi = 0, 0
            for i in range(len(self.cf2x)):
                tmp = abs(self.cf2x[i]) / self.weight - abs(
                    (self.cf1x[i] * self.cf1x[i])
                ) / (self.weight * self.weight)
                sumi += tmp
                maxi = max(maxi, tmp)
            return math.sqrt(maxi)
        else:
            return float("inf")

    def insert(self, point, n, label=None):
        """
        Incremental insert of point into CoreMicroCluster (Property 3.1 Cao et al.)

        :param point = (timestamp, np.array<features>)
        :param n =
        """
        ts, point_vals = point[0], point[1]
        self.setWeight(n, ts)
        self.cf1x = self.cf1x + point_vals
        self.cf2x = self.cf2x + point_vals * point_vals
        self.pts += n
        if label != None:
            self.lbl_counts[label] += 1
        self.label = self.getLabel()

    def insertAtT(self, point, time, n, label=None):
        """
        Incremental insert of point at time into CoreMicroCluster (Property 3.1 Cao et al.)

        :param point = numpy.ndarray
        :param time = timestamp of point arrival
        :param n =
        """
        self.setWeight(n, time)
        self.cf1x = self.cf1x + point
        self.cf2x = self.cf2x + point * point
        self.pts += n
        if label != None:
            self.lbl_counts[label] += 1
        self.label = self.getLabel()

    def mergeWithOtherMC(self, otherMC):
        return CoreMicroCluster(
            self.cf2x + otherMC.cf2x,
            self.cf1x + otherMC.cf1x,
            self.weight + otherMC.weight,
            self.t0,
            max(self.lastEdit, otherMC.lastEdit),
            self.lmbda,
        )
