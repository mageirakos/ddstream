import math
from typing import List

# TODO: Possible redundancies/stuff we don't need
# TODO: Test code


class CoreMicroCluster:
    # TODO: add default vals to params
    # TODO: cf2x is element wise multiplication. but on the paper they say weighted SUM of product
    #           - check if it is correct....
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
        self.pts = 1 #pts will keep the num of pts arrived in current batch
        self.num_labels = num_labels
        self.lbl_counts = [0] * self.num_labels
        self.lbl_counts[label] += 1
        self.correctPts = 1
        self.label = self.lbl_counts.index(max(self.lbl_counts))
        self.purity = None

    # TODO: handle
    def calcPurity(self):
        # print(f"in calcPurity for {self}")
        # print(f"lbl(prev) = {self.label}")
        lbl = self.getLabel()
        # print(f"lbl(after) = {lbl}")
        # print(f"lbl_counts = {self.lbl_counts}")
        # print(f"purity(calc) = {self.correctPts} / {sum(self.lbl_counts)}")
        # print(f"purity(prev) = {self.purity}")
        self.purity = self.correctPts / sum(self.lbl_counts)
        # print(f"purity(after) = {self.purity}")
        # print("\n\n")
        return self.purity

    def getLabel(self):
        # print(f"in getLabel")
        # print(f"lbl_counts = {self.lbl_counts}")
        # print(f"correctPts(prev) = {self.correctPts}")
        self.correctPts = max(self.lbl_counts)
        # print(f"correctPts(after) = {self.correctPts}")
        self.label = self.lbl_counts.index(self.correctPts)
        # print(f"label (calculated) = {self.label}")
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
        # print(f"in calcCf1x for {self}")
        delta = math.pow(2, -self.lmbda * dt)
        # print(f"cf1x BEFORE {self.cf1x} delta = {delta}")
        self.cf1x = self.cf1x * delta
        # print(f"cf1x AFTER : {self.cf1x}")

    # - self.lastEdit is the last time we took a step (calculated weight,cf1x,cf2x etc.)
    def setWeight(self, n, t):
        """Definition 3.4/Property 3.1 - Cao et al."""
        dt = t - self.lastEdit  # time passed since lastEdit
        # print(f"in setWeight(n={n}, t={t}\ndt=t-lastEdit={t}-{self.lastEdit}={dt}")
        self.lastEdit = t
        # TODO: what is n? Is it the number of points merged to the cluster during last dt? (I think yes)
        #                   or is n in (0,1) depending on notmerge/merge
        # temp = self.weight
        # print(f"pmic {self}")
        # print(f"\tweight = {self.weight}\n\tcf1x={self.cf1x} cf2x={self.cf2x} center = {self.getCentroid()}")
        self.weight = self.weight * math.pow(2, -self.lmbda * dt) + n
        # # print(f"NEW WEIGHT: {self.weight} MC: {self}\n")
        # with open('blah.txt','a') as f:
        #     f.write(f"in setWeight\n")
        #     f.write(f"n={n}, t={t}\n")
        #     f.write(f"weight before = {temp}\n")
        #     f.write(f"self.weight * math.pow(2, -self.lmbda * dt) + n = {temp} * {math.pow(2, -self.lmbda * dt)} + {n}\n")
        #     f.write(f"lambda={self.lmbda}, dt={dt}, -lmbda*dt={-self.lmbda * dt}, 2**(-lmbda*dt){2**(-self.lmbda * dt)}, math.pow(2,-lmbda*dt)={math.pow(2, -self.lmbda * dt)}, rounded()={round(math.pow(2, -self.lmbda * dt), 2)}\n")
        #     f.write(f"weight after = {self.weight}\n")
        self.calcCf1x(dt)
        self.calcCf2x(dt)
        # print(f"\t(new) weight = {self.weight}\n\tcf1x={self.cf1x} cf2x={self.cf2x} center = {self.getCentroid()}")

    def setWeightWithoutDecaying(self, n):
        self.weight += n

    def getWeightAtT(self, t):
        # print(f"\n\nHERE in getWeightAtT({t})")
        if self.lastEdit == t:
            # print(f"self.lastEdit ({self.lastEdit}) == t ({t}) -> w {self.weight}")
            return self.weight
        else:
            # print(f"self.lastEdit ({self.lastEdit}) != t ({t}) -> ")
            self.setWeight(0, t)
            return self.weight

    def getCentroid(self):
        """Get center of CoreMicroCluster (Definition 3.3 - Cao et al.)"""
        # print(f"in getCentroid for {self}")
        # TODO: Understand relationship between weight and number of points in microcluster or total_num_pmcs
        if self.weight > 0:
            # print(f'self.weight > 0;\t self.cf1x / self.weight = {self.cf1x}/{self.weight}')
            return self.cf1x / self.weight
        else:
            # print(f'self.weight < 0;\t self.cf1x = {self.cf1x}')
            return self.cf1x

    # - it is usually set to 1.0
    def getRMSD(self):
        """Get radius of p-micro cluster (Definition 3.4 - Cao et al.)"""
        # with open('blah.txt','a') as f:
        #     n+=1
        #     f.write(f"In getRMSD\n")
        #     f.write(f"self={self}, weight = {self.weight}\n")
        if self.weight > 1:
            sumi, maxi = 0, 0
            for i in range(len(self.cf2x)):
                tmp = self.cf2x[i] / self.weight - (self.cf1x[i] * self.cf1x[i]) / (
                    self.weight * self.weight
                )
                sumi += tmp  # use sumi for RMSD on mean radius
                maxi = max(maxi, tmp)  # use maxi if for RMSD on max radius
            return math.sqrt(maxi)
        else:
            return float("inf")

    def insert(self, point, n, label=None):
        """
        Incremental insert of point into CoreMicroCluster (Property 3.1 Cao et al.)

        :param point = (timestamp, np.array<features>)
        :param n =
        """
        # print(f"insert - point = {point}, lbl_counts = {self.lbl_counts}")
        ts, point_vals = point[0], point[1]
        self.setWeight(n, ts)
        self.cf1x = self.cf1x + point_vals
        self.cf2x = self.cf2x + point_vals * point_vals
        self.pts += n
        if label != None:
            self.lbl_counts[label] += 1
        self.label = self.getLabel() # must call this for corrPts to be calculated
        # print(f"lbl_counts = {self.lbl_counts}")


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
        self.label = self.getLabel() # must call this for corrPts to be calculated

    # TODO : Is this correct merging? Should weight be recalculated? Why max(lastEdit)?
    # TODO : Not using this anywhere...
    def mergeWithOtherMC(self, otherMC):
        return CoreMicroCluster(
            self.cf2x + otherMC.cf2x,
            self.cf1x + otherMC.cf1x,
            self.weight + otherMC.weight,
            self.t0,
            max(self.lastEdit, otherMC.lastEdit),
            self.lmbda,
        )
