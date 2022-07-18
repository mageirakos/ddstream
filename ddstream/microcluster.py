import math
from typing import List

# TODO: Possible redundancies/stuff we don't need
# TODO: tfactor also might be redudant
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
        tfactor=1.0,
    ):
        """
        Initialize CoreMicroCluster. (Definition 3.4 - Cao et al.)

        :param cf1x     = weighted linear sum of the points (numpy.ndarray of length dimensions)
        :param cf2x     = weighted squared sum of the points (numpy.ndarray of length dimensions)
        :param weight   = param to determine threshold of outlier relative to c-micro cluster (w > beta*mu)
        :param t0       =
        :param lastEdit = last time an edit (weight, cf1, cf2 recalculation) happened
        :param lmbda    =
        :param tfactor  =
        """
        self.cf2x = cf2x
        self.cf1x = cf1x
        self.weight = weight
        self.t0 = t0
        self.lastEdit = lastEdit
        self.lmbda = lmbda
        self.tfactor = tfactor

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

    # - self.lastEdit is the last time we took a step (calculated weight,cf1x,cf2x etc.)
    def setWeight(self, n, t):
        """Definition 3.4/Property 3.1 - Cao et al."""
        dt = t - self.lastEdit  # time passed since lastEdit
        self.lastEdit = t
        # TODO: what is n? Is it the number of points merged to the cluster during last dt? (I think yes)
        #                   or is n in (0,1) depending on notmerge/merge
        # temp = self.weight
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

    def setWeightWithoutDecaying(self, n):
        self.weight += n

    def getWeight(self, t):
        if self.lastEdit == t:
            return self.weight
        else:
            self.setWeight(0, t)
            return self.weight

    def getWeight(self):
        return self.weight

    def getCentroid(self):
        """Get center of CoreMicroCluster (Definition 3.3 - Cao et al.)"""
        # TODO: Understand relationship between weight and number of points in microcluster or total_num_pmcs
        if self.weight > 0:
            return self.cf1x / self.weight
        else:
            return self.cf1x

    # TODO: What is tfactor
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
            # TODO: tfactor might be redundant
            return self.tfactor * math.sqrt(maxi)
        else:
            return float("inf")

    def copy(self):
        # TODO: Redandunt can use copy.deepcopy?
        return CoreMicroCluster(
            cf2x=self.cf2x,
            cf1x=self.cf1x,
            weight=self.weight,
            t0=self.t0,
            lastEdit=self.lastEdit,
            lmbda=self.lmbda,
            tfactor=self.tfactor,
        )

    # TODO: Fix this as point is some sort of np array
    # TODO: See where this is used
    # point: ((None, DenseVector(<features>)),1) ?
    # point: (timestamp, list[float])
    def insert(self, point, n):
        print(f"\nIn Insert {point}\n")
        self.setWeight(n, point[0])  # point[0] einai to key/timestamp
        self.cf1x = self.cf1x + point[1]
        self.cf2x = self.cf2x + point[1] * point[1]
        print("type(cf2x,cf1x) after insert: ", type(self.cf2x), type(self.cf1x))
        # self.setCf1x([a + p for a, p in zip(self.cf1x, point[1])])
        # self.setCf2x([a + p * p for a, p in zip(self.cf2x, point[1])])

    # this is used during initDBSCAN
    def insertAtT(self, point, time, n):
        """
        Incremental insert of point into CoreMicroCluster (Property 3.1 Cao et al.)

        :param point = numpy.ndarray
        """
        # call seWeight first which recalculated weight, cf1, cf2 for new 'time'
        self.setWeight(n, time)
        self.cf1x += point
        self.cf2x += point * point

    # TODO : Is this correct merging? Should weight be recalculated? Why max(lastEdit)?
    def mergeWithOtherMC(self, otherMC):
        return CoreMicroCluster(
            self.cf2x + otherMC.cf2x,
            self.cf1x + otherMC.cf1x,
            self.weight + otherMC.weight,
            self.t0,
            max(self.lastEdit, otherMC.lastEdit),
            self.lmbda,
            self.tfactor,
        )
