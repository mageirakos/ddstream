import math
from typing import List

# TODO: Possible redundancies ( all setters getters are much simples in python by simply accessing/seting with self.)
# TODO: tfactor also might be redudant
# TODO: Test code

# SOS TODO:
# TODO: change List[float] cf1x, cf2x into numpy arrays and do np.multiply instead of the element wise I'm doing now
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
        self.lmbda = lmbda  # lmbda != lambda (careful how I name my variables)
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
        self.weight = self.weight * math.pow(2, -self.lmbda * dt) + n
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

    def getWeight(self):  # second getWeight func
        return self.weight

    def getCentroid(self):
        """Get center of CoreMicroCluster (Definition 3.3 - Cao et al.)"""
        # TODO: Understand relationship between weight and number of points in microcluster or total_num_pmcs
        if self.weight > 0:
            return self.cf1x / self.weight
        else:
            return self.cf1x

    # TODO: Understand this function
    # TODO: What is tfactor
    # - it is usually set to 1.0
    # TODO: Change this funciton based on numpyarray cf2x and cf1x
    def getRMSD(self) -> float:
        """Get radius of p-micro cluster (Definition 3.4 - Cao et al.)"""
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
            self.cf2x,
            self.cf1x,
            self.weight,
            self.t0,
            self.lastEdit,
            self.lmbda,
            self.tfactor,
        )

    # TODO: Fix this as point is some sort of np array
    # TODO: See where this is used
    # point: (timestamp, list[float])
    def insert(self, point: tuple((int, List[float])), n: float):
        self.setWeight(n, point[0])
        self.setCf1x([a + p for a, p in zip(self.cf1x, point[1])])
        self.setCf2x([a + p * p for a, p in zip(self.cf2x, point[1])])

    # this is used during initDBSCAN
    def insert(self, point, time, n):
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
