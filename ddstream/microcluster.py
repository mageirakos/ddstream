import math
from typing import tuple, List, Type

# TODO: Possible redundancies ( all setters getters are much simples in python by simply accessing/seting with self.)
# TODO: tfactor also might be redudant
# TODO: Test code
# TODO: change List[float] cf1x, cf2x into numpy arrays and do np.multiply instead of the element wise I'm doing now
class CoreMicroCluster:
    # TODO: add default vals to params
    # TODO: add correct types. eg( cf2x => breeze.linalg.Vector[Double])
    # - nomizw to cf1,cf2 einai list kai kala gia ta diaforetika dimensions tou microcluster OXI gia kathe microcluster
    # - den eimai sigouros omws ara prepei na to katalavw
    def __init__(
        self,
        cf2x: List[float],
        cf1x: List[float],
        weight: float,
        t0: int,
        lastEdit: int,
        lmbda: float,
        tfactor: float = 1.0,
    ):
        """
        Initialize CoreMicroCluster. (Definition 3.4 - Cao et al.)

        :param cf1x     = weighted linear sum of the points
        :param cf2x     = weighted squared sum of the points
        :param weight   = param to determine threshold of outlier relative to c-micro cluster (w > beta*mu)
        :param t0       =
        :param lastEdit =
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

    def calcCf2x(self, dt: int):
        self.cf2x = [x * math.pow(2, -self.lmbda * dt) for x in self.cf2x]

    def setCf2x(self, cf2x):
        self.cf2x = cf2x

    def getCf2x(self) -> List[float]:
        return self.cf2x

    def calcCf1x(self, dt: int):
        self.cf1x = [x * math.pow(2, -self.lmbda * dt) for x in self.cf1x]

    def setCf1x(self, cf1x):
        self.cf1x = cf1x

    def getCf1x(self):
        return self.cf1x

    # TODO: understand what self.lastEdit exactly does
    def setWeight(self, n: float, t: int):
        """ """
        dt = t - self.lastEdit  # time passed since lastEdit
        self.setlastEdit(t)
        # TODO: what is n? Is it the number of points merged to the cluster during last dt?
        #                   or is n in (0,1) depending on notmerge/merge
        self.weight = self.weight * math.pow(2, -self.lmbda * dt) + n
        self.calcCf1x(dt)
        self.calcCf2x(dt)

    def setWeightWithoutDecaying(self, n):
        self.weight += n

    def getWeight(self, t) -> float:
        if self.lastEdit == t:
            return self.weight
        else:
            self.setWeight(0, t)
            return self.weight

    def getWeight(self) -> float:  # second getWeight func
        return self.weight

    def getT0(self) -> int:
        return self.t0

    def setT0(self, t0: int):
        self.t0 = t0

    def getCentroid(self):
        """
        Center (Definition 3.3 - Cao et al.)
        """
        # TODO: Understand relationship between weight and number of points in microcluster or total_num_pmcs
        if self.weight > 0:
            return [a / b for a, b in zip(self.cf1x, self.weight)]
        else:
            return self.cf1x

    # TODO: Understand this function
    # TODO: where is it used?
    # - I think this is §crocluster radius
    # TODO: What is tfactor
    # - it is usually set to 1.0
    def getRMSD(self) -> float:
        """
        Get radius of p-micro cluster (Definition 3.4 - Cao et al.)

        - uses max instead of mean rmsd
        """
        if self.weight > 1:
            sumi, maxi = 0, 0
            for i in range(len(self.cf2x)):
                tmp = self.cf2x[i] / self.weight - (self.cf1x[i] * self.cf1x[i]) / (
                    self.weight * self.weight
                )
                sumi += tmp  # is sumi used anywhere? (sum of factors under the sqrt for euclidean)
                maxi = max(maxi, tmp)
            return self.tfactor * math.sqrt(maxi)
        else:
            return float("inf")

    def getlastEdit(self) -> int:
        return self.lastEdit

    def setlastEdit(self, t: int):
        self.lastEdit = t

    # python has __copy__
    def copy(self) -> Type[CoreMicroCluster]:
        return CoreMicroCluster(
            self.cf2x,
            self.cf1x,
            self.weight,
            self.t0,
            self.lastEdit,
            self.lmbda,
            self.tfactor,
        )

    # point: (timestamp, list[float])
    def insert(self, point: tuple(int, List[float]), n: float):
        # we must first set the weight and then cf1, cf2
        self.setWeight(n, point[0])
        self.setCf1x([a + p for a, p in zip(self.cf1x, point[1])])
        self.setCf2x([a + p * p for a, p in zip(self.cf2x, point[1])])

    # TODO: Figure out if I want point to be tuple with ts and list[float] or just single list[float]
    def insert(self, point: List[float], time: int, n: float):
        self.setWeight(n, time)
        self.setCf1x([a + p for a, p in zip(self.cf1x, point)])
        self.setCf2x([a + p * p for a, p in zip(self.cf2x, point)])

    def mergeWithOtherMC(
        self, otherMC: Type[CoreMicroCluster]
    ) -> Type[CoreMicroCluster]:
        return CoreMicroCluster(
            [a + b for a, b in zip(self.getCf2x, otherMC.getCf2x())],
            [a + b for a, b in zip(self.getCf1x, otherMC.getCf1x())],
            self.getWeight() + otherMC.getWeight(),
            self.t0,
            max(self.lastEdit, otherMC.lastEdit),
            self.lmbda,
            self.tfactor,
        )
