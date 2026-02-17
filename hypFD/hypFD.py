import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from math import sin, pi
import pdb

class AbstractHyperbolicMethod:
    """
    Class for an abstract method for the periodic hyperbolic equation, where
    update step is specified by class inheritance.
    """
    def __init__(self, a, eta, ncel, tau, xint=[0, 1], tint=[0, 10]):

        ## Initialize Passed variables
        self.a = a
        self.eta = eta
        self.ncel = ncel
        self.tau = tau
        self.left = xint[0]
        self.right = xint[1]
        self.t0 = tint[0]
        self.tfinal = tint[1]

        ## Define calculated variables
        self.h = (self.right - self.left)/self.ncel
        self.xpoints = np.arange(self.left, self.right + self.h, self.h)
        self.Uinit = self.eta(self.xpoints)
        self.U = [self.Uinit]
        self.times = [self.t0]

    def method(self, t):
        """
        Discretization method to be defined by user
        """
        pass

    def genHyp(self):
        t = self.t0

        Running = True
        while Running:
            t += self.tau
            if t + self.tau > self.tfinal:
                self.tau = self.tfinal - t
                Running = False
            self.times.append(t)

            self.U.append(self.method(t))

        return (self.U, self.times, self.xpoints)

class MOLFE(AbstractHyperbolicMethod):
    def __init__(self, a, eta, ncel, tau, xint=[0, 1], tint=[0, 5]):
        self.xint = xint
        self.tint = tint
        super().__init__(a, eta, ncel, tau)
    
    def method(self, t):
        newU = np.zeros(self.ncel+1)
        mult = -((self.tau*self.a)/(2*self.h))
        newU[0] = self.U[-1][0] + mult*(self.U[-1][1] - self.U[-1][self.ncel])
        newU[self.ncel] = self.U[-1][self.ncel] + mult*(self.U[-1][0] - self.U[-1][self.ncel-1])
        for j in range(1,self.ncel):
            newU[j] = self.U[-1][j] + mult*(self.U[-1][j+1] - self.U[-1][j-1])

        return newU


if __name__ == "__main__":
    eta = lambda x: np.sin(2*pi*x)
    Ex = MOLFE(0.5, eta, 5, 0.01)
    Usol, times, xpoints = Ex.genHyp()

    plt.plot(xpoints, Usol[2])
    plt.show()

    print(Usol[1])



