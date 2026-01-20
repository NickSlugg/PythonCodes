import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp

def sourcef(xval):
    return -(2*xval - 2*(1 - 2*xval) + 4*xval*(xval - xval**2))*exp(-xval*xval)

def dg1dsolve(nel, ss, sig0):
    """
    Solves 1D Darcy Flow problem using second order DG methods
    Inputs:
        - nel: number of elements
        - ss: symmetrizing term 
        - penal: penalty term
    """

    #   Define Local Matrices
    Amat = nel*np.array([[0, 0, 0],
                        [0, 4, 0],
                        [0, 0, 16/3]])

    Bmat = nel*np.array([[sig0, 1-sig0 , -2+sig0],
                        [-ss-sig0, -1+ss+sig0, 2-ss-sig0],
                        [2*ss+sig0, 1-2*ss-sig0, -2+2*ss+sig0]])

    Cmat = nel*np.array([[sig0, -1+sig0 , -2+sig0],
                        [ss+sig0, -1+ss+sig0, -2+ss+sig0],
                        [2*ss+sig0, -1+2*ss+sig0, -2+2*ss+sig0]])

    Dmat = nel*np.array([[-sig0, -1+sig0 , 2-sig0],
                        [-ss-sig0, -1+ss+sig0, 2-ss-sig0],
                        [-2*ss-sig0, -1+2*ss+sig0, 2-2*ss-sig0]])

    Emat = nel*np.array([[-sig0, 1-sig0 , 2-sig0],
                        [ss+sig0, -1+ss+sig0, -2+ss+sig0],
                        [-2*ss-sig0, 1-2*ss-sig0, 2-2*ss-sig0]])

    F0mat = nel*np.array([[sig0, 2-sig0 , -4+sig0],
                        [-2*ss-sig0, -2+2*ss+sig0, 4-2*ss-sig0],
                        [4*ss+sig0, 2-4*ss-sig0, -4+4*ss+sig0]])

    FNmat = nel*np.array([[sig0, -2+sig0 , -4+sig0],
                        [2*ss+sig0, -2+2*ss+sig0, -4+2*ss+sig0],
                        [4*ss+sig0, -2+4*ss+sig0, -4+4*ss+sig0]])

    #   Define dimension constants
    locdim = 3
    glodim = nel*locdim

    #   Initialize global matrix and rhs
    Aglobal = np.zeros((glodim, glodim))
    rhsglobal = np.zeros((glodim, 1))

    #   Gauss quadrature weights/points
    wg = [1, 1]
    sg = [-1/sqrt(3), 1/sqrt(3)]

    #   Assemble first block/first three elements of rhs

    for i in range(locdim):
        for j in range(locdim):
            Aglobal[i, j] += Amat[i, j] + F0mat[i, j] + Cmat[i, j]
            je = locdim + j
            Aglobal[i, je] += Dmat[i, j]

    rhsglobal[0] = nel*sig0
    rhsglobal[1] = -nel*sig0 - 2*ss*nel
    rhsglobal[2] = nel*sig0 + 4*nel*ss

    for ig in range(2):
        rhsglobal[0] += wg[ig]*sourcef((sg[ig] + 1)/(2*nel))/(2*nel)
        rhsglobal[1] += wg[ig]*sg[ig]*sourcef((sg[ig] + 1)/(2*nel))/(2*nel)
        rhsglobal[2] += wg[ig]*sg[ig]*sg[ig]*sourcef((sg[ig] + 1)/(2*nel))/(2*nel)

    #   Assemble intermediate blocks

    for i in range(1,nel-1):
        for ii in range(locdim):
            ie = ii + i*locdim
            for jj in range(locdim):
                je = jj + i*locdim
                Aglobal[ie, je] += Amat[ii, jj] + Bmat[ii, jj] + Cmat[ii, jj]
                je = jj + (i+1)*locdim
                Aglobal[ie, je] += Dmat[ii, jj]
                je = jj + (i-1)*locdim
                Aglobal[ie, je] += Emat[ii, jj]

            for ig in range(2):
                rhsglobal[ie] += wg[ig]*(sg[ig]**(ii))*sourcef((sg[ig] + 2*(i-1) + 1)/(2*nel))/(2*nel)

    #   Assemble final blocks

    for i in range(locdim):
        ie = i + (nel-1)*locdim
        for j in range(locdim):
            je = j + (nel-1)*locdim
            Aglobal[ie, je] += Amat[i, j] + FNmat[i, j] + Bmat[i, j]
            je = j + (nel-1)*locdim
            Aglobal[ie, je] += Emat[i, j]

        for ig in range(2):
            rhsglobal[ie] += wg[ig]*(sg[ig]**(i))*sourcef((sg[ig] + 2*(nel-1) + 1)/(2*nel))/(2*nel)

    #   Solve Linear System
    ysol = np.linalg.solve(Aglobal, rhsglobal)

    return (ysol, Aglobal, rhsglobal)

if __name__ == "__main__":

    nel = 100

    ysol, A, rhs = dg1dsolve(nel, -1, 2)

    XX = np.linspace(0, 1, 500)

    def evalbasis(x):
        m = 1
        while x > m/nel:
            m += 1
        return ysol[3*(m-1)] + ysol[3*(m-1) + 1]*(2*nel*(x - (m - 1/2)/nel)) + ysol[3*(m-1)+2]*(2*nel*(x - (m - 1/2)/nel))**2

    YY = np.zeros((len(XX), 1))
    YYEx = np.zeros((len(XX), 1))

    for i in range(XX.size):
        YY[i] = evalbasis(XX[i])
        YYEx[i] = (1-XX[i])*exp(-XX[i]**2)

    plt.plot(XX, YY)
    plt.plot(XX, YYEx)
    plt.show()



