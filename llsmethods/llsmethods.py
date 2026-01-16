import numpy as np
import matplotlib.pyplot as plt
from Typing import List
from math import sqrt

def cg(A, b:np.array, eps = 10 ** (-4), x:np.array = None, kmax = 100)-> (np.array, List):
    """
    Method for solving system Ax = b by conjugate gradient method without preconditioning. 
    inputs:
        A - Function for matrix vector multiplication 
            #TODO - if enter matrix, construct function by default 
        b - RHS vector
        eps - error tolerance (10^-4 by default)
        x - Initial guess for solution (zero vector by default)
        kmax - maximum number of iterations (100 by default)

    outputs:
        x - numpy array of solution vector
        res - numpy array of residual history
    """

    m, _ = A.shape()

    # initialize undetermined initial guess to zero vector
    if x == None:
        x = np.zeros(m)

    res = []    # list of squared residuals
    r = b - A(x)    # residual
    res.append(np.linalg.norm(r)**2)
    k = 1   # number of iterations

    while True:
        if sqrt(res[k-1]) <= eps * np.linalg.norm(b) or k == kmax:
            break   # terminate iteration at these conditions

        if k == 1:
            p = r   # initial search direction
        else:
            beta = res[k-1]/res[k-2]    # compute beta
            p = r + beta * p    # beta search direction

        w = A(p)
        alpha = res[k-1]/(np.transpose(p)@w)    # optimal value

        #   update vector and residual
        x = x + alpha * p
        r = r - alpha * w

        res.append(np.linalg.norm(r)**2)    # append squared residual norm
        k = k + 1

    return (x, res)
