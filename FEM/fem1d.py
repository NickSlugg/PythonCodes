import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def eval_basis(x, m, nodes, type="lagrange", degree=1):
    """
    Evaluates the basis functions for a set of nodes. 
    """
    if type == "lagrange":
        if degree == 1:
            ##  Case for zero
            if m == 0:
                h = nodes[1] - nodes[0]
                if (nodes[0] <= x) and (x <= nodes[1]):
                    return (nodes[1] - x)/h
                else:
                    return 0

            ## Case for right side of interval
            elif m == (len(nodes) - 1):
                h = nodes[m] - nodes[m-1]
                if (nodes[m-1] <= x) and (x <= nodes[m]):
                    return (x - nodes[m-1])/h
                else:
                    return 0

            ## General Case
            else:
                if (nodes[m-1] <= x) and (x <= nodes[m]):
                    return (x - nodes[m-1])/(nodes[m] - nodes[m-1])
                elif (nodes[m] <= x) and (x <= nodes[m+1]):
                    return (nodes[m+1] - x)/(nodes[m+1] - nodes[m])
                else:
                    return 0

def eval_approx(x, alphasol, nodes):
    y = 0
    for i in range(len(alphasol)):
        y += alphasol[i]*eval_basis(x, i, nodes)

    return y

def _2ptGauss(f, a, b):
    return ((b - a)/2)*(f((2/(b-a))*(-sqrt(3)-(a+b)/2)) + f((2/(b-a))*(sqrt(3)-(a+b)/2)))

def solve_galerkin(nodes, p, f):
    
    # Compute stiffness matrix for equation
    # (p(x) u_x)_x = f(x)
    # u(a) = u(b) = 0

    nel = len(nodes) - 1
    M = np.zeros((nel, nel))
    b = np.zeros((nel, 1))

    M[0, 0] = 2*_2ptGauss(p, nodes[0], nodes[1])
    M[0, 1] = -_2ptGauss(p, nodes[0], nodes[1])
    M[nel-1, nel-1] = 2*_2ptGauss(p, nodes[nel-2], nodes[nel-1])
    M[nel-1, nel-2] = -_2ptGauss(p, nodes[nel-2], nodes[nel-1])

    b[0] = _2ptGauss(lambda x: f(x)*(x - nodes[0])/(nodes[1] - nodes[0]), nodes[0], nodes[1])
    b[nel-1] = _2ptGauss(lambda x: f(x)*(nodes[nel-1] - x)/(nodes[nel-1] - nodes[nel-2]), nodes[nel-2], nodes[nel-1])

    for i in range(1, nel-1):
        M[i, i] = _2ptGauss(p, nodes[i-1], nodes[i+1])
        M[i, i-1] = -_2ptGauss(p, nodes[i-1], nodes[i])
        M[i, i+1] = -_2ptGauss(p, nodes[i], nodes[i+1])

        b[i] = _2ptGauss(lambda x : f(x)*(x - nodes[i-1])/(nodes[i] - nodes[i-1]), nodes[i-1], nodes[i]) + _2ptGauss(lambda x : f(x)*(nodes[i+1] - x)/(nodes[i+1] - nodes[i]), nodes[i], nodes[i+1])

    return (np.linalg.solve(M, b), M)

if __name__ == "__main__":

    nel = 5

    nodes = np.arange(0, 1 + 1/nel, 1/nel)

    p = lambda x: 1
    f = lambda x: -2

    alphasol, Amat = solve_galerkin(nodes, p, f)

    XX = np.linspace(0, 1, 500)

    YY0 = np.zeros(len(XX))
    for i in range(len(XX)):
        print(eval_approx(0, alphasol, nodes))
        YY0[i] = eval_approx(XX[i], alphasol, nodes)[0]

    plt.plot(XX, YY0)
    plt.show()

    



