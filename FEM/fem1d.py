import numpy as np
import matplotlib.pyplot as plt

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

def solve_galerkin():
    pass

if __name__ == "__main__":

    nel = 10

    nodes = np.arange(0, 1 + 1/nel, 1/nel)
    print(nodes)
    alphasol = np.random.randn(nel)

    XX = np.linspace(0, 1, 500)

    YY0 = np.zeros(len(XX))
    for i in range(len(XX)):
        YY0[i] = eval_approx(XX[i], alphasol, nodes)

    plt.plot(XX, YY0)
    plt.show()

    



