import numpy as np
from llsmethods import cg
import matplotlib.pyplot as plt
from math import cos, exp

n = 10  # number of nodes
h = 1/(n+2)
afun = lambda x, y: exp(x*y)
rhsfun = lambda x, y: -10 * (y*(1-y)*(y*exp(x*y) - 2) + x*(1-x)*(x*exp(x*y) - 2))

x = np.arange(0, 1, h)
a = np.array([[afun(x[i], x[j]) for j in range(n+2)] for i in range(n+2)])

uinit = np.zeros((n**2, 1))

def A(u, a, n):
    """Elliptic Operator on u vector"""

    V = np.zeros((n**2, 1))

    for i in range(n):
        for j in range(n):
            V[i + n*j] += (a[i, j] + a[i+1, j])*(u[i+1 + n*j] - u[i + n*j])
            V[i + n*j] -= (a[i-1, j] + a[i, j])*(u[i + n*j] - u[i-1 + n*j])
            V[i + n*j] += (a[i, j+1] + a[i, j])*(u[i + n*(j+1)] - u[i + n*j])
            V[i + n*j] -= (a[i, j] + a[i, j-1])*(u[i + n*j] - u[i + n*(j-1)])


b = np.zeros((n**2, 1))

for i in range(n):
    for j in range(n):
        b[i + n*j] = rhsfun(x[i+1], x[j+1])

y, res = cg(lambda u: A(u, a, n), b)

res = np.array(res)

plt.plot(range(len(res)), np.sqrt(res))
plt.show()



