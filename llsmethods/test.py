import numpy as np
from llsmethods import cg
import matplotlib.pyplot as plt
from math import cos, exp

n = 4 # number of nodes
h = 1/(n+2)
afun = lambda x, y: 1
rhsfun = lambda x, y: 2*x*(1-x) + 2*y*(1-y)

x = np.arange(0, 1, h)
a = np.array([[afun(x[i], x[j]) for j in range(n+2)] for i in range(n+2)])

uinit = np.zeros((n**2, 1))

def A(u, a, n):
    """Elliptic Operator on u vector"""

    V = np.zeros((n, n))
    utemp = u.reshape((n, n))
    u = np.zeros((n+2, n+2))
    u[1:n+1, 1:n+1] = utemp

    for i in range(n):
        for j in range(n):
            V[i, j] += (a[i, j] + a[i+1, j])*(u[i+1, j] - u[i, j])
            V[i, j] -= (a[i-1, j] + a[i, j])*(u[i, j] - u[i-1, j])
            V[i, j] += (a[i, j+1] + a[i, j])*(u[i, j+1] - u[i, j])
            V[i, j] -= (a[i, j] + a[i, j-1])*(u[i, j] - u[i, j-1])

    return V.reshape((n**2, 1))


b = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        b[i, j] = rhsfun(x[i], x[j])

b = b.reshape((n**2, 1))

y, res = cg(lambda u: A(u, a, n), b)

res = np.array(res)

print(y)

#plt.plot(range(len(res)), np.sqrt(res))
#plt.show()



