## Functions for Spectral Collocation method in Python

import numpy as np
import matplotlib.pyplot as plt

def chebyshev(s, n):

    if n == 0:
        return 1
    elif n == 1:
        return s
    else:
        return 2*s*chebyshev(s, n-1) - chebyshev(s, n-2)

def chebyshevp(s, n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return 2*(chebyshev(s, n-1) + s*chebyshevp(s, n-1)) - chebyshevp(s, n-2)

def chebyshevpp(s, n):
    if n == 0:
        return 0
    elif n == 1:
        return 0
    else:
        return 2*(chebyshevp(s, n-1) + s*chebyshevpp(s, n-1) + chebyshevp(s, n-1)) - chebyshevpp(s, n-2)

def spectralCollocation(xpts, bcd):
    N = len(xpts)
    A = np.zeros((N, N))
    b = np.zeros((N, 1))
    D = 1
    f = lambda x: 2

    ## assemble first and last row
    for i in range(N):
        A[0, i] = chebyshev(xpts[0], i)
        A[N-1, i] = chebyshev(xpts[-1], i)
    b[0] = bcd[0]
    b[N-1] = bcd[1]

    ## assemble intermediate rows
    for i in range(1, N-1):
        x = xpts[i]
        b[i] = f(x)
        for j in range(N):
            A[i, j] = - chebyshevpp(x, j)

    ## solve linear system
    alpha = np.linalg.solve(A, b)
    print(alpha)

    return alpha

def collocEval(alpha, s):
    y = 0
    N = len(alpha)
    for j in range(N):
        y += alpha[j]*chebyshev(s, j)
    return y

if __name__ == "__main__":

    N = 50
    h = 1/N
    xpts = np.arange(0, 1+h, h)
    bcd = [1, 0]
    alpha = spectralCollocation(xpts, bcd)
    true = lambda x: (x - 1)**2

    XX = np.linspace(0, 1, 100)
    Ys = []
    Yt = []
    for x in XX:
        Ys.append(collocEval(alpha, x))
        Yt.append(true(x))
    YY = np.array(Ys)
    Ytrue = np.array(Yt)

    plt.plot(XX, YY)
    plt.plot(XX, Ytrue, linestyle='--')
    plt.title("spectral collocation method")
    plt.show()



