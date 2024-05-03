import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from excludeboundarycell import excludeboundarycell
from periodicY import periodicY

# you will need to call the excludeboundarycell and periodicY functions for this script to work

def flowBasin(A, MANN, h, ho, d, dx, DH, T, periodic, kro):
    Uo = 1
    A[A == 22] = 1
    A[A == 3] = 1
    MANN[np.isnan(MANN)] = 0.1
    csi = h ** (1 / 3) / MANN ** 2 / Uo * 24 * 3600

    D = csi * h ** 2 / (dx ** 2)
    
    G = np.zeros_like(A)
    a = np.nonzero(A)
    NN = len(a[0])
    G[a] = np.arange(1, NN+1)
    rhs = np.ones(NN) * DH[a] / (T / 2 * 3600 * 24)

    N, M = G.shape
    
    i = []
    j = []
    s = []
    
    a = np.nonzero(np.logical_or(A == 2, A == 21))
    i.extend(G[a].flatten())
    j.extend(G[a].flatten())
    s.extend(np.ones_like(a[0]))
    # print(f'rhs is {rhs}')
    # print(G.shape)
    # print(G[a[0], a[1]])
    foo = np.array([G[a[0], a[1]]])
    foo = foo.astype(int)
    foo = foo - 1
    # print(foo)
    rhs[foo] = 0
    
    S = np.zeros_like(G)
    
    p = np.nonzero(np.logical_or(A == 1, A == 10))
    # print(len(p[0]))
    # print(p)
    # print(A.shape)
    # stop
    # row, col = np.unravel_index(p, A.shape)

    for k in [NN, -1, 1, -NN]:
        if periodic == 0:
            a, q = excludeboundarycell(k, NN, M, p)
            # a =
        elif periodic == 1:
            a, q = periodicY(k, NN, M, p)

        # test this section
        print(a)
        # print(A)
        # print(q)
        # print(p)

        # working here.  There is an issue with the dimentions between the different arrays.


        print(f'q shape {q.shape}')
        # test section
        q = np.squeeze(q)
        # print(q.shape)
        # print(q[:,a])
        # stop
        print(a.shape)
        stop
        # print(a)
        # print(A.shape)
        # print(A)

        a = a[A[q[:,a], q[a,:]] > 0]
        stop
        DD = (D[p[a]] + D[q[a]]) / 2
        S[p[a]] = S[p[a]] + DD
        i.extend(G[q[a]].flatten())
        j.extend(G[p[a]].flatten())
        s.extend(-DD)
    
    i.extend(G[p].flatten())
    j.extend(G[p].flatten())
    s.extend(S[p])
    
    ds2 = csr_matrix((s, (i, j)), shape=(NN, NN))
    p = ds2.tocsr().solve(rhs)
    
    P = np.zeros_like(G)
    P[G > 0] = p[G[G > 0]]
    P[A == 2] = 0
    
    D = D / h * dx
    
    Ux = np.zeros_like(A)
    Uy = np.zeros_like(A)
    U1 = np.zeros_like(A)
    Um1 = np.zeros_like(A)
    UN = np.zeros_like(A)
    UmN = np.zeros_like(A)
    
    p = np.nonzero(np.logical_or(np.logical_or(A == 1, A == 10), A == 2))
    row, col = np.unravel_index(p, A.shape)
    
    for k in [NN, -1, 1, -NN]:
        if periodic == 0:
            a, q = excludeboundarycell(k, NN, M, p)
        elif periodic == 1:
            a, q = periodicY(k, NN, M, p)
        
        a = a[A[q[a]] > 0]
        DD = np.minimum(D[p[a]], D[q[a]])
        
        if k == 1:
            U1[p[a]] = U1[p[a]] + np.sign(k) * (P[p[a]] - P[q[a]]) * DD
        elif k == -1:
            Um1[p[a]] = Um1[p[a]] + np.sign(k) * (P[p[a]] - P[q[a]]) * DD
        elif k == NN:
            UN[p[a]] = UN[p[a]] + np.sign(k) * (P[p[a]] - P[q[a]]) * DD
        elif k == -NN:
            UmN[p[a]] = UmN[p[a]] + np.sign(k) * (P[p[a]] - P[q[a]]) * DD
    
    Uy = np.maximum(np.abs(U1), np.abs(Um1)) * np.sign(U1 + Um1)
    Ux = np.maximum(np.abs(UN), np.abs(UmN)) * np.sign(UN + UmN)
    
    U = np.sqrt(Ux ** 2 + Uy ** 2)
    q = U * h
    
    return U, Ux, Uy, q, P
