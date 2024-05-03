def cumSumReset(A):
    import numpy as np
    a = np.where(A == 0)
    A[a] = 1 - np.diff(np.concatenate(([0], a[0])))
    F = np.cumsum(A, axis=0)

    return(F)

def cumSumResetEXTRA(A, extrafetch):
    import numpy as np
    # cumsumsetExtra script
    S = A.copy()
    S[1, 2:-2] = extrafetch
    S[S == 0] = np.nan
    S[S == 1] = 0
    G = np.cumsum(S, 1)
    G[np.np.isnan(G)] = 0

    a = np.where(A == 0)
    A[a] = 1 - np.diff(np.concatenate(([0], a[0])))
    F = np.cumsum(A, axis=0)
    F = F + G

    return(F)

def cumSumResetEXTRALateral(A, extrafetch, Lbasin, selfindex):
    import numpy as np
    S = A
    # ll = length(S(1, 2:end - 1));
    # S(1, 2: end - 1)=extrafetch * [1: ll] / ll;

    S[1, -Lbasin:-2] = extrafetch;
    # S(1,:)=extrafetch;

    S[S == 0] = np.nan
    S[S == 1] = 0
    G = np.cumsum(S, 1)
    G[np.isnan(G)] = 0

    a = np.where(A == 0)
    A[a] = 1 - np.diff(np.concatenate(([0], a[0])))
    F = np.cumsum(A, 1)
    F = F + G

    return(F)

def diffuseFetch(A, F, alpha, dx):
    import numpy as np
    from scipy import sparse

    D=A*(alpha*3600*24)/(dx**2)

    G=0*F
    p=np.where(A==1) # exclude the NOLAND CELLS (A==0)
    NN=len(p)
    G[p]=range(0, NN)
    rhs=F[p]
    [m,n]=G.shape()
    i=[]
    j=[]
    s=[]

    S=0*G
    tmp = selfindex[p]
    row, col = np.unravel_index(tmp, shape=(n, m))
    # [row, col]=ind2sub(size(A),p);
    for k in [m, -1, 1, -m]:
        # avoid to the the cells out of the domain (risk to make it periodic...)
        if k == m:
            aa = np.where(col+1<=n)
        if k == -m:
            aa = np.where(col-1>0)
        if k == -1:
            aa = np.where(row-1>0)
        if k == 1:
            aa = np.where(row+1<=m)


        q=p+k #the translated cell
        a=aa(A[q[aa]]==1 ) #only inclued the cells in whcih you can creep to

        value = min(D(p(a)),D(q(a))) # value=(D(p(a))+D(q(a)))/2.*facNL

        S[p[a]]=S[p[a]] + value # exit from that cell
        i = [i, G[q[a]]]
        j = [j, G[p[a]]]
        s = [s, -value] # gain from the neighbor cell

    # summary of the material that exits the cell
    i=[i, G[p]]
    j=[j, G[p]]
    s=[s, 1+S[p]]
    ds2 = sparse.csc_array((s, (ilog, jlog)), shape=(n * m, n * m))
    #ds2 = sparse(i,j,s) # solve the matrix inversion
    P = ds2\rhs
    F[G>0]=full[P[G[G>0]]]

def YeV(fetch, wind, h):
    import numpy as np
    # the below code was AI translated from MATLAB and has not yet been double checked
    g = 9.8
    delta = h * g / wind ** 2
    chi = fetch * g / wind ** 2
    epsilon = 3.64 * 10 ** -3 * (np.tanh(0.493 * delta ** 0.75) * np.tanh(
        3.13 * 10 ** -3 * chi ** 0.57 / np.tanh(0.493 * delta ** 0.75))) ** 1.74
    ni = 0.133 * (np.tanh(0.331 * delta ** 1.01) * np.tanh(
        5.215 * 10 ** -4 * chi ** 0.73 / np.tanh(0.331 * delta ** 1.01))) ** -0.37
    Hs = 4 * np.sqrt(wind ** 4 * epsilon / g ** 2)
    Tp = wind / ni / g

    return(Hs, Tp)

def wavek(F, H):
    import numpy as np
    g = 9.80171

    e1 = 4 * np.pi ** 2 * F ** 2 * H / g  # f4 = omega^2 * h1/g
    e2 = 1 + 0.6666666 * e1 + 0.355555555 * e1 ** 2 + 0.1608465608 * e1 ** 3 + \
         0.0632098765 * e1 ** 4 + 0.0217540484 * e1 ** 5 + 0.0065407983 * e1 ** 6
    e3 = +e1 ** 2 + e1 / e2
    K1 = np.sqrt(e3) / H

    # compute error as basis for interpolation

    o1 = np.sqrt(g * K1 * np.tanh(K1 * H))
    e1 = o1 ** 2 * H / g
    e2 = 1 + 0.6666666 * e1 + 0.355555555 * e1 ** 2 + 0.1608465608 * e1 ** 3 + \
         0.0632098765 * e1 ** 4 + 0.0217540484 * e1 ** 5 + 0.0065407983 * e1 ** 6
    e3 = +e1 ** 2 + e1 / e2
    K2 = np.sqrt(e3) / H

    # interpolate
    K = 2 * K1 - K2

    return(K)

def edgeErosion(P,z,aw,maxedgeheight,fox,dt,dx,MASK,A,Y2OX):
    import numpy as np

    # MASK 1 are the mudflat
    # MASK 2 is where the are no waves
    # fox=0;

    # the non-periodic cell are walls
    MASK[0, :] = 0
    MASK[-1, :] = 0

    # you get the wave power from the 4 surrounding cells
    Pedge = np.zeros_like(P)
    Pedge[:, 1:] += P[:, :-1]
    Pedge[:, :-1] += P[:, 1:]
    Pedge[1:, :] += P[:-1, :]
    Pedge[:-1, :] += P[1:, :]
    Pedge[MASK == 1] = 0  # the wave power in the mudflat becomes zero!

    # find the marsh cells with some wave power around it
    edg = np.where((A == 1) & (MASK == 0) & (Pedge > 0))

    # these cells will erode (the "high" cells)
    r = np.random.rand(len(edg[0]))
    a = np.where(r < aw * Pedge[edg] * dt / dx)

    # these cells will erode (the "high" cells)
    edger = edg[0][a]

    deltaY2 = np.zeros_like(MASK)
    EdgeERY2 = np.zeros_like(MASK)

    # m  max height that is eroded, to avoid strange erosion of channels
    N, M = z.shape
    for i in range(len(edger)):

        # these are the adjacent cells
        I, J = np.unravel_index(edger[i], z.shape)
        p = [np.ravel_multi_index(((I + 1 - 1) % N, J), z.shape),
             np.ravel_multi_index(((I - 1 - 1) % N, J), z.shape),
             np.ravel_multi_index((I, (J + 1 - 1) % M), z.shape),
             np.ravel_multi_index((I, (J - 1 - 1) % M), z.shape)]

        # standard way
        # a = np.where((MASK[p]==1) & (A[p]==1))  # only chooses the adjacent cells if they are mudflat and if they are standard cells

        # to erode also at the open boundary
        a = np.where(MASK[p] == 1)

        if len(a[0]) > 0:  # yes, there are adjacent cells
            dz = z[edger[i]] - z[p[a]]
            dz = np.mean(np.maximum(dz, 0))
            dz = min(dz, maxedgeheight)

            # this is how much the bed is lowered, includes everything!!!
            deltaY2[edger[i]] += dz

            # This goes into resuspension. This is what is conserved!!!
            EdgeERY2[edger[i]] += dz * (1 - fox)

            # Keep track of how much you oxidized!!!!
            Y2OX += dz * fox

def diffuseEdgeSediments(A,F,alpha,dx):
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    D = A * (alpha * 3600 * 24) / (dx ** 2)

    G = 0 * F
    p = np.where(A == 1)[0]  # exclude the NOLAND CELLS (A==0)
    NN = len(p)
    G[p] = np.arange(1, NN + 1)
    rhs = F[p]
    m, n = G.shape
    i = []
    j = []
    s = []

    S = 0 * G
    row, col = np.unravel_index(p, A.shape)
    for k in [m, -1, 1, -m]:
        # avoid to the the cells out of the domain (risk to make it periodic...)
        if k == m:
            aa = np.where(col + 1 <= n)[0]
        elif k == -m:
            aa = np.where(col - 1 > 0)[0]
        elif k == -1:
            aa = np.where(row - 1 > 0)[0]
        elif k == 1:
            aa = np.where(row + 1 <= m)[0]

        q = p + k  # the translated cell
        a = aa[A[q[aa]] == 1]  # only include the cells in which you can creep to

        value = np.min(D[p[a]], D[q[a]])
        # value=(D(p(a))+D(q(a)))/2.*facNL;

        S[p[a]] = S[p[a]] + value  # exit from that cell
        i.extend(G[q[a]].tolist())
        j.extend(G[p[a]].tolist())
        s.extend((-value * np.ones_like(G[q[a]])).tolist())  # gain from the neighbor cell

    # summary of the material that exits the cell
    i.extend(G[p].tolist())
    j.extend(G[p].tolist())
    s.extend((1 + S[p]).tolist())

    ds2 = sparse.csr_matrix((s, (i, j)), shape=G.shape)  # solve the matrix inversion
    P = spsolve(ds2, rhs)
    F[G > 0] = P[G[G > 0] - 1]


