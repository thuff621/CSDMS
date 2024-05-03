class bedcreepponds:
    def __init__(self, z,A, Active, crMUD, crMARSH, dx, dt, VEG, S, Qs2, rbulk2, alphaMUD):
        self._elev = z
        self._model_domain = A
        self._dt = dt
        self._dx = dx
    def bedcreepponds(self):
        import numpy as np
        import scipy
        from scipy import sparse
        from scipy.sparse.linalg import LinearOperator

        # matlab script function z=bedcreepponds(z,A,Active,Yreduction,crMUD,crMARSH,dx,dt,VEG,S,Qs,rbulk2,alphaMUD);
        # values bein input in original matlab script bedcreepponds(z, A, Active, A * 0 + 1, crMUD, crMARSH, dx, dt, VEG, S, Qs2, rbulk2, alphaMUD)
        # A(Active == 0) = 0 # not sure this is necessary in the python script
        self._Qs = self._Qs/self._rbulk

        # setup a placeholder for z
        zhld = self._elev.copy()

        Yreduction = self._model_domain * 0 + 1

        creep = self._model_domain * 0
        creep[self._vegetation == 0] = self._crMUD + (self._alphaMUD * 3600*24*self._Qs[self._vegetation == 0])
        creep[self._vegetation == 1] = self._crMARSH

        D = (creep) / (self._dx**2)*self._dt # yreduction

        G = self._elev * 0
        p = np.where(self._model_domain == 1, True, False)
        G[p] = self._index[p]
        rhs = self._elev[p]
        i = []
        j = []
        s = []

        Spond = S # this is the location of the ponds
        #
        S = G * 0  # this is no longer pond locations.  Not sure why this is done this way in the matlab code...


        # indexing methodology daken directory from the revious morphology steps
        N, M = self._grid.shape
        S = np.zeros((N * M))
        ilog = []
        jlog = []
        s = []
        for k in [N, -1, 1, -N]: # calculate the gradianets between cells in the x and y direction
            # print(k)
            tmp = self._index[p]
            row, col = np.unravel_index(tmp, shape=(N, M)) # sort this out.
            # indTemp = np.reshape(self._index, (N, M))
            if k == N:
                a = np.where(col + 1 < M, True, False)
                q = tmp + 1
            if k == -N:
                a = np.where(col - 1 >= 0, True, False)
                q = tmp - 1 # originally tmp was tmp[p]
            if k == -1:
                a = np.where(row - 1 >= 0, True, False)
                q = tmp - M
            if k == 1:
                a = np.where(row + 1 < N, True, False)
                q = tmp + M

            # numerical array that cooresponds to the index values covered by water
            parray = self._index[p]

            # only include cells that can creep
            a[q[a]] = np.where(a[q[a]] == True, np.where(self._model_domain[q[a]] != 1,
                         True, False), False)


            value = (D[parray[a]] + D[q[a]]) / 2

            value = value * np.minimum((Yreduction[parray[a]]), (Yreduction[q[a]]))

            # do not allow the edges of ponds to creep
            value[(Spond[parray[a]] == 1) and (Spond[q[q]] == 0)] = 0
            value[(Spond[parray[a]] == 0) and (Spond[q[q]] == 1)] = 0

            try:
                ilog = list(ilog) + list(q[a])
                jlog = list(jlog) + list(parray[a])
            except:
                print("There was an issue with ilog or jlog creation")
            # build a list of values exiting the node
            s = list(s) + list(-value)
        ilog = list(ilog) + list(q[a])
        jlog = list(jlog) + list(parray[a])
        print("There was an issue with ilog or jlog creation")
        # build a list of values exiting the node
        s = list(s) + list(1 + S[p])
        ds2 = sparse.csc_array((s, (ilog, jlog)), shape=(N * M, N * M))
        try:
            P = scipy.sparse.linalg.spsolve(ds2, rhs) # was working with .lsqr
        except:
            print("Matrix solution was singular. Reverting to lsqr to solve matrix inversion")
            P = scipy.sparse.linalg.lsqr(ds2, rhs, iter_lim = 5000)[0]
        zhld[G>0] = P[G[G>0]]
        return(zhld)
