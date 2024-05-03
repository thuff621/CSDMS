#! /usr/bin/env python
"""pyMarshMorpho2D model."""
import numpy as np
import pandas as pd
from landlab import Component
from landlab.components import TidalFlowCalculator
import math
import sys

np.set_printoptions(threshold=sys.maxsize)
import warnings
import scipy
from scipy import sparse
import numpy as np

warnings.filterwarnings("error")


class pymm2d(Component):
    """Simulate tidal marsh evolution."""

    _name = "pymm2d"

    _cite_as = """
    """

    _info = {
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "water_depth": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Water depth",
        },
        "fully_wet__depth": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Water depth, with depth > 0 everywhere",
        },
        "veg_is_present": {
            "dtype": bool,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "True where marsh vegetation is present",
        },
        "vegetation": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "?",
            "mapping": "node",
            "doc": "Some measure of vegetation...?",
        },
        "roughness": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "s/m^1/3",
            "mapping": "node",
            "doc": "Manning roughness coefficient",
        },
        "tidal_flow": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/s",
            "mapping": "node",
            "doc": "half tidal range flow velocities",
        },
        "percent_time_flooded": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "percent",
            "mapping": "node",
            "doc": "The percent of a 24 hour day that the node is covered by water.",
        },
        "land_cover": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "percent",
            "mapping": "node",
            "doc": "The classified land cover of the marsh by the model. 0=water, 1=low marsh, 2=high marsh/uplands",
        },
        "land_cover_change": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "percent",
            "mapping": "node",
            "doc": "The classified land cover of the marsh by the model with the addition of areas that changed."
                   " 0=water, 1=low marsh, 2=high marsh/uplands, 3=change areas",
        },
        "accretion_over_time": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "percent",
            "mapping": "node",
            "doc": "Spatial map of marsh accretion over time"
        },
        "p_mask": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "percent",
            "mapping": "node",
            "doc": "Spatial map of masked out areas excluded from flow morphology calculations"
        },
        "a_mask": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "percent",
            "mapping": "node",
            "doc": "Spatial map of masked out areas excluded from flow morphology calculations after the q and parry gradient filters"
        },
    }

    def __init__(self, grid,
                 rel_sl_rise_rate=None,  # 4.0/1000/365,  # originally 2.74e-6
                 tidal_range = 1.2192,  # tidal range taken from Atlantic city NOAA tide gauge.
                 tidal_range_for_veg = 1.225296,
                 roughness_with_veg = 0.1,
                 roughness_without_veg = 0.02,
                 tidal_period = 12.5 / 24,
                 roughness = None,
                 model_domain=None,
                 unit = None,
                 boundaryValues=None,
                 runMEM=None,
                 printAll=None
                 ):
        """Initialize the MarshEvolver.

        Parameters
        ----------
        grid : ModelGrid object
            Landlab model grid
        rel_sl_rise_rate : float
            Rate of relative sea-level rise, m/day
        tidal_range : float
            Tidal range, m
        tidal_range_for_veg : float
            Tidal range for vegetation model, m (normally same as tidal range)
        """

        super(pymm2d, self).__init__(grid)
        self.initialize_output_fields()

        # Get references to fields
        self._elev = self._grid.at_node['topographic__elevation']
        self._water_depth = self._grid.at_node['water_depth']
        self._fully_wet_depth = self._grid.at_node['fully_wet__depth']
        self._veg_is_present = self._grid.at_node['veg_is_present']
        self._vegetation = self._grid.at_node['vegetation']
        self._roughness = self._grid.at_node['roughness']
        self._tidal_flow = self._grid.at_node['tidal_flow']
        self._percent_time_flooded = self._grid.at_node['percent_time_flooded']
        self._boundaryValues = boundaryValues  # These are the topo values that boundary areas will be set back to at the end of a loop
        self._withMEM = runMEM

        # apply roughness through hyperparameter
        from landlab.grid.raster_mappers import map_mean_of_links_to_node
        self._roughness = map_mean_of_links_to_node(grid, roughness)

        # Set parameter values
        self._mean_sea_level = 0  # m NAVD88
        self._elev_relative_to_MSL = self._elev - self._mean_sea_level
        self._rel_sl_rise_rate = rel_sl_rise_rate
        self._tidal_range = tidal_range
        self._tidal_range_for_veg = tidal_range_for_veg
        self._tidal_half_range = tidal_range / 2.0
        self._tidal_period = tidal_period  # this is the tidal period
        self._numberOfTidesPerYear = 365 / self._tidal_period  # get the number of tides in a year
        self._roughness_with_veg = roughness_with_veg
        self._roughness_without_veg = roughness_without_veg
        self._taucr = 0.2  # a user defined value as part of the mud paramaters in the original code it equals 0.2
        self._taucrVEG = 0.5  # a user defined value as aprt of the vegetation parameters "Critical Shear Stress for vegetated areas. The original value was 0.5
        self._me = 0.1 * pow(10,
                             -4) * 24 * 3600  # mud parameter per day *unsure exactly what this is but its part of the calucation for E
        self._wsB = 1 / 1000  # P.wsB=1/1000;#Mud Settling velocity for vegetated area
        self._ws2 = 0.2 / 1000  # P.ws2=0.2/1000;#0.2/1000;# m/s
        self._DoMUD = 1  # base diffusivity of suspended mud [m2/s]. Process not related to tides (e.g. wind and waves, other ocean circulation)
        self._Diffs = 1  # [-]coefficient for tidal dispersion [-]. 1 DO NOT CHANGE
        # self._tidal_period = tidal_period# 12.5/24 # tidal period [day]
        self._rhos = 2650  # sediment density(quartz - mica)[kg / m3]
        self._por2 = 0.7  # A user defined porocity value
        self._rbulk = self._rhos * (1 - self._por2)
        self._sea_SSC = 60 / 1000  # 40/1000; # Sea boundary SSC for mud [g/l]
        self._limitdeltaz = 2  # meters maximum erosion value
        self._limitmaxup = 1  # meters maximum deposition value
        self._min_water_depth = 0.2  # minimuim water depth
        self._suspended_sediment = 23  # 73.0 # suspended sediment concentration, mg/l
        self._accretion_over_time = np.zeros(self._grid.shape).flatten()  # logs the accretion over time
        self._KBTOT = 0  # something
        self._crMUD = 3.65 / 365  # creep coefficient
        self._alphaMUD = 0.25  # coefficient for bedload downslope of mud.added April 2019. Similar to P.alphaSAND
        self._crMARSH = 0.1 / 365  # creep coefficient vegetated
        self._unit = int(abs(self._grid.x_of_node[0] - self._grid.x_of_node[1]))# unit  # this is the spatial grid size or unit.
        self._printAll = printAll  # this is to toggle printing on or off. If printAll = None then no printing occures in regards to the "P" model solver.


        # lower and upper limits for veg growth [m]
        # see McKee, K.L., Patrick, W.H., Jr., 1988.
        # these are "dBlo" and "dBup" in matlab original
        self._min_elev_for_veg_growth = -(0.237 * self._tidal_range_for_veg
                                          - 0.092)
        self._max_elev_for_veg_growth = self._tidal_range_for_veg / 2.0

        # default model domain
        self._model_domain = np.reshape(model_domain, np.shape(self._elev))

        # setup index
        self._index = np.array(range(0, len(self._elev)))

    def exportSelfParam(self, saveLocation=None):
        import pandas as pd
        # this function prints/writes out the model settings
        print("Initial Model Parameters")
        print("-" * 60)
        print('\n')
        if saveLocation == None:  # Just print out the values
            for i in vars(self):
                try:
                    if len(vars(self)[i]) < 10:
                        print()
                except:
                    if i != "_grid":
                        print(f"    self.{i} = {vars(self)[i]}")
        else:
            att = []
            val = []
            for i in vars(self):
                try:
                    if len(vars(self)[i]) < 10:
                        print()
                except:
                    if i != "_grid":
                        att.append(''.join(['self.', str(i)]))
                        val.append(vars(self)[i])
                        print(f"    self.{i} = {vars(self)[i]}")
            hd = pd.DataFrame(zip(att, val), columns=["attribute_name", "value"])
            name = ''.join([saveLocation, '.csv'])
            print(f'Initial Model parameters have been exported to {name}')
            hd.to_csv(name)
        print("-" * 60)
        print('\n')

    def edgeEffects(self, arr):
        bndryNodes = (self._grid.closed_boundary_nodes)
        adjNodes = self._grid.adjacent_nodes_at_node[(bndryNodes)]

        # get the edge nodes and remove from consideration with boundary nodes.
        # this is meant to get all of the actual edge nodes and compare the model edge nodes.
        indexMatrix = np.reshape(self._index, self._grid.shape)
        edgeList = [indexMatrix[:,0], indexMatrix[:,-1], indexMatrix[0,:], indexMatrix[-1,:]]
        edgeNodes = []
        for e in edgeList:
            for val in e:
                edgeNodes.append(val)

        # fill the boundary node values with the mean value from adjacent nodes
        for i in range(len(bndryNodes)):
            idx = adjNodes[i]
            idx = idx[idx != -1]
            tf = np.in1d(idx, edgeNodes)
            idx = idx[tf == False]

            if bndryNodes[i] in edgeNodes:
                if len(idx) > 0:
                    arr[bndryNodes[i]] = np.mean(arr[idx])
                else:
                    # if no matcher exists the point has to be in a corner and needs to get the value on a diagonal.
                    # find the index value for corners
                    col, row = self._grid.shape
                    corners = [self._index[0], self._index[row - 1], self._index[(row * col) - row],
                               self._index[row * col - 1]]
                    replacementIndex = np.array(
                        [self._index[row + 1], self._index[row * 2 - 2], self._index[((col - 2) * row) + 1],
                         self._index[row * (col - 1) - 2]])
                    msk = np.array(np.in1d(corners, bndryNodes[i]))
                    idx = replacementIndex[msk]
                    try:
                        arr[bndryNodes[i]] = arr[idx][0]
                    except:
                        print(f'There was an issue with an unknown node {bndryNodes[i]}.')
        return (arr)

    def bedcreepponds(self, dx, dt):
        import numpy as np
        import scipy
        from scipy import sparse

        self._dx = dx
        self._dt = dt

        # matlab script function z=bedcreepponds(z,A,Active,Yreduction,crMUD,crMARSH,dx,dt,VEG,S,Qs,rbulk2,alphaMUD);
        # values bein input in original matlab script bedcreepponds(z, A, Active, A * 0 + 1, crMUD, crMARSH, dx, dt, VEG, S, Qs2, rbulk2, alphaMUD)
        self._Qs = self._Qs / self._rbulk

        Yreduction = (self._model_domain * 0) + 1

        creep = self._model_domain * 0
        creep[self._vegetation == 0] = self._crMUD + (self._alphaMUD * 3600 * 24 * self._Qs[self._vegetation == 0])
        creep[self._vegetation == 1] = self._crMARSH

        D = (creep) / (self._dx ** 2) * self._dt  # yreduction

        G = self._elev * 0
        p = np.where(self._model_domain == 1, True, False)
        G[p] = range(0, len(self._index[p]))  # self._index[p]
        rhs = self._elev[p]

        Spond = self._elev * 0

        # indexing methodology daken directory from the revious morphology steps
        N, M = self._grid.shape
        S = np.zeros((N * M))
        ilog = []
        jlog = []
        s = []

        # figure out this section
        for k in [N, -1, 1, -N]:  # calculate the gradianets between cells in the x and y direction
            # print(k)
            tmp = self._index[p].copy().astype(int)
            N, M = self._grid.shape
            row, col = np.unravel_index(tmp, shape=(N, M))  # sort this out.
            # indTemp = np.reshape(self._index, (N, M))
            if k == N:
                a = np.where(col + 1 < M, True, False)
                q = tmp + 1
            if k == -N:
                a = np.where(col - 1 >= 0, True, False)
                q = tmp - 1  # originally tmp was tmp[p]
            if k == -1:
                a = np.where(row - 1 >= 0, True, False)
                q = tmp - M
            if k == 1:
                a = np.where(row + 1 < N, True, False)
                q = tmp + M

            parray = tmp

            # 8-28-2023 updated logic to speed up processing
            ptmpArray = np.array(list(range(len(tmp))))[a]  # updated 9-29-2023
            cls = np.where(a[ptmpArray] == True, np.where(self._model_domain[q[a]] != -1, True, False), False)
            a[ptmpArray[cls]] = True
            a[ptmpArray[cls == False]] = False

            value = (D[parray[a]] + D[q[a]]) / 2
            value = value * np.minimum((Yreduction[parray[a]]), (Yreduction[q[a]]))

            # do not allow the edges of ponds to creep
            value[np.where(Spond[parray[a]] == 1, np.where(Spond[q[a]] == 0, True, False), False)] = 0
            value[np.where(Spond[parray[a]] == 0, np.where(Spond[q[a]] == 1, True, False), False)] = 0

            try:
                ilog = list(ilog) + list(G[q[a]].astype(int))
                jlog = list(jlog) + list(G[parray[a]].astype(int))
            except:
                print("There was an issue with ilog or jlog creation")
            # build a list of values exiting the node
            S[parray[a]] = S[parray[a]] + value
            s = list(s) + list(-value)
        ilog = list(ilog) + list(G[p].astype(int))
        jlog = list(jlog) + list(G[p].astype(int))
        # build a list of values exiting the node
        s = list(s) + list(1 + S[p])
        ds2 = sparse.csc_array((s, (ilog, jlog)))
        try:
            P = scipy.sparse.linalg.spsolve(ds2,
                                            rhs)  # was working with .lsqr # look into skcuda and cusparse to accelerate the model computation.
        except:
            if self._printAll != None:
                print("Bedcreep matrix solution was singular. Reverting to lsqr to solve matrix inversion")
            P = scipy.sparse.linalg.lsqr(ds2, rhs, iter_lim=5000)[0]
        zhld = self._elev.copy()
        zhld[:] = 0
        # zhld = np.zeros(self._elev)
        zhld[self._index[p]] = np.array(P)
        # zhld[np.isnan(zhld)] = 0
        # apply edge effect fix
        zhld = self.edgeEffects(zhld)
        return (zhld)

    def get_water_depth(self, min_depth=0.01):
        """Calculate the water depth field."""

        depth_at_mean_high_water = np.maximum((-self._elev)
                                              + self._mean_sea_level
                                              + self._tidal_half_range, 0.0)
        self._depth_at_mean_high_water = depth_at_mean_high_water  # record this value for later use
        self._fully_wet_depth = (0.5 * (depth_at_mean_high_water
                                        + np.maximum(depth_at_mean_high_water
                                                     - self._tidal_range, 0.0)))
        self._water_depth_ho = self._fully_wet_depth.copy()
        self._hydroperiod = np.minimum(1, np.maximum(0.001, (depth_at_mean_high_water / self._tidal_range)))
        self._fully_wet_depth[self._fully_wet_depth < min_depth] = min_depth
        self._water_depth[:] = self._fully_wet_depth
        self._water_depth[self._elev > (self._mean_sea_level + self._tidal_half_range)] = 0.0
        # self._water_depth[self._water_depth < self._min_water_depth] = self._min_water_depth
        hxxx = self._water_depth[self._water_depth < min_depth]
        relax = 10
        self._water_depth[self._water_depth < min_depth] = np.maximum(self._min_water_depth,
                                                                      min_depth * (1 - np.exp(-hxxx * relax)) / (
                                                                                  1 - np.exp(-min_depth * relax)))

    def update_flow(self):
        from landlab.grid.raster_mappers import map_mean_of_horizontal_links_to_node, map_mean_of_vertical_links_to_node
        from landlab.grid.mappers import map_mean_of_link_nodes_to_link
        from landlab.components import TidalFlowCalculator
        # from . import ERDC_TidalFlowCalculculator as flowCal
        """update the flow within the grid"""
        # you need to set the boundaries on the grid
        roughnessArray = map_mean_of_link_nodes_to_link(self._grid, self._roughness)
        # flood = flowCal.TidalFlowCalculator(self._grid, mean_sea_level=self._mean_sea_level,
        #                           tidal_range=self._tidal_range, tidal_period = self._tidal_period*24*3600,
        #                           roughness=roughnessArray, min_water_depth=0.1, scale_velocity=1)

        flood = TidalFlowCalculator(self._grid, mean_sea_level=self._mean_sea_level,
                                  tidal_range=self._tidal_range, tidal_period = self._tidal_period*24*3600,
                                  roughness=roughnessArray, min_water_depth=0.1, scale_velocity=1)
        flood.run_one_step()

        # tidal inundation time using Rober Hickey's method https://link.springer.com/article/10.1007/s11852-019-00690-2
        mhw = self._mean_sea_level + (self._tidal_range / 2)
        mlw = self._mean_sea_level - (self._tidal_range / 2)
        hld = 2 * (self._elev - mlw) / (mhw - mlw) - 1
        hld[hld > 1] = 1
        hld[hld < -1] = -1
        A = 2 * np.pi - np.arccos(hld)
        rising_time_over_cell = (self._tidal_period / 2 * 24) * (A / np.pi - 1)
        hld = 2 * (self._elev - mhw) / (mlw - mhw) - 1
        hld[hld > 1] = 1
        hld[hld < -1] = -1
        A = 2 * np.pi - np.arccos(hld)
        fall_time_over_cell = (self._tidal_period / 2 * 24) * (A / np.pi - 1)

        # calculate the number of hours in a day that the marsh surface is inundated.
        innundation_time = ((abs(rising_time_over_cell - (
                    self._tidal_period / 2 * 24)) + fall_time_over_cell)) * self._numberOfTidesPerYear  # / ((self._tidal_period * 24) / 24))
        self._percent_time_flooded = innundation_time / 8760 * 100  # inundation time / hours in a year * percent conversion
        self.grid.at_node['percent_time_flooded'] = self._percent_time_flooded

        self._ebb_tide_vel = flood._ebb_tide_vel
        self._flood_tide_vel = flood._flood_tide_vel
        self._Uy = map_mean_of_vertical_links_to_node(self._grid,
                                                      flood._flood_tide_vel)  # not sure about the units but this matches the matlab output. Thomas added the *2/100 conversion factor
        self._Ux = map_mean_of_horizontal_links_to_node(self._grid,
                                                        flood._flood_tide_vel)  # not sure about the units but this matches the matlab output. Thomas added the *2/100 conversion factor

        # swapping direction is necessary to match the matlab code
        self._Ux = self._Ux * -1
        self._Uy = self._Uy * -1

        # fill in the outer edges of the grid.
        # Landlab leaves these edges as zero due to the core matrix calculations however this causes issues with the matrix inversion step
        # in the morphology calculator
        # find the edges of the grid that are not open as part of the domain
        gridNum = np.reshape(self._index, (self._grid.shape))
        edges = [gridNum[0, :], gridNum[-1, :], gridNum[:, 0], gridNum[:, -1]]
        # find the numbers that are different from the domain boundary

        # closedBndrys
        # gridNum = np.reshape(self._index, (self._grid.shape))

        self._Ux = self.edgeEffects(self._Ux)
        self._Uy = self.edgeEffects(self._Uy)

        self._U = np.sqrt(
            (self._Ux ** 2) + (self._Uy ** 2))  # not sure about the units but this matches the matlab output.
        self.grid.at_node['tidal_flow'] = self._U

    def update_morphology(self, dt):
        """Update morphology
        This is currently only the simplest version (without ponding (in progress), wave errsion, etc.)"""

        # Thomas' NOTES.
        # The update_morphology function is taking the place of both the "TotalsedimenterosionMUDsine.m" script and the "sedtran.m" script.
        # The python implementation of TotalsedimenterosionMUDsine.m has been completed and validated against the matlab script.


        # record the starting elevation
        origz = np.copy(self._elev)
        p = np.where((self._model_domain > 0) & (
                    self._water_depth_ho > 0))  # Thomas add this line 9-28-2023 to manage unwetted areas.,

        p_mask = self._model_domain.copy() * 0
        p_mask[p] = 1

        self._grid.at_node['p_mask'] = p_mask

        fUpeak = math.pi / 2
        taucro = self._elev * 0 + self._taucr
        taucro[self._veg_is_present == 1] = self._taucrVEG

        # tidal current erosion ################################################################################################
        ncyc = 10
        E = 0
        for i in range(-1, ncyc):
            i = i + 1
            Utide = self._U * fUpeak * math.sin(i / ncyc * math.pi / 2)
            try:
                watPow = self._water_depth ** (-1 / 3)  # changed to fully wetted depth
            except:
                print("Zero water depth detected")
            #     watPow = self._water_depth ** (-1 / 3)
            watPow[watPow == np.inf] = 0  # set inf numbers to zero
            tauC = 1030 * 9.81 * self._roughness ** (2) * watPow * Utide ** 2
            E = E + (1 / (ncyc + 1)) * self._me * (np.sqrt(1 + (tauC / taucro) ** 2) - 1)
        E[self._model_domain == 2] = 0
        E[E == np.inf] = 0  # clean out any infinite values

        # ## CURRENT-DRIVEN TRANSPORT (Tide and River)######################################################
        # Advection-Diffusion Sediment transport
        WS = (self._elev * 0) + self._ws2
        WS[self._veg_is_present == 1] = self._wsB
        # WS(S==1)=ws2# SHOULD NOT BE NECEEARY BECUASE VEG alreeady set equal to zero where S=1 (see above).  ->Do not add the vegetation settling velocity in the ponds! #WS(S==1)=0.000000000001#E2(S==1)=0

        ###################### Sedtran ############################################
        # This is all modified by Thomas Huff from the original MatLab code
        dx = self._unit * 2  # this is possibly a velocity or erosion modifier originally 5*2
        Dxx = (self._DoMUD * 24 * 3600 + self._Diffs * self._tidal_period / 2 * np.abs(self._Ux * self._Ux) * (
                    24 * 3600) ** 2) / (dx ** 2) * self._water_depth
        Dyy = (self._DoMUD * 24 * 3600 + self._Diffs * self._tidal_period / 2 * np.abs(self._Uy * self._Uy) * (
                    24 * 3600) ** 2) / (dx ** 2) * self._water_depth

        N, M = self._grid.shape
        S = np.zeros((N * M))
        ilog = []
        jlog = []
        s = []
        for k in [N, -1, 1, -N]:  # calculate the gradianets between cells in the x and y direction
            # print(k)
            tmp = self._index[p]
            row, col = np.unravel_index(tmp, shape=(N, M))  # sort this out.
            # indTemp = np.reshape(self._index, (N, M))
            if k == N:
                a = np.where(col + 1 < M, True, False)
                q = tmp + 1
            if k == -N:
                a = np.where(col - 1 >= 0, True, False)
                q = tmp - 1  # originally tmp was tmp[p]
            if k == -1:
                a = np.where(row - 1 >= 0, True, False)
                q = tmp - M
            if k == 1:
                a = np.where(row + 1 < N, True, False)
                q = tmp + M
            # numerical array that corresponds to the index values covered by water
            parray = self._index[p]
            qtmpArray = np.array(list(range(len(q))))[a]
            a[qtmpArray] = np.where(a[qtmpArray] == True, np.where(self._model_domain[q[a]] != 1,
                                                                   np.where(self._model_domain[q[a]] != 2,
                                                                            np.where(
                                                                                self._model_domain[q[a]] == 10,
                                                                                True, False), True), True), False)
            a_mask = self._model_domain.copy() * 0
            a_mask[parray[a]] = 1
            self._grid.at_node['a_mask'] = a_mask

            # calculate DD value
            if (k == N) | (k == -N):
                D = Dyy
            else:
                D = Dxx

            numeric = 1  # This will need to be set by a hyperparameter in the future
            if numeric == 1:
                try:
                    DD = (D[parray[a]] + D[q[a]]) / 2
                except:
                    print("There was an issues with the DD calculation")
            else:
                DD = np.minimum(D[parray[a]], D[q[a]])
            try:
                value = DD / self._water_depth[parray[a]] / self._hydroperiod[parray[a]]
            except:
                print("There was an issue with the value calculation")

            # calculate a mask for flux entering or leaving a given node.
            Fin = np.copy(self._elev)
            Fin[:] = 0
            Fout = np.copy(Fin)
            # There are some challenges with getting the indexes to match and this is how I am getting around it.
            Fin[parray[a][np.in1d(parray[a], self._index[self._model_domain == 1])]] = 1
            tmpInd = q[a]
            Fout[self._index[tmpInd[self._model_domain[q[a]] == 1]]] = 1

            #######################################################################################################
            # build a list of values entering a node
            S[parray[a]] = S[parray[a]] + (value * Fin[parray[a]])
            try:
                ilog = list(ilog) + list(q[a])
                jlog = list(jlog) + list(parray[a])
            except:
                print("There was an issue with ilog or jlog creation")
            # build a list of values exiting the node
            s = list(s) + list(-value * Fout[q[a]])
        settling = np.copy(self._elev) * 0
        settling[p] = 24 * 3600 * WS[p] / self._water_depth[p] * self._hydroperiod[p]
        settling[self._index[self._model_domain == 2]] = 0
        S[self._index[self._model_domain == 2]] = 1
        aa = self._index[self._model_domain == 10]
        S[parray[aa]] = 1  # impose the b.c.
        del (aa)

        #################################################################################

        try:
            ilog = list(ilog) + list(parray)
            jlog = list(jlog) + list(parray)
            s = list(s) + list(S[p] + settling[p])
        except:
            print("Error with ilog or jlog")
        ds2 = sparse.csc_array((s, (ilog, jlog)), shape=(N * M, N * M))  # , shape = (N, M))
        rhs = np.array(E)
        rhs[:] = 0
        rhs[p] = E[p]
        aa = self._index[self._model_domain == 2]
        try:
            rhs[aa] = (60 / 1000) * self._water_depth[aa] * self._hydroperiod[aa]
        except:
            print('rhs creation issue')
        MUD = 1  # This is just for testing.  Make sure to remove this later on
        if MUD == 1:
            aa = self._index[self._model_domain == 10]
            rhs[aa] = np.nan * self._water_depth[aa] * self._hydroperiod[aa]
            del (aa)
        try:
            P = scipy.sparse.linalg.spsolve(ds2, rhs)  # was working with .lsqr
        except:
            if self._printAll != None:
                print("Morphology matrix solution was singular. Reverting to lsqr to solve matrix inversion")
            P = scipy.sparse.linalg.lsqr(ds2, rhs, iter_lim=5000)[0]

        SSM = np.array(P)  # apply the P value to the SSM varible
        EmD = np.copy(self._elev)
        EmD[:] = 0  # zero out a new array
        EmD[p] = (E[p] - SSM[p] * settling[p]) / self._rbulk
        EmD[self._model_domain == 2] = 0  # impose boundary conditions
        self._elev = self._elev - (dt * EmD)  # calculate the change in elevation.
        self._elev[self._model_domain == 2] = self._boundaryValues  # set the boundary areas to -10
        self._elev[self._model_domain == 50] = 100  # set the boundary areas to -10

        #######################################################################################################
        # add organic accretion
        noPond = np.where(S == 0, True, False)
        self._vegetation[noPond] = self._vegetation[noPond] * S[noPond]
        AccreteOrganic = 1
        if AccreteOrganic == 1:
            if self._withMEM == None or self._withMEM == True:
                # spatial accretion based on teh MEM https://github.com/tilbud/rCMEM/blob/master/R/sedimentInputs.R
                actualWaterDepth = self._water_depth
                actualWaterDepth[actualWaterDepth < 0] = 0
                # 1e-3 converts mg/l to kg/m^3
                self._accretion = ((((((self._suspended_sediment * 1e-6) * self._numberOfTidesPerYear * (
                            (actualWaterDepth * 100) * 0.5 * 1 * 1))) / (
                                                 self._rbulk / 1000) / 100)) / 365) * dt  # m accretion/day with dt modifying it to a year.
                self._accretion[self._veg_is_present == 0] = 0  # no veg no accretion!
                self._elev = self._elev + self._accretion  # put organic on mud!!!
                self._accretion_over_time += self._accretion
                self._grid.at_node['accretion_over_time'] = self._accretion_over_time
            else:
                # Original MM2D accretion values
                self._Korg = 6 / 1000 / 365  # mm/yr organic accretion rate
                self._elev = self._elev + self._vegetation * self._Korg * dt  # put organic on mud!!!
                self._accretion_over_time += self._vegetation * self._Korg * dt
                self._grid.at_node['accretion_over_time'] = self._accretion_over_time

        ##################### BED EVOLUTION DIVERGENCE ###############################################

        # fix this section to add in bed creep ponds... Pond creep seems to do more than just for ponds.....
        # EVOLUTION OF z
        self._Qs = E / (self._ws2 * 3600 * 24) * self._U * np.maximum(0,
                                                                      self._water_depth)  # kg/m/s #[hcorrected]=getwaterdepth(Trange,Hsurge,msl,z,kro,hpRIV);  #VEG=(z-msl)>dBlo;
        # znew=bedcreepponds(z,A,Active,A*0+1,crMUD,crMARSH,dx,dt,VEG,S,Qs2,rbulk2,alphaMUD)  # MUD CREEP  MARSH
        znew = pymm2d.bedcreepponds(self, dx=dx, dt=dt)
        deltaz = self._elev - znew
        deltaz[self._model_domain == 2] = 0  # DO NOT UPDATE THE BOUNDARY
        self._elev = self._elev - deltaz

        #######################################################################################################
        # return the maxdeltaz and maxup values for the loop computation in "run_one_step
        mxDelta = np.percentile(abs(self._elev - origz), 99)
        mxchup = np.maximum(0, np.maximum(0, self._elev - origz)) * (
                    (self._elev) > (self._mean_sea_level + self._tidal_range / 2))
        mxUp = np.max(mxchup[:])
        return (mxDelta, mxUp)

    # we aren't using vegetation for this example
    def update_vegetation(self, round=None):
        # Adjusted script to use Emily's vegetation coverage and roughness values
        """
        Created on Thu Sep  7 13:31:34 2023

        @author: RDEL1ERR
        """
        import numpy as np
        # This is the quadratic equation of form ax2+bx+c, where a,b,c are biomass_coef
        # from the MEM model
        biomass_coef = [-0.0002, 0.015, 0.6637]
        # The roots are where y=0 (~MLW and ~MHW)
        roots = np.sort(np.roots(biomass_coef))
        roots = roots / 100  # convert to meters

        # initalize # vegetation coverage across grid -- everything can start as 0#
        # nrow,ncol = size of landlab grid
        veg_cover = np.zeros(self._grid.shape).flatten()
        # Update only # cover where elevation is greater than roots[0] and <roots[1]
        # I don't actually have an "elev" variable, but this should be the elevation of the grid relative to MSL

        # Update elevation relative to MSL
        self._elev_relative_to_MSL = self._elev - self._mean_sea_level
        #
        # veg_cover[(elev_in_cm > roots[0]) & (elev_in_cm < roots[1])] = -0.002 * elev_in_cm[(elev_in_cm > roots[0]) & (elev_in_cm < roots[1])]**2 + 0.15 * elev_in_cm[(elev_in_cm > roots[0]) & (elev_in_cm < roots[1])] + 0.6637

        # in meters
        veg_cover[(self._elev_relative_to_MSL > roots[0]) & (self._elev_relative_to_MSL < roots[1])] = -2.0023 * \
                                                                                                       self._elev_relative_to_MSL[
                                                                                                           (
                                                                                                                       self._elev_relative_to_MSL >
                                                                                                                       roots[
                                                                                                                           0]) & (
                                                                                                                       self._elev_relative_to_MSL <
                                                                                                                       roots[
                                                                                                                           1])] ** 2 + (
                                                                                                                   1.5007 *
                                                                                                                   self._elev_relative_to_MSL[
                                                                                                                       (
                                                                                                                                   self._elev_relative_to_MSL >
                                                                                                                                   roots[
                                                                                                                                       0]) & (
                                                                                                                                   self._elev_relative_to_MSL <
                                                                                                                                   roots[
                                                                                                                                       1])]) + 0.6637

        if self._withMEM == None or self._withMEM == True:
            self._vegetation = np.zeros(self._grid.shape).flatten()
            self._vegetation[veg_cover > 0] = veg_cover[veg_cover > 0]
            self._vegetation[(self._elev - self._mean_sea_level) < self._min_elev_for_veg_growth] = 0
            # self._vegetation[:] = 0 # thomas added this section for testing
            self.grid.at_node['vegetation'] = self._vegetation

            # set vegetation present cuttoffs
            self._veg_is_present = np.zeros(self._grid.shape).flatten()
            self._veg_is_present[self._vegetation > 0] = 1
            # self._veg_is_present[self._elev_relative_to_MSL > self._min_elev_for_veg_growth] = 1
            self.grid.at_node['veg_is_present'] = self._veg_is_present

            # figure out landcover.
            self._land_cover = np.zeros(self._grid.shape).flatten()
            # self._land_cover[self._veg_is_present > 0] = 1 # this is low marsh.

            # marsh coverage
            self._land_cover[veg_cover > 0.75] = 1
            self._land_cover[(veg_cover <= 0.75) & (veg_cover >= 0.25)] = 2
            self._land_cover[(veg_cover < 0.25) & (self._veg_is_present == 1)] = 3

            # flats
            self._land_cover[(self._percent_time_flooded != 100) & (self._veg_is_present != 1) & (
                        self._elev > self._min_elev_for_veg_growth)] = 4
            try:
                self._land_cover[self._elev > max(self._elev[self._veg_is_present == 1])] = 5  # upland or high marsh
            except:
                print("There was no detected upland vegetation")
            self.grid.at_node['land_cover'] = self._land_cover
            if round == 0:  # preserve the land cover at the first time step
                self._initial_land_cover = self._land_cover
            # get land cover change over time
            if round != None:
                lcClasses = list(
                    range(0, max((self._land_cover).astype(int)) + 1))  # this is the number of classes at each step
                lcChange = self._land_cover.copy()
                for lc in lcClasses:
                    for cc in lcClasses:
                        lcChange[(self._initial_land_cover == lc) & (self._land_cover == cc)] = int(
                            ''.join([str(1), str(lc), str(cc)]))
                        # lcChange = np.where(self._initial_land_cover == lc, np.where(self._land_cover == cc, int(''.join([str(1), str(lc), str(cc)])),
                        #                                                              self._land_cover), self._land_cover)
                self.grid.at_node['land_cover_change'] = lcChange
        else:
            # Below was the original Greg Tucker script
            """Update vegetation."""
            height_above_msl = self._elev - self._mean_sea_level
            self._veg_is_present[:] = (height_above_msl
                                       > self._min_elev_for_veg_growth)
            self._vegetation = (4 * (height_above_msl
                                     - self._max_elev_for_veg_growth)
                                * (self._min_elev_for_veg_growth
                                   - height_above_msl)
                                / (self._min_elev_for_veg_growth
                                   - self._max_elev_for_veg_growth) ** 2)
            self._vegetation[height_above_msl > self._max_elev_for_veg_growth] = 0.0
            self._vegetation[height_above_msl < self._min_elev_for_veg_growth] = 0.0
            self.grid.at_node['vegetation'] = self._vegetation
            self.grid.at_node['veg_is_present'] = self._veg_is_present

            # figure out landcover.
            self._land_cover = np.zeros(self._grid.shape).flatten()
            # self._land_cover[self._veg_is_present > 0] = 1 # this is low marsh.

            # marsh coverage
            self._land_cover[self._veg_is_present == 1] = 1

            # flats
            self._land_cover[(self._percent_time_flooded != 100) & (self._veg_is_present != 1) & (
                        self._elev > self._min_elev_for_veg_growth)] = 4
            try:
                self._land_cover[self._elev > max(self._elev[self._vegetation != 0])] = 5  # upland or high marsh
            except:
                print("There was no detected upland vegetation")
            self.grid.at_node['land_cover'] = self._land_cover
            if round == 0:  # preserve the land cover at the first time step
                self._initial_land_cover = self._land_cover
            # get land cover change over time
            if round != None:
                lcClasses = list(
                    range(0, max((self._land_cover).astype(int)) + 1))  # this is the number of classes at each step
                lcChange = self._land_cover.copy()
                for lc in lcClasses:
                    for cc in lcClasses:
                        lcChange[(self._initial_land_cover == lc) & (self._land_cover == cc)] = int(
                            ''.join([str(1), str(lc), str(cc)]))
                self.grid.at_node['land_cover_change'] = lcChange

    # we aren't using roughness calculation from vegetation.
    def update_roughness(self):
        # Update roughness value based on #
        # Initialize roughness value at each cell -- start with 0.02 which is open water roughness
        if self._withMEM == None or self._withMEM == True:
            roughness = np.ones(self._grid.shape).flatten() * 0.02
            # Note everything that has a veg_cover = 0 will get a roughness value of 0
            # This will include transition and upland regions.
            roughness[self._vegetation > 0] = (1 - self._vegetation[self._vegetation > 0]) * 0.03 + (
                        self._vegetation[self._vegetation > 0] * 0.13)
            roughness[self._elev > max(self._elev[self._vegetation > 0])] = max(roughness)  # this is here for testing
            self._roughness = roughness
            self.grid.at_node['roughness'] = self._roughness
        else:
            # Below is the original Greg Tucker code
            """Update Manning's n values."""
            self._roughness[:] = self._roughness_without_veg
            self._roughness[self._veg_is_present] = self._roughness_with_veg


    def run_one_step(self, timeStep, round, model_domain, roughness,
                     relative_sea_level_rise_rate_mmPerYr=None, saveModelParamsFile=None):
        """Advance in time."""

        from landlab.grid.raster_mappers import map_mean_of_links_to_node
        self._roughness = map_mean_of_links_to_node(self._grid, roughness) # define roughness as it will change for each loop.

        if round == 0:
            if saveModelParamsFile == None:
                self.exportSelfParam()
            else:
                self.exportSelfParam(saveLocation=saveModelParamsFile)
            t = 1
            dto = 0.00001
        else:
            dto = 365
        dti = 0
        dt = dto
        while dti < dto:
            firstattempt = 1
            maxdeltaz = self._limitdeltaz + 1
            maxup = self._limitmaxup + 1

            while maxdeltaz > self._limitdeltaz or maxup > self._limitmaxup:  # changed | to or
                if firstattempt != 1:
                    try:
                        dt = dt / 2 * np.minimum((self._limitdeltaz / maxdeltaz), (self._limitmaxup / maxup))
                    except:
                        print("There was a divide by zero issue")
                    # print(f'dt bing reduced. dt value is now {dt}')
                firstattempt = 0
                if round <= 1:
                    dt = np.minimum(0.5 * 365, dt)
                    # print(f' -----> updated dt value {dt}')
                # Update sea level
                if relative_sea_level_rise_rate_mmPerYr == None:
                    self._mean_sea_level = self._mean_sea_level + (self._rel_sl_rise_rate * dt)
                else:
                    self._mean_sea_level = self._mean_sea_level + ((
                                                                               relative_sea_level_rise_rate_mmPerYr / 1000 / 365) * dt)  # convert sea level rise rate to m/day

                # water depth
                self.get_water_depth()

                # # calculate ponding.  Originally this is calculated at the first step.
                # self.update_ponding(dt)
                #
                # # vegetation
                # self.update_vegetation(round)
                #
                # # roughness
                # self.update_roughness()

                # roughness will come simply from a hyperparameter.

                # tidal_flow
                self.update_flow()
                # bed morphology
                maxdeltaz, maxup = self.update_morphology(dt)
            dti = dti + dt
            dt = np.minimum((dt * 2), (np.maximum(0, dto - dti)))

