#!/usr/bin/env python
# coding: utf-8

# # Modeling marsh herbivor effects on marsh surviability.
# 
# *(modified from Greg Tucker, August 2020)*
# 
# This tutorial takes the parts that have previously been demonstrated in the marsh grass herbivory and tidal_flow_with_veg tutorials.  We will use the random herbivory of the agent based model to stear the roughness calculations in the tidalFlowCalculator and examin how these changes lead to reponsense in the marsh morphology. 

# # Creating the domain
# 
# Previously we have made an numerically generated domain for the purposes of the example. However, we will now import a digital elevation model (DEM) to act as the basis of our Landlab model grid.

# In[1]:

# # Setup herbivory
# 
# We will use the agent based random herbivory model from the marsh grass herbivory example to drive biomass reduction. Grazed vegetation will have lower roughness values and will be subject to reduced accretion. To do this we have modified the MarshMorpho2D script to accept the hervibory function. Vegetation coverage is calculated by the Coastal Wetland Equilibrium Model. The resulting area is then augmented by grazing before being sent to the roughness calculator.

# In[6]:


from mesa import Model
from collections import defaultdict

from mesa import Agent
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivationByType
from typing import Callable, Optional, Type

class RandomActivationByTypeFiltered(RandomActivationByType):
    """
    A scheduler that overrides the get_type_count method to allow for filtering
    of agents by a function before counting.

    Example:
    >>> scheduler = RandomActivationByTypeFiltered(model)
    >>> scheduler.get_type_count(AgentA, lambda agent: agent.some_attribute > 10)
    """

    def get_type_count(
        self,
        type_class: Type[Agent],
        filter_func: Optional[Callable[[Agent], bool]] = None,
    ) -> int:
        """
        Returns the current number of agents of certain type in the queue
        that satisfy the filter function.
        """
        if type_class not in self.agents_by_type:
            return 0
        count = 0
        for agent in self.agents_by_type[type_class].values():
            if filter_func is None or filter_func(agent):
                count += 1
        return count

class RandomWalker(Agent):
    """
    Class implementing random walker methods in a generalized manner.

    Not indended to be used on its own, but to inherit its methods to multiple
    other agents.

    """

    grid = None
    x = None
    y = None
    moore = True

    def __init__(self, unique_id, pos, model, moore=True):
        """
        grid: The MultiGrid object in which the agent lives.
        x: The agent's current x coordinate
        y: The agent's current y coordinate
        moore: If True, may move in all 8 directions.
                Otherwise, only up, down, left, right.
        """
        super().__init__(unique_id, model)
        self.pos = pos
        self.moore = moore

    def random_move(self):
        """
        Step one cell in any allowable direction.
        """
        # Pick the next cell from the adjacent cells.
        next_moves = self.model.grid.get_neighborhood(self.pos, self.moore, True)
        next_move = self.random.choice(next_moves)
        # Now move:
        self.model.grid.move_agent(self, next_move)


class Herbivore(RandomWalker):
    """
    A herbivore that walks around, reproduces (asexually) and eats grass.

    The init is the same as the RandomWalker.
    """

    energy = None

    def __init__(self, unique_id, pos, model, moore, energy=None, food_preference=400):
        #Herbivore(self.next_id(), (x, y), self, True, energy)
        super().__init__(unique_id, pos, model, moore=moore)
        self.energy = energy
        self.food_preference = food_preference

    def step(self):
        """
        A model step. Move, then eat grass and reproduce.
        """
        self.random_move()
        living = True

        if self.model.grass:
            # Reduce energy
            self.energy -= 15

            # If there is grass available, eat it
            this_cell = self.model.grid.get_cell_list_contents([self.pos])
            grass_patch = [obj for obj in this_cell if isinstance(obj, GrassPatch)][0]
            if isinstance(grass_patch.biomass_proportion, float) == False and isinstance(grass_patch.biomass_proportion, int) == False :
                print('STOP')
            if grass_patch.biomass_proportion > self.food_preference:
                self.energy += self.model.herbivore_gain_from_food
                grass_patch.fully_grown = False
                grass_patch.biomass -= self.model.grass_loss_from_grazing

            # Death
            if self.energy < 0:
                # self.model.grid._remove_agent(self.pos, self)
                self.model.grid.remove_agent(self)
                self.model.schedule.remove(self)
                living = False

        if living and self.random.random() < self.model.herbivore_reproduce:
            # Create a new sheep:
            if self.model.grass:
                self.energy /= 2
            juvenile = Herbivore(
                self.model.next_id(), self.pos, self.model, self.moore, self.energy
            )
            self.model.grid.place_agent(juvenile, self.pos)
            self.model.schedule.add(juvenile)


class GrassPatch(Agent):
    """
    A patch of grass that grows at a fixed rate and it is eaten by a herbivore
    """

    def __init__(self, unique_id, pos, model, fully_grown, countdown, biomass, max_biomass):
        """
        Creates a new patch of grass

        Args:
            grown: (boolean) Whether the patch of grass is fully grown or not
            countdown: Time for the patch of grass to be fully grown again
        """
        super().__init__(unique_id, model)
        self.countdown = countdown
        self.fully_grown = fully_grown
        self.max_biomass = max_biomass
        self.biomass_proportion = biomass
        self.biomass = self.biomass_proportion * self.max_biomass
        if self.fully_grown:
            self.percent_of_max = 1.0
            self.biomass = self.biomass_proportion * self.max_biomass
        else:
            self.percent_of_max =  (self.model.grass_regrowth_time - self.countdown) / self.model.grass_regrowth_time
            self.biomass = (self.biomass_proportion * self.percent_of_max) * self.max_biomass

        self.pos = pos

    def step(self):
        if not self.fully_grown:
            if self.countdown <= 0:
                self.fully_grown = True
                self.countdown = self.model.grass_regrowth_time
                self.percent_of_max = 1
                self.biomass = self.biomass_proportion * self.max_biomass
            else:
                self.percent_of_max = (self.model.grass_regrowth_time - self.countdown) / self.model.grass_regrowth_time
                self.biomass = self.biomass_proportion * self.max_biomass
                self.countdown -= 1



"""
Wolf-Sheep Predation Model
================================

Replication of the model found in NetLogo:
    Wilensky, U. (1997). NetLogo Wolf Sheep Predation model.
    http://ccl.northwestern.edu/netlogo/models/WolfSheepPredation.
    Center for Connected Learning and Computer-Based Modeling,
    Northwestern University, Evanston, IL.
"""


class HerbivoreGrass(Model):
    """
    Herbivore Grass Model
    """

    initial_herbivores = 10

    herbivore_reproduce = 0.04

    grass = False
    grass_regrowth_time = 2
    herbivore_gain_from_food = 4

    verbose = False  # Print-monitoring

    description = (
        "A model for simulating wolf and sheep (predator-prey) ecosystem modelling."
    )

    def __init__(
        self,
        initial_herbivores=10,
        herbivore_reproduce=0.02,
        grass=False,
        grass_regrowth_time=1,
        grass_loss_from_grazing=10,
        grass_max_biomass=700,
        herbivore_gain_from_food=50,
        verbose=False,
        biomass = None,
        gridShape = None
    ):
        """
        Create a new Wolf-Sheep model with the given parameters.

        Args:
            initial_sheep: Number of sheep to start with
            initial_wolves: Number of wolves to start with
            sheep_reproduce: Probability of each sheep reproducing each step
            wolf_reproduce: Probability of each wolf reproducing each step
            wolf_gain_from_food: Energy a wolf gains from eating a sheep
            grass: Whether to have the sheep eat grass for energy
            grass_regrowth_time: How long it takes for a grass patch to regrow
                                 once it is eaten
            herbivore_gain_from_food: Energy herbivore gain from grass, if enabled.
        """
        super().__init__(verbose=verbose)
        # Set parameters


        height, width = gridShape
        self.height = height
        self.width = width
        self.initial_herbivores = initial_herbivores
        self.herbivore_reproduce = herbivore_reproduce
        self.grass = grass
        self.grass_regrowth_time = grass_regrowth_time
        self.grass_loss_from_grazing = grass_loss_from_grazing
        self.herbivore_gain_from_food = herbivore_gain_from_food
        self.verbose = verbose
        self.biomass_proportion = biomass

        self.schedule = RandomActivationByTypeFiltered(self)
        self.grid = MultiGrid(self.height, self.width, torus=True)
        self.datacollector = DataCollector(
            {
                "Herbivores": lambda m: m.schedule.get_type_count(Herbivore),
                "Grass": lambda m: m.schedule.get_type_count(
                    GrassPatch, lambda x: x.fully_grown
                ),
            }
        )
        import warnings
        warnings.filterwarnings('ignore')
        # Create herbivores:
        for i in range(self.initial_herbivores):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            while x >= self.width -10:
                x = self.random.randrange(self.width)
            while y >= self.height -10:
                y = self.random.randrange(self.height)
            energy = self.random.randrange(2 * self.herbivore_gain_from_food)
            herbivore = Herbivore(
                self.next_id(),
                (x, y),
                self,
                True,
                energy,
                food_preference=350
                )
            self.grid.place_agent(herbivore, (x, y))
            self.schedule.add(herbivore)

        # Create grass patches
        if self.grass:
            for agent, (x, y) in self.grid.coord_iter():
                fully_grown = self.random.choice([True, False])

                if fully_grown:
                    countdown = self.grass_regrowth_time
                else:
                    countdown = self.random.randrange(self.grass_regrowth_time)

                # if x >= self.width:
                #     x = self.width-1
                # if y >= self.height:
                #     y = self.height-1
                posBiomass = self.biomass_proportion[x, y]
                patch = GrassPatch(self.next_id(), (x, y), self, fully_grown, countdown, posBiomass, grass_max_biomass)# grass_max_biomass)
                self.grid.place_agent(patch, (x, y))
                self.schedule.add(patch)

        self.running = True
        self.datacollector.collect(self)



    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)
        if self.verbose:
            print(
                [
                    self.schedule.time,
                    self.schedule.get_type_count(Herbivore),
                ]
            )

    def run_model(self, step_count=200):
        if self.verbose:
            print("Initial number herbivores: ", self.schedule.get_type_count(Herbivore))

        for i in range(step_count):
            self.step()

        if self.verbose:
            print("")
            print("Final number herbivores: ", self.schedule.get_type_count(Herbivore))




def graze_grid(model):
    grass_map = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, (x, y) = cell
        for agent in cell_content:
            if type(agent) is GrassPatch:
                veg = agent.biomass  # 700 is the maximum biomass available.
                # for the purposes of this example we cant have veg =0 but it can equal a very small number
                if veg == 0:
                    veg = 0.000001
                grass_map[x][y] = veg

    return grass_map



# # pymm2d integration
# While it is a bit messy we are going to include the entireity of the python MM2D model in this package to show where we will be modifying the code.

# In[7]:

# pymm2d

"""pyMarshMorpho2D model."""
import numpy as np
import pandas as pd
from landlab import Component
from landlab.components import TidalFlowCalculator
import math
import sys

np.set_printoptions(threshold=sys.maxsize)
# import warnings
import scipy
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
import numpy as np

# warnings.filterwarnings("error")


class mainEvolution(Component):
    """Simulate tidal marsh evolution."""

    _name = "mainEvolution"

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
                 tidal_range=1.225296,  # tidal range taken from Atlantic city NOAA tide gauge.
                 tidal_range_for_veg=1.225296,
                 roughness_with_veg=0.1,
                 roughness_without_veg=0.02,
                 tidal_period=12.5 / 24,
                 model_domain=None,
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

        super(mainEvolution, self).__init__(grid)
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
        # Set parameter values
        self._mean_sea_level = -0.122  # m NAVD88
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
        self._sea_SSC = 60 / 1000  # 40/1000; #Sea boundary SSC for mud [g/l]
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

        # ponding
        self._zpondcr = -0.2  # P.Trange/4. base of new pond formation with respect to MSL
        self._minponddepth = 0.1  # min ponding depth
        self._maxdpond = np.maximum(0.2, np.maximum(self._minponddepth * 1.1, 0.15 * self._tidal_range))
        self._Epondform = 0.4 * 10 ** -5  # 4*10^-4/10 probabiliy of new pond formation per year (area/area)
        self._pondLoss = 0
        self._zntwrk = (
                                   self._tidal_range / 2) * 0.5  # depth above msl that defines the channel network.  the smaller the harder to drain!
        self._aPEXP = 0.015 * 10  # isolated pond expansion rate m/yr
        self._pondDeepRate = 0.003  # m/yr

        # waves
        self._hwSea_lim = 0.2  # 0.5; %limiter water deth of sea waves %THIS IS USED TO FIND THE "EDGE"%NEEDS TO BE LARGER THAN KO!!!!
        self._dBlo = 0  # The original value was listed in vegetation parameters

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
        from scipy.sparse.linalg import LinearOperator
        self._dx = dx
        self._dt = dt

        # matlab script function z=bedcreepponds(z,A,Active,Yreduction,crMUD,crMARSH,dx,dt,VEG,S,Qs,rbulk2,alphaMUD);
        # values bein input in original matlab script bedcreepponds(z, A, Active, A * 0 + 1, crMUD, crMARSH, dx, dt, VEG, S, Qs2, rbulk2, alphaMUD)
        # A(Active == 0) = 0 # not sure this is necessary in the python script
        self._Qs = self._Qs / self._rbulk

        # setup a placeholder for z
        # zhld = self._elev.copy()

        Yreduction = (self._model_domain * 0) + 1

        creep = self._model_domain * 0
        creep[self._vegetation == 0] = self._crMUD + (self._alphaMUD * 3600 * 24 * self._Qs[self._vegetation == 0])
        creep[self._vegetation == 1] = self._crMARSH

        D = (creep) / (self._dx ** 2) * self._dt  # yreduction

        G = self._elev * 0
        p = np.where(self._model_domain == 1, True, False)
        G[p] = range(0, len(self._index[p]))  # self._index[p]
        rhs = self._elev[p]

        Spond = self._elev * 0  # THIS IS ONLY FOR TESTING! REMOVE WHEN ACTUAL PONDS EXIST! # this is the location of the ponds 5 is just there for testing.
        #
        # S = G * 0  # this is no longer pond locations.  Not sure why this is done this way in the matlab code...

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

            # if max(q[a]) == 9902:
            #     print("Stop")
            # numerical array that corresponds to the index values covered by water
            parray = tmp

            # 8-28-2023 updated logic to speed up processing
            ptmpArray = np.array(list(range(len(tmp))))[a]  # updated 9-29-2023
            cls = np.where(a[ptmpArray] == True, np.where(self._model_domain[q[a]] != -1, True, False), False)
            a[ptmpArray[cls]] = True
            a[ptmpArray[cls == False]] = False

            # cls = np.where(a[parray[a]] == True, np.where(self._model_domain[q[a]] != -1, True, False), False)
            # a[parray[a][cls]] = True
            # a[parray[a][cls == False]] = False

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
        ds2 = sparse.csc_array((s, (ilog, jlog)))  # , shape=(len(self._index[p]), len(self._index[p])))
        try:
            P = scipy.sparse.linalg.spsolve(ds2,
                                            rhs)  # was working with .lsqr # look into skcuda and cusparse to accelerate the model computation.
        except:
            if self._printAll != None:
                print("Bedcreep matrix solution was singular. Reverting to lsqr to solve matrix inversion")
            P = scipy.sparse.linalg.lsqr(ds2, rhs, iter_lim=5000)[0]
        # zhld[G>0] = P[G[G>0]]

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
        # for testing we will trun (True, False, True, True
        # self._tidal_flow = TidalFlowCalculator(self._grid, mean_sea_level=self._mean_sea_level, tidal_range=self._tidal_range, tidal_period = self._tidal_period*24*3600, roughness=0.02).calc_tidal_inundation_rate()
        roughnessArray = map_mean_of_link_nodes_to_link(self._grid, self._roughness)
        flood = TidalFlowCalculator(self._grid, mean_sea_level=self._mean_sea_level,
                                    tidal_range=self._tidal_range, tidal_period=self._tidal_period * 24 * 3600,
                                    roughness=roughnessArray, min_water_depth=0.1, scale_velocity=1)
        #
        # flood = flowCal.TidalFlowCalculator(self._grid, mean_sea_level=self._mean_sea_level,
        #                           tidal_range=self._tidal_range, tidal_period = self._tidal_period*24*3600,
        #                           roughness=roughnessArray, min_water_depth=0.1, scale_velocity=1)

        flood.run_one_step()
        # # linear method for calculating tidal flood duration
        # self._tidal_flow = flood.calc_tidal_inundation_rate()
        # vertical_inundation_rate = flood._tidal_range / flood._tidal_half_period
        # time_spent_flooded = (flood._water_depth / vertical_inundation_rate) * 2 # seconds spent flooded
        # # calculate the percent of time spent flooded
        # percent_time_flooded = time_spent_flooded / 86400
        # # filter anything above 1 is just 1
        # percent_time_flooded[percent_time_flooded > 1] = 1
        #
        # # add the variable to self to be used later in mapping and integration with the MEM
        # self._percent_time_flooded = percent_time_flooded * 100
        # # factor out the minimum water depth issue
        # self._percent_time_flooded[self._percent_time_flooded == (((0.1 / vertical_inundation_rate) * 2) / 86400 * 100)] = 0
        # self.grid.at_node['percent_time_flooded'] = self._percent_time_flooded

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
        # Currently the issue is with the sedtran.m implementation.

        ###### erosion base values for all types of erosion (currently only one type of erosion being used.

        # variables to figure out
        # taucr = a user defined value as part of the mud paramaters in the original code it equals 0.2
        # taucrVEG = a user defined value as aprt of the vegetation parameters "Critical Shear Stress for vegetated areas. The original value was 0.5
        # MANN = self._roughness
        # VEG = self._vegetation
        # U = Is a returned value from the tidalFlow Function.  I think it is the tidal velocity.
        # me = P.me=0.1*10^-4*24*3600# a defined value in the parameters for mud P.me=0.1*10^-4*24*3600;  #per day!!!
        # h = Is water depth self._water_depth
        # we are going to have to define a domain. someway.

        # record the starting elevation
        origz = np.copy(self._elev)

        # exclude cells that cant get sediment as they are above water.
        # p = np.where(self._model_domain > 0, np.where(self._water_depth <= 0, False, True),
        #              False)  # original working code. however it didn't remove above water areas from the calculation.
        p = np.where((self._model_domain > 0) & (
                    self._water_depth_ho > 0))  # Thomas add this line 9-28-2023 to manage unwetted areas.,
        # np.where(self._water_depth <=self._min_water_depth, False, True), False) # double check this logic.  This seems backwards

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
            # print(math.sin(i / ncyc * math.pi / 2))
            # print(Utide[1])
            try:
                watPow = self._water_depth ** (-1 / 3)  # changed to fully wetted depth
            # if watPow.any() == np.inf:
            #     print("There are infs")
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
        # (self._DoMUD * 24 * 3600 + self._DiffS * self._tidal_period / 2 * np.abs(self._Ux * self._Ux) * (24 * 3600) ** 2) / (dx ** 2) * self._water_depth
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
                # aIndex = np.where(col + 1 < M, tmp, np.nan)
                # aIndex = np.array(aIndex[~np.isnan(aIndex)]).astype('int')
                a = np.where(col + 1 < M, True, False)
                q = tmp + 1
            if k == -N:
                # aIndex = np.where(col - 1 >= 0, tmp, np.nan)
                # aIndex = np.array(aIndex[~np.isnan(aIndex)]).astype('int')
                a = np.where(col - 1 >= 0, True, False)
                q = tmp - 1  # originally tmp was tmp[p]
            if k == -1:
                # aIndex = np.where(row - 1 >= 0, tmp, np.nan)
                # aIndex = np.array(aIndex[~np.isnan(aIndex)]).astype('int')
                a = np.where(row - 1 >= 0, True, False)
                q = tmp - M
            if k == 1:
                # aIndex = np.where(row + 1 < N, tmp, np.nan)
                # aIndex = np.array(aIndex[~np.isnan(aIndex)]).astype('int')
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
            # aIndex_Match = np.in1d(aIndex, q[a])
            # qaIndex_Match = np.in1d(q[a], aIndex)
            # a[aIndex_Match] = np.where(qaIndex_Match == True, np.where(self._model_domain[q[a]] != 1, np.where(self._model_domain[q[a]] != 2,np.where(self._model_domain[q[a]] == 10, True, False), True), True), False)

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

            ########################################################################################################
            # matlab code currently not implemented
            # #river flow component
            # if computeriver==1
            # if (k==N);UR=URy(p(a));up=find(UR>0);F=UR(up);end #East-west
            # if (k==-N);UR=URy(p(a));up=find(UR<0);F=-UR(up);end
            # if (k==1);UR=URx(p(a));up=find(UR>0);F=UR(up);end  #North-south
            # if (k==-1);UR=URx(p(a));up=find(UR<0);F=-UR(up);end
            # value(up)=value(up)+F*3600*24/dx;
            # end
            #
            #######################################################################################################
            # residualcurrents = 1
            # if residualcurrents == 1:
            #     # print("Calculating residual currents")
            #     # tidal residual currents and transport.
            #     # (I imposed no residual currents are the open boundary to avoid
            #     # calculating the fluxes to get the mass balance at 100#)
            #     if k == N:
            #         UR = self._Ux[parray[a]]
            #         up = np.where(UR > 0, True, False)
            #         F = UR[up]
            #     if k == -N:
            #         UR = self._Ux[parray[a]]
            #         up = np.where(UR < 0, True, False)
            #         F = -UR[up]
            #     if k == 1:
            #         UR = self._Uy[parray[a]]
            #         up = np.where(UR > 0, True, False)
            #         F = UR[up]
            #     if k == -1:
            #         UR = self._Uy[parray[a]]
            #         up = np.where(UR < 0, True, False)
            #         F = -UR[up]
            #     value[up] = value[up] + F * 3600 * 24 / dx
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

        ################################################################################
        # #sea boundary
        # a=find(A(p)==2);#find the co b.c.
        # settling(a)=0;#do not settle in the b.c. (to not change the SSC)
        # S(p(a))=1;#to impose the b.c.
        #
        # #river boundary
        # if MUD==1; #if mud, handlew this as an imposed SSC
        # a=find(A(p)==10);#find the co b.c.
        # settling(a)=0;#do not settle in the b.c. (to not change the SSC)
        # S(p(a))=1;#to impose the b.c.
        # end
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
            nacount = np.isnan(P)
            if len(nacount[nacount==True]) == len(P):
                P = scipy.sparse.linalg.lsqr(ds2, rhs, iter_lim=5000)[0]
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

        # self._KBTOT = self._KBTOT + sum(self._vegetation[self._model_domain == 1]) * self._accretion.mean()
        #######################################################################################################
        # ##################### BED EVOLUTION DIVERGENCE ###############################################

        # fix this section to add in bed creep ponds... Pond creep seems to do more than just for ponds.....
        # EVOLUTION OF z
        znew = self._elev.copy()
        # z=znew;  # NEED TO RE-UDPATED FROM THE Y1
        self._Qs = E / (self._ws2 * 3600 * 24) * self._U * np.maximum(0,
                                                                      self._water_depth)  # kg/m/s #[hcorrected]=getwaterdepth(Trange,Hsurge,msl,z,kro,hpRIV);  #VEG=(z-msl)>dBlo;
        # znew=bedcreepponds(z,A,Active,A*0+1,crMUD,crMARSH,dx,dt,VEG,S,Qs2,rbulk2,alphaMUD)  # MUD CREEP  MARSH
        znew = mainEvolution.bedcreepponds(self, dx=dx, dt=dt)
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

        # organic accretion has not been added yet.

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

    def update_ponding(self, dt): # This function still needs to be tested against the matlab version.
        import numpy as np
        import random

        self._dt = dt

        # print("Computing ponding dynamics")

        dx = self._unit * 2  # modifier with spatial units

        zsill = self._tidal_half_range
        # pondformation
        z = self._elev.copy()

        p = np.where(self._model_domain == 1, True, False)

        pIndex = self._index[p]

        p_mask = np.where(p == True, True, False)
        # p_mask = np.array(p_mask).astype(int)
        dz = np.maximum((z - self._zpondcr),
                        0)  # base level of the pond. was dz = np.maximum((z[p_mask] - self._zpondcr), 0) Thomas changed this 12-26-2023
        dz = np.minimum(self._maxdpond, dz)  # maximum scour depth

        # only create new ponds where new ponds can be created
        a = np.where(dz[p_mask] > 0, np.where(z[p_mask] < zsill, True, False), False)
        # turn a into an index
        a = pIndex[a]

        if len(a) > 0:
            r = np.random.rand(len(a), 1)
            a = np.reshape(a, (len(a), 1))
            aa = a[r < (
                        self._Epondform * self._dt * dx ** 2)]  # Thomas modified the dt/365 statement as that was already done.
            # ^ additional changes made from the existing code of (self._Epondform * self._dt * dx**2)
            if len(aa) > 0:
                for i in range(0, len(aa)):
                    try:
                        dz[aa[i]]
                    except:
                        print("problem")

                    if dz[aa[i]] > 0:
                        z[self._index[aa[i]]] = z[self._index[aa[i]]] - dz[aa[i]]  # changed self._index from p_index 12-26-2023
                        # self._pondLoss = self._pondLoss + dz(aa[i])
        deltaZ = self._elev.copy()

        # Find isolated ponds
        Z = self._elev.copy() - self._mean_sea_level

        ZB = self._model_domain.copy()
        ZB[:] = 0
        ZB = np.reshape(ZB, (self._grid.shape))
        ZB[0, :] = 1
        ZB[-1, :] = 1
        ZB[:, -1] = 1
        ZB[:, 0] = 1
        ZB = np.reshape(ZB, (self._model_domain.shape))
        Z[np.where(ZB == 1, True, False)] = 10
        Z[self._model_domain == 10] = 10

        Z = np.maximum(self._zntwrk, Z)
        Z = np.minimum(zsill, Z)  # flooding sill height

        ZZ = Z.copy()
        ZZ[np.isnan(ZZ)] == 0

        ZZ = np.reshape(ZZ, (self._grid.shape))

        # double check the below code as it was AI generated.
        ZZ = np.minimum(ZZ, np.minimum(np.concatenate((ZZ[:, 0:1] * 0, ZZ[:, :-1]), axis=1),
                                       np.minimum(np.concatenate((ZZ[0:1, :] * 0, ZZ[:-1, :]), axis=0),
                                                  np.minimum(np.concatenate((ZZ[:, 1:], ZZ[:, -1:] * 0), axis=1),
                                                             np.concatenate((ZZ[1:, :], ZZ[-1:, :] * 0), axis=0)))))
        ZZ = np.reshape(ZZ, (self._model_domain.shape))

        DIF = abs(ZZ - Z)

        # zero out edge nodes
        DIF = np.reshape(DIF, (self._grid.shape))
        DIF[0, :] = 1
        DIF[-1, :] = 1
        DIF[:, -1] = 1
        DIF[:, 0] = 1
        DIF = np.reshape(DIF, (self._model_domain.shape))

        DIF[DIF > 0.001] = ZZ[DIF > 0.001] - abs(self._elev[DIF > 0.001])
        S = np.reshape(np.where(DIF > self._minponddepth, True, False), self._grid.shape)
        AC = np.where(self._elev < self._zntwrk, np.where(DIF <= self._minponddepth, True, False), False)

        # set border to zero
        S[0, :] = False
        S[-1, :] = False
        S[:, -1] = False
        S[:, 0] = False
        # ponded nodes
        pondNodes = self._grid.nodes[S]
        if len(pondNodes) > 0:
            allNodes = []
            sourceElevation = []
            for pn in pondNodes:
                adjNodes = self._grid.active_adjacent_nodes_at_node[(pn),]
                for adj in adjNodes:
                    allNodes.append(adj)
                    sourceElevation.append(self._elev[pn])

            allClean = [x for x in allNodes if x >= 0]
            sourceClean = [sourceElevation[x] for x in range(len(allNodes)) if allNodes[x] >= 0]
            ztmp = self._elev.copy()
            ztmp[allClean] = sourceClean

            # find unique values between the allClean list and the original pondNodes list. The unique values are the ones that will erode.
            uni = list(set(allClean).symmetric_difference(set(pondNodes)))
            # the only nodes allowed to erode are marsh nodes. So a statement will have to be added to restrict where
            # the ponding erosion is allowed to happen.

            zUpdate = self._elev.copy()
            zUpdate[uni] = ztmp[uni]

            deltaYpondExp = self._elev - zUpdate

            ####################################################################################
            self._elev = self._elev - deltaYpondExp  # apply the change values to the elevation.
            ####################################################################################

            # pond deepening

            zDeepen = self._elev.copy()

            for i in pondNodes:
                dz = self._pondDeepRate * dt
                zDeepen[i] = zDeepen[i] - dz

            deltaY = self._elev - zDeepen

            ####################################################################################
            self._elev = self._elev - deltaY  # apply the pond deepening to the elevation.
            ####################################################################################


    def update_waves(self, angle, ndir): # this function also calls the waveErosionFunction
        import numpy as np
        from waveErosionFunctions import cumSumReset, cumSumResetEXTRA, cumSumResetEXTRALateral, YeV, wavek, \
            diffuseEdgeSediments
        # SeaWaves ##################################################
        MASK = self._elev.copy()
        MASK[:] = 0
        # filter the mask layer
        MASK[self._fully_wet_depth < self._hwSea_lim] = 0
        MASK[self._modeldomain == 0] = 0
        MASK[self._vegetation == 1] = 0
        MASK[self._elev >= self._dBlo] = 0

        # setup A as a copy of the model domain
        A = self._modeldomain.copy()

        # [Uwave_sea,Tp_sea,Hsea,Fetch,kwave,PWsea]=SeaWaves(h,angleWIND,hwSea_lim,Trange,wind,MASK,64,dx) # 72,dx

        # Thomas inserted code here from the SeaWaves function
        # angle=0;
        Um = self._elev.copy
        Um[:] = 0
        TP = self._elev.copy
        TP[:] = 0
        HS = self._elev.copy
        HS[:] = 0

        dx = self._unit * 2  # this involves the spatial units used for computation

        extrafetch = 5000  # [m}
        Lbasin = 1000 / dx
        Fetchlim = np.maximum(50, dx * 2)  # dx*2;#600;#dx*2*10
        dlo = self._hwSea_lim  # minimum water depth to calculate wave. below this you don't calculate it

        extra = 1  # if (angle<90 | angle>270);extra=1;else;extra=1;end

        # The standard way of fetch calculation
        if extra == 0:
            # F=calculatefetch(MASK,ndir,dx,angle) # put in this function

            # function F=calculatefetch(A,ndir,dx,angle)
            angle = angle - 180
            [N, M] = self._grid.shape()
            F = np.zeros(N, M)
            A[np.isnan(A) == True] = 0  # any NaN is a boundary, that is, a 0

            # Every boundary is a wall, need to do this so the fetch does not warp around!
            A[1, 1:] = 0
            A[-1, 1:] = 0
            A[1:, 1] = 0
            A[1:, -1] = 0

            di = 1 + np.mod(math.floor(angle / 360 * ndir + 0.5), ndir)
            alfa = (di - 1) / ndir * 2 * np.pi
            m = np.maximum(abs(np.cos(alfa)), abs(np.sin(alfa)))
            # the below code was AI corrected and has not been verified as I do not have a working MATLAB install
            if (di <= (ndir * 1 / 8) | (di > ndir * 3 / 8 & di <= ndir * 5 / 8) | di > ndir * 7 / 8):
                IND = (1 + np.mod(np.round(np.dot(np.arange(1, N + 1)[:, None], np.cos(alfa)) / m), N) @ np.ones(
                    (1, M))).astype(int) + np.mod(np.ones((N, 1)) @ np.arange(1, M + 1)[None, :] + np.round(
                    np.dot(np.arange(1, N + 1)[:, None], np.sin(alfa)) / m) @ np.ones((1, M)) - 1, M) * N
            else:
                IND = (1 + np.mod(np.ones((M, 1)) @ np.arange(1, N + 1) + np.round(
                    np.arange(1, M + 1)[:, None] * np.cos(alfa) / m) @ np.ones((1, N))) - 1) + np.mod(
                    np.round(np.arange(1, M + 1)[:, None] * np.sin(alfa) / m) @ np.ones((1, N)), M) * N

            F = cumSumReset(A)
            F[IND] = F / m * dx

            # at the boundary, impose the fetch of the nearby cell
            F[1, 1:] = F[2, 1:]
            F[-1, 1:-1] = F[-2, 1:-1]
            F[1:, 1] = F[1:, 2]
            F[1:, -1] = F[1:, -2]

        # For the idealize basin
        # extrafetch=10000;#[m}
        if extra == 1:  # this section performs the "calculatefetchWITHEXTRAS function
            # F=calculatefetchWITHEXTRAS(MASK,ndir,dx,angle,extrafetch,Lbasin,h-0.1-range/4)
            angleo = angle
            # function F=calculatefetchWITHEXTRAS(A,ndir,dx,angle,extrafetch,Lbasin,h)
            # angle=315-180-0.
            angle = angle - 180
            [N, M] = self._grid.shape()
            F = np.zeros(N, M)
            A[np.isnan(A) == True] = 0  # any NaN is a boundary, that is, a 0

            # Every boundary is a wall, need to do this so the fetch does not warp around!
            A[1, 1:] = 0
            A[-1, 1:] = 0
            A[1:, 1] = 0
            A[1:, -1] = 0

            # di=1+mod(floor(angle/360*ndir+0.5),ndir);
            di = 1 + np.mod(math.floor(angle / 360 * ndir + 0.5), ndir)
            alfa = (di - 1) / ndir * 2 * np.pi
            m = np.maximum(abs(np.cos(alfa)), abs(np.sin(alfa)))
            if (di <= (ndir * 1 / 8) | (di > ndir * 3 / 8 & di <= ndir * 5 / 8) | di > ndir * 7 / 8):
                IND = (1 + np.mod(np.round(np.arange(1, N + 1)[:, None] * np.cos(alfa) / m) @ np.ones((1, M))),
                       N) + np.mod(np.ones((N, 1)) @ np.arange(1, M + 1)[None, :] + np.round(
                    np.arange(1, N + 1)[:, None] * np.sin(alfa) / m) @ np.ones((1, M)) - 1, M) * N
            else:
                IND = (1 + np.mod(np.ones((M, 1)) * np.arange(1, N + 1) + np.round(
                    np.arange(1, M + 1).reshape(-1, 1) * np.cos(alfa) / m) * np.ones((1, N)) - 1, N)) + (
                          np.mod(np.round(np.arange(1, M + 1).reshape(-1, 1) * np.sin(alfa) / m) * np.ones((1, N)),
                                 M)) * N
            # F(IND)=cumsumreset(A(IND),1)/m*dx

            # error
            if angleo < 0 | angleo > 360:
                A = 'error'

            # #if (angleo<44 | angleo>316)
            # if (angleo<134 | angleo>226)
            # F(IND)=cumsumresetEXTRA(A(IND),extrafetch/dx)/m*dx
            # else
            # F(IND)=cumsumreset(A(IND))/m*dx
            # end

            padding = Lbasin * 2  # must be larger than fetchlim!!!
            if (angleo < 45 | angleo > 315):
                F[IND] = cumsumresetEXTRA(A[IND], extrafetch / dx) / m * dx
                if angleo < 44:
                    # ll=length(F(2:end-1-floor(N*0.5),end))
                    # F(2+floor(N*0.5):end-1,end-padding:end)=2*extrafetch*[1:ll]'/ll*ones(padding+1,1)'
                    # F(end,end-padding:end)=extrafetch
                    a = np.where(h[:, -1] < 0)
                    if a > 0:
                        Lside = a[-1] - 1
                    else:
                        Lside = 1
                    F[Lside:, -100:] = np.maximum(extrafetch, F[Lside:,
                                                              -100:])  # TOGLI PER EVITRARE IL FETCH ALTO NELLE MUDFLAT SEAWARD
                    # F(Lside:end,end-10:end)=max(extrafetch,F(Lside:end,end-10:end))
                    # F(:,end)=extrafetch
                else:
                    # ll=length(F(2:end-1-floor(N*0.5),1))
                    # F(2+floor(N*0.5):end-1,1:1+padding)=2*extrafetch*[1:ll]'/ll*ones(padding+1,1)'
                    # F(1,end-padding:end)=extrafetch
                    # F(1,:)=extrafetch
                    a = np.where(h[:, 1] < 0)
                    if a > 0:
                        Lside = a[-1] - 1;
                    else:
                        Lside = 1
                    F[Lside:, 0:100] = np.maximum(extrafetch, F[Lside:,
                                                              0:100])  # TOGLI PER EVITRARE IL FETCH ALTO NELLE MUDFLAT SEAWARD
                    # F(Lside:end,1:11)=max(extrafetch,F(Lside:end,1:11))

            elif (angleo >= 45 and angleo < 90):  # 134)
                a = np.where(h[:, -1] < 0)
                if a > 0:
                    Lside = N - a[-1] - 1
                else:
                    Lside = N - 1
                F[IND] = cumsumresetEXTRAlateral1(A(IND), extrafetch / dx, Lside) / m * dx
                # (IND)=cumsumreset(A(IND))/m*dx
                # a=find(MASK(
                F[-Lside:, :] = extrafetch  # offshore boudnary
                # ll=length(F(2:end-1-N/2,end));
                # F(2+N/2:end-1,end-padding:end)=extrafetch*[1:ll]'/ll*ones(padding+1,1)'

            elif(angleo > 275 and angleo <= 315):
                a = np.where(h[:, 1] < 0)
                if a > 0:
                    Lside = N - a[-1] - 1
                else:
                    Lside = N - 1
                F[IND] = cumsumresetEXTRAlateral1(A(IND), extrafetch / dx, Lside) / m * dx
                # F(IND)=cumsumreset(A(IND))/m*dx;
                F[-Lside:, :] = extrafetch;  # offshore boudnary
                # ll=length(F(2:end-1-N/2,1));
                # F(2+N/2:end-1,1:1+padding)=extrafetch*[1:ll]'/ll*ones(padding+1,1)'
        else:
            F[IND] = cumsumreset(A[IND]) / m * dx;
        # at the boundary, impose the fetch of the nearby cell
        F[1, 1:] = F[2, 1:]
        F[-1, 1:-1] = F[-2, 1:-1]
        F[1:, 1] = F[1:, 2]
        F[1:, -1] = F[1:, -2]

        Fo = F

        #             #For all the modified ways. Creates a buffer on the side
        #             #boundaries. Just used as a mask, the actual value is not
        #             #importnat, just need to be larger than fetchlim.
        #             #####################$#$#$&^$#^$&#^$#^$#&^$$*&^#&*#$*^#$#&*$*&##$&*#$&#*$#&*
        #             #####################$#$#$&^$#^$&#^$#^$#&^$$*&^#&*#$*^#$#&*$*&##$&*#$&#*$#&*
        #             #####################$#$#$&^$#^$&#^$#^$#&^$$*&^#&*#$*^#$#&*$*&##$&*#$&#*$#&*
        #             [N,M]=size(h);
        #             Fo(2+floor(N*0.5):end-1,1:20)=9999;
        #             Fo(2+floor(N*0.5):end-1,end-20:end)=9999;
        #             ###################
        #             #####################$#$#$&^$#^$&#^$#^$#&^$$*&^#&*#$*^#$#&*$*&##$&*#$&#*$#&*

        F[Fo <= Fetchlim] = 0

        # usa questo per isolared la mudflat
        ###########################
        if extra == 1:
            MASK[-Lbasin:, :] = 1
            F[-Lbasin:, :] = extrafetch
            Fo[-Lbasin:, :] = extrafetch
        #############################

        # diffuse the fetch field
        alphadiffusefetch = 0.1  # messo 10 for the VCR wave validation 10;#0;###QUESTO ERA 1 FINO AD APRILE 23 2018!!!!!
        F = diffusefetch(MASK, F, alphadiffusefetch, dx, self._index)
        F[Fo <= Fetchlim | MASK == 0] = 0
        ###############################
        #
        # figure
        # imagesc(F)
        # colormap('jet')
        # caxis([0 max(F(:))])
        # pause
        #


        ##############################
        a = np.where(Fo > Fetchlim, np.where(h > dlo, np.where(F > 0, np.where(
            MASK == 1))))  # h>dlo & #a=find(Fo>dx*2);#h>dlo & #a=find(h>dlo)
        D = h[a]
        Ff = F[a]
        ###############################


        # ###TRCUCCOZO TO AVOID depths too small
        # hbedsheatresslim=0.5;
        # h(h<hbedsheatresslim)=hbedsheatresslim;
        # ######################################


        # [Hs,Tp]=YeV_correction(Ff,wind,D);#[Hs,Tp]=YeV(Ff,wind,min(3,D));  #TRUCCO PER EVITARE LARGE WAVES IN CHANELS
        Hs, Tp = YeV(Ff, wind, D)  # [Hs,Tp]=YeV(Ff,wind,min(3,D));  #TRUCCO PER EVITARE LARGE WAVES IN CHANELS
        HS[a] = Hs
        TP[a] = Tp
        TP[TP == 0] = 1

        # do not diffuse in cells outside the MASK
        # HS=diffusefetch(MASK,HS,alpha,dx);
        # TP=diffusefetch(MASK,TP,alpha,dx);


        # hlimbedshearstress=0.5;
        # h=max(hlimbedshearstress,h);# to reduce the bed shear stress for very small water depth


        kwave = 0 * h
        kk = wavek(1 / TP[a], h[a])  # kk=wavekFAST(1./Tp,D);
        kwave[a] = kk
        kwave[kwave == 0] = 1

        Um = pi * HS / (TP * np.sinh(kwave * h))

        cg = (2 * pi / kwave / TP) * 0.5 * (1 + 2 * kwave * h / (np.sinh(2 * kwave * h)))
        PW = cg * 1030 * 9.8 * HS ** (2 / 16)

        Um[MASK == 0] = 0
        PW[MASK == 0] = 0

        PWsea = PW
        del (PW)
        # END OF SEAWAVES MATLAB FUNCTION

        # h=[0.1:0.1:3.5];
        # [Hs,Tp]=YeV(3000,7,h);#[Hs,Tp]=YeV(Ff,wind,min(3,D));  #TRUCCO PER EVITARE LARGE WAVES IN CHANELS
        # kk=wavek(1./Tp,h);#kk=wavekFAST(1./Tp,D);
        # Um=pi*Hs./(Tp.*sinh(kk.*h));
        # ko=0.1/1000*3;
        # aw=Tp.*Um/(2*pi);
        # fw=0.00251*exp(5.21*(aw/ko).^-0.19);fw(aw/ko<pi/2)=0.3;
        # figure;plot(h,0.5*1000*0.015*Um.^2,h,0.5*1000*fw.*Um.^2)
        ########################################################


        Uwave_sea = Uwave_sea * (VEG == 0) & (S == 0)
        Hsea = Hsea * (VEG == 0) & (
                    S == 0)  # vegetation effect and no waves in isolated pond 9because we also redcued ws!!1)#Uwave_sea=Uwave_sea.*(VEG==0); Hsea=Hsea.*(VEG==0); #vegetation effect
        Uwave_sea = 0 * A
        Tp_sea = 0 * A
        Hsea = 0 * A
        Fetch = 0 * A
        QsWslope_sea = 0 * A
        # #########################################################################

        # ######################Wave-induced edge erosion############################################################
        if (computeEdgeErosionSea == 1 | computeEdgeErosionSwell == 1):  # ###MASK=0*A+1;MASK(h<hwSea_lim | A==0)=0;
            PW = A * 0
            if computeEdgeErosionSea == 1:
                PW = PW + PWsea * fTide  # Wave power reduction for hydroperiod
            [deltaz, Pedge, zOX, EdgeERz] = Edgeerosion(PW, z, aw, maxedgeheight, fox, dt, dx, MASK, A, zOX)
            z = self._elev - deltaz
            # Redistribute the eroded sediment
            EDGESED = diffuseedgesediments((A == 1), EdgeERz, 1 * h, dx)
            z = z + EDGESED
        else:
            Pedge = A * 0
            PW = 0
        ##############################################################################################

    def update_roughness(self):
        # Emily's new code
        # Update roughness value based on #
        # Initialize roughness value at each cell -- start with 0.02 which is open water roughness
        if self._withMEM == None or self._withMEM == True:
            roughness = np.ones(self._grid.shape).flatten() * 0.02

            # Candice gave the upper limit of roughness values as 0.13 for Spatina alterniflora
            # and 0.03 for mud flat
            # We can assume a weighted average between these for roughness
            # # Flat = 1-veg_cover
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


    def run_one_step(self, timeStep, round, model_domain,
                     relative_sea_level_rise_rate_mmPerYr=None, saveModelParamsFile=None):

        """Advance in time."""
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

                # calculate ponding.  Originally this is calculated at the first step.
                self.update_ponding(dt)

                # vegetation
                self.update_vegetation(round)

                # imshow_grid(self._grid, self._vegetation,  cmap = 'YlGn')
                plt.show()

                # imshow_grid(self._grid, self._elev,  cmap = 'YlGnBu')
                plt.show()

                # run herbivore model.
                # define agent based herb model
                hb = HerbivoreGrass(grass=True, gridShape=self._grid.shape, biomass = np.reshape(self._vegetation, self._grid.shape))
                hb.run_model(step_count=1)
                grazeGrid = graze_grid(hb)
                # imshow_grid(self._grid, grazeGrid,  cmap = 'YlGn')
                plt.show()
                self._vegetation = self._vegetation * grazeGrid.flatten() # adjust grazing
                # imshow_grid(self._grid, self._vegetation, cmap = 'YlGn')
                plt.show()

                #imshow_grid(self._grid, self._elev, cmap='YlGnBu')
                plt.show()

                # roughness
                self.update_roughness()

                # tidal_flow
                self.update_flow()
                # bed morphology
                maxdeltaz, maxup = self.update_morphology(dt)
            dti = dti + dt
            dt = np.minimum((dt * 2), (np.maximum(0, dto - dti)))
            # print(f'<---- updated dti value {dti}')
            # print(f'<---- updated dt value {dt}')


import numpy as np
import pandas as pd
from landlab.io import read_esri_ascii
import matplotlib.pyplot as plt
from landlab import imshow_grid
from landlab.io.esri_ascii import write_esri_ascii
# from pymarshmorpho2d import mainevolver
# from pymarshmorpho2d import mainEvolution
from tqdm import tqdm
# import rasterio as rio
import os
from glob import glob
# from pymarshmorpho2d import pyMM2DUtilities as pu

print(os.getcwd())

grid_raster = 'Gull_Island_mini_10meter_units_meters.asc'

(grid, topo) = read_esri_ascii(grid_raster, name='topographic__elevation')
# src = rio.open(grid_raster)
# crs = src.crs
# print(f'The input projection is {crs}')
# grid = RasterModelGrid((500,300))
# topo = grid.add_zeros("topographic__elevation", at="node")
topo[topo == -9999] = np.nan
topo[np.isnan(topo) == True] = -5
# grid.set_nodata_nodes_to_closed(topo, 9999)
# topo = topo.reshape((500,300))
sp = grid.shape
# topo = topo.reshape((sp))


# start reference
saveGrid = grid
saveTopo = topo
###################################
# This section was added by Thomas
# filter out any no data values
# topo[:] = -1
# put a channel down the middle of the test marsh
# topo[:,int(sp[1]/2)-10:int(sp[1]/2)+10] = -5
# for rw in range(0, len(topo[:,0])):
#     per = rw/len(topo[:,0])
#     val = (10 * per) * -1
#     topo[rw,:] = val
# topo[-2:,:] = -10 # set the bottom row to 2

print(topo.shape)

model_domain = np.copy(topo)
model_domain[:] = 1

# create a barrier in the center of the modeled marsh
# third = int(len(model_domain[0,:])/3)
# model_domain[int(len(model_domain)/2), 0:third] = 50 # this is a wall!
# model_domain[int(len(model_domain)/2), -third:len(model_domain[0:])] = 50
# model_domain[0:2, :] = 2

# topo[model_domain == 50] = 100 # If there is a wall sort that out

# model_domain = model_domain.tolist()

##################################
# grid boundaries
grid.set_closed_boundaries_at_grid_edges(right_is_closed=True,
                                               top_is_closed=True,
                                               left_is_closed=True,
                                               bottom_is_closed=False)


def complete_domain(model_domain):
    checkDomain = np.reshape(model_domain, grid.shape)
    domainIndex = np.reshape(range(0,len(model_domain)), grid.shape)
    if len(checkDomain[checkDomain[:,0] == 2]) == len(checkDomain[:,0])-2:
        print("The model is open along the left edge")
        checkDomain[:,0] = 2
    if len(checkDomain[checkDomain[:,-1] == 2]) == len(checkDomain[:,-1])-2:
        print("The model is open along the Right edge")
        checkDomain[:,-1] = 2
    updated_bndry_nodes = domainIndex[checkDomain ==2]
    return(checkDomain, updated_bndry_nodes)


model_domain, bndryNodes = complete_domain(model_domain)
boundaryTopo = topo.flatten()[bndryNodes]
# testing closing some of the boundaries.
model_domain[-50:,0] = 50 # this is just testing if this closes the boundaries.
model_domain[-1,-75:] = 50
model_domain[-1,:50] = 50
grid.set_nodata_nodes_to_closed(np.array(model_domain).flatten(), 50)
# model_domain[-100:,0] = 1
model_domain[-50:,0] = 1 # this is just testing if this closes the boundaries.
model_domain[-1,-75:] = 1
model_domain[-1,:50] = 1

model_domain[bndryNodes] = 2
boundaryTopo = topo[bndryNodes]
grid.set_nodata_nodes_to_closed(np.array(model_domain).flatten(), 50)

####### RSLR scenarios ######################################################################
def NOAA_SLR_Senarios(senario, rnd):
    # this assumes a starting year of 2020
    yr = rnd + 2020
    if senario == 'High':
        y = (6.18*10**-7*yr**2 - 0.00229*yr + 2.12) - 0.01588719999999988
    elif senario == 'Medium':
        y = ((-2.44*10**-6)*yr**2 + 0.0102*yr - 10.7) + 0.05217599999999578
    else:
        y = ((-1.70*10**-6)*yr**2 + 0.00712*yr - 7.44) - 0.005719999999999281
    return(y * 1000) # this is the predicted slr in mm

def USACE_SLR_Senarios(senario, rnd):
    # this assumes a starting year of 2020
    x = rnd + 2020
    if senario == 'High':
        y = (-4.11*10**-3)*x**2 + 17.049*x - 17661
    elif senario == 'Medium':
        y = (-2.5644*10**-3)*x**2 + 10.639*x - 11022
    else:
        y = ((-1.3623*10**-3)*x**2 + 5.6493*x - 5847.5)
    return(y) # this is the predicted slr in mm. The above equations are already in mm
############################################################################################



mev = mainEvolution(grid, model_domain=model_domain, boundaryValues = boundaryTopo, runMEM=True)

versions = ['High', 'Medium', 'Low']

v = versions[1]

# modelRunName = ''.join(['Ponding_V2_', v,'_RSLR_test'])
#
# fldir = ''.join(['examples/', modelRunName])
#
# if os.path.exists(fldir) == False:
#     os.mkdir(fldir)
#     print(f'Creating directory {fldir}')


# RUN the model #####################################################################################

print("Starting main loop")
for i in tqdm(range(10), colour = "green"):
    slr = 4 # USACE_SLR_Senarios(v, rnd = i)
    mev.run_one_step(timeStep = 1, round = i, model_domain = model_domain, relative_sea_level_rise_rate_mmPerYr = slr) # dt is a sea level rate modifier.
    grid.at_node["topographic__elevation"] = mev._elev
    # pu.saveModelProgression(mev, dir = fldir, rnd = i)
    if i == 0:
        testEle = (mev._elev)
print(f'The MSL at the end of the model run was {mev._mean_sea_level}')

imshow_grid(mev._grid, (mev._elev - saveTopo), cmap = 'RdBu')
plt.show()

imshow_grid(mev._grid, (mev._vegetation), cmap = 'RdBu')
plt.show()

# plot data #####################################################################################################

# foo = np.copy(mev._elev)
# imshow_grid(mev._grid, (foo),  cmap = 'terrain')
# plt.show()
#
# imshow_grid(mev._grid, (mev._percent_time_flooded),  cmap = 'terrain')
# plt.show()
#
# imshow_grid(mev._grid, (mev._elev - testEle), cmap = 'RdBu', vmax = max((mev._elev - testEle))+0.1, vmin = -(max((mev._elev - testEle))+0.1))
# plt.show()
###########################################################################################################


# files = write_esri_ascii(''.join([fldir,'/', flnm]), mev.grid)
# [os.path.basename(name) for name in sorted(files)]

# fldir = ''.join(['examples/', modelRunName, '/modelProgressionFiles'])




# pu.calculateRasterStats(grid = grid, fldir = fldir, classNames =['water', 'healthy_marsh', 'moderate_marsh', 'struggling_marsh', 'transitional', 'high_marsh'],
#                          rasterClassNumbers=[0, 1, 2, 3, 4, 5])
# savelocation = ''.join(['E:/models/MarshMorpho2D/Figures/', modelRunName, '.png'])
# pu.plotModelRuns(dir = fldir, classNames =['water', 'marsh', 'uplands'],
#                          rasterClassNumbers=[0, 1,  5], limits = 120, saveFile = savelocation)

# csvLocation = r'E:\models\MarshMorpho2D\Thomas_pyMarshMorpho2D\pymarshmorpho2d-master\examples\exported_Model_Run_CSV\Extraction_file_NOMEM.csv'
# # ptsLocation = r'E:\models\MarshMorpho2D\observation_points\Obversation_Points.shp'
# pu.extractAsCSV(dir = fldir, saveFile = csvLocation, gridData='veg_is_present_projected', observationPoints = None)



print("Script Complete")


