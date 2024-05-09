import numpy as np
import pandas as pd
from landlab.io import read_esri_ascii
from landlab.io.esri_ascii import write_esri_ascii
# from pymarshmorpho2d import mainevolver
from pymarshmorpho2d import mainEvolution
from tqdm import tqdm
import rasterio as rio
import os
from glob import glob
from pymarshmorpho2d import pyMM2DUtilities as pu

print(os.getcwd())

grid_raster = 'examples/Gull_Island_10meter.asc'

(grid, topo) = read_esri_ascii(grid_raster, name='topographic__elevation')
src = rio.open(grid_raster)
crs = src.crs
print(f'The input projection is {crs}')
# grid = RasterModelGrid((500,300))
# topo = grid.add_zeros("topographic__elevation", at="node")
topo[topo == -9999] = np.nan
grid.set_nodata_nodes_to_closed(topo, 9999)
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
                                               top_is_closed=False,
                                               left_is_closed=False,
                                               bottom_is_closed=True)


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



mev = mainEvolution(grid, model_domain=model_domain, boundaryValues = boundaryTopo, runMEM=False)

versions = ['High', 'Medium', 'Low']

v = versions[1]

modelRunName = ''.join(['Ponding_V2_', v,'_RSLR_test'])

fldir = ''.join(['examples/', modelRunName])

if os.path.exists(fldir) == False:
    os.mkdir(fldir)
    print(f'Creating directory {fldir}')


# RUN the model #####################################################################################

print("Starting main loop")
for i in tqdm(range(50), colour = "green"):
    slr = USACE_SLR_Senarios(v, rnd = i)
    # if i < 30:
    #     slr = 4
    # else:
    #     slr = 100
    mev.run_one_step(timeStep = 1, round = i, model_domain = model_domain, relative_sea_level_rise_rate_mmPerYr = slr,
                     saveModelParamsFile = fldir) # dt is a sea level rate modifier.
    grid.at_node["topographic__elevation"] = mev._elev
    pu.saveModelProgression(mev, dir = fldir, rnd = i)
    if i == 0:
        testEle = (mev._elev)
print(f'The MSL at the end of the model run was {mev._mean_sea_level}')

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

fldir = ''.join(['examples/', modelRunName, '/modelProgressionFiles'])


pu.modelGridProjection(fldir = fldir, crs = crs)


# pu.calculateRasterStats(grid = grid, fldir = fldir, classNames =['water', 'healthy_marsh', 'moderate_marsh', 'struggling_marsh', 'transitional', 'high_marsh'],
#                          rasterClassNumbers=[0, 1, 2, 3, 4, 5])
# savelocation = ''.join(['E:/models/MarshMorpho2D/Figures/', modelRunName, '.png'])
# pu.plotModelRuns(dir = fldir, classNames =['water', 'marsh', 'uplands'],
#                          rasterClassNumbers=[0, 1,  5], limits = 120, saveFile = savelocation)

# csvLocation = r'E:\models\MarshMorpho2D\Thomas_pyMarshMorpho2D\pymarshmorpho2d-master\examples\exported_Model_Run_CSV\Extraction_file_NOMEM.csv'
# # ptsLocation = r'E:\models\MarshMorpho2D\observation_points\Obversation_Points.shp'
# pu.extractAsCSV(dir = fldir, saveFile = csvLocation, gridData='veg_is_present_projected', observationPoints = None)

pu.extractLandcover(fldir, classNames =['water', 'marsh', 'marsh', 'marsh', 'uplands'],
                           rasterClassNumbers=[0, 1, 2, 3, 4, 5])

print("Script Complete")