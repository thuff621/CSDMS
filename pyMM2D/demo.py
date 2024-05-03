import numpy as np
import matplotlib.pyplot as plt
from landlab.io import read_esri_ascii
from landlab import imshow_grid
from landlab import RasterModelGrid
# from pymarshmorpho2d import mainevolver
from pymarshmorpho2d import mainEvolution
from tqdm import tqdm
import os

print(os.getcwd())

#(grid, topo) = read_esri_ascii('pymarshmorpho2d-master/examples/Stone_Harbor_Point_Subset_bridge_removed_3meter.asc', name='topographic__elevation')
grid = RasterModelGrid((100,100))
topo = grid.add_zeros("topographic__elevation", at="node")
topo[topo == -9999] = np.nan
grid.set_nodata_nodes_to_closed(topo, 999)
# topo = topo.reshape((500,300))
sp = grid.shape
topo = topo.reshape((sp))


# start reference
saveGrid = grid
saveTopo = topo
###################################
# This section was added by Thomas
# filter out any no data values
topo[:] = -1
# put a channel down the middle of the test marsh
# topo[:,int(sp[1]/2)-10:int(sp[1]/2)+10] = -5
# for rw in range(0, len(topo[:,0])):
#     per = rw/len(topo[:,0])
#     val = (10 * per) * -1
#     topo[rw,:] = val
topo[-2:,:] = -10 # set the bottom row to 2

print(topo.shape)

model_domain = np.copy(topo)
model_domain[:] = 1
# model_domain = model_domain.flatten()

# create a barrier in the center of the modeled marsh
# third = int(len(model_domain[0,:])/3)
# model_domain[int(len(model_domain)/2), 0:third] = 50 # this is a wall!
# model_domain[int(len(model_domain)/2), -third:len(model_domain[0:])] = 50
# model_domain[-1:, :] = 2

# topo[model_domain == 50] = 100 # If there is a wall sort that out

# model_domain = model_domain.tolist()

# boundary close function
# def set_closed_boundaries_mm2d(model_domain, grid, right_is_closed=True, top_is_closed=True, left_is_closed = True, bottom_is_closed=True):
#         closed_domain = model_domain.copy()
#         if right_is_closed == False:
#             model_domain[:,-1] = 2
#         else:
#             closed_domain[:,-1] = 2
#         if left_is_closed == False:
#             model_domain[:,0] = 2
#         else:
#             closed_domain[:,0] = 2
#         if top_is_closed == False:
#             model_domain[-1,:] = 2
#         else:
#             closed_domain[-1,:] = 2
#         if bottom_is_closed == False:
#             model_domain[0,:] = 2
#         else:
#             closed_domain[0,:] = 2
#
#         grid.set_nodata_nodes_to_closed(model_domain.flatten(), 2)
#         return(model_domain, grid, closed_domain)
#
#
# ##################################
# # grid boundaries
# model_domain, grid, closed_domain = set_closed_boundaries_mm2d(model_domain, grid,
#                                                 right_is_closed=True,
#                                                top_is_closed=False,
#                                                left_is_closed=True,
#                                                bottom_is_closed=True)
# model_domain = model_domain.flatten()
# bndryNodes = grid.closed_boundary_nodes
# model_domain[bndryNodes] = 2
# boundaryTopo = topo.flatten()[bndryNodes]
# grid.set_nodata_nodes_to_closed(np.array(closed_domain).flatten(), 2)

# grid boundaries
grid.set_closed_boundaries_at_grid_edges(right_is_closed=True,
                                               top_is_closed=False,
                                               left_is_closed=True,
                                               bottom_is_closed=True)
bndryNodes = grid.open_boundary_nodes
model_domain = model_domain.flatten()
model_domain[bndryNodes] = 2

# double check that all nodes on an edge are open in the model domain.
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
grid.set_nodata_nodes_to_closed(np.array(model_domain).flatten(), 50)

# plot the grid with topography displayed.
imshow_grid(grid, topo)
plt.show()

mev = mainEvolution(grid, model_domain=model_domain, boundaryValues = boundaryTopo)

print("Starting main loop")
for i in tqdm(range(100)):
    # print("\n")
    # print(f'Running time step {i}')
    # print("\n")
    # print("*"*60)
    mev.run_one_step(timeStep = 1, round = i, model_domain = model_domain) # dt is a sea level rate modifier.
    grid.at_node["topographic__elevation"] = mev._elev

    # imshow_grid(mev._grid, (mev._elev), cmap='terrain')
    # plt.show()
    if i == 0:
        # print("Initializing base values")
        # test = (mev._vegetation)# * mev._veg_is_present)
        testEle = (mev._elev)
        # print(mev._elev)
    # if i >= 18:
    #     imshow_grid(mev._grid, (mev._elev), cmap='terrain')
    #     plt.show()
foo = np.copy(mev._elev)
foo[foo < -10] = -10
foo[foo > 10] = 10
imshow_grid(mev._grid, (foo),  cmap = 'terrain')
plt.show()
imshow_grid(mev._grid, (mev._elev - testEle), cmap = 'terrain')
plt.show()

# print(mev._mean_sea_level)
# #
# # imshow_grid(grid, (mev._vegetation), cmap = 'RdBu')
# # plt.show()
# #
# # imshow_grid(grid, (mev._tidal_flow), cmap='RdBu')
# # plt.show()
#
# # imshow_grid(grid, (mev._elev - testEle), cmap = 'RdBu')
# # plt.show()
# imshow_grid(grid, (mev._elev), cmap = 'RdBu')
# plt.show()

