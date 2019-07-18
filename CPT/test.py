# from CPT.CPT import CPT as CPT
from CPT.CPT import *
import numpy as np
from itertools import combinations, product
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon

MY_DPI = 96
def measurement_optimization_plt(DPI=100):
    fig, ax = plt.subplots(sharey=True,figsize=(500/MY_DPI, 500/MY_DPI), dpi=DPI) # note we must use plt.subplots, not plt.subplot

    for i in range(0,len(points)):
        if i == 0:
            ax.scatter(points[i][0] - map_center[0], 
                       points[i][1] - map_center[1], 
                       marker='o', color='black', s=10,zorder=1000, label = "measurements")
        else:
            ax.scatter(points[i][0] - map_center[0], 
                       points[i][1] - map_center[1], 
                       marker='o', color='black', s=10,zorder=1000)            


    for i in range(0,len(discs)):
        if i == 0:
            ax.scatter(discs[i][0] - map_center[0], 
                       discs[i][1] - map_center[1], 
                       marker='o', color='red', s=2,zorder=1000, label = "discs")
            ax.add_artist(plt.Circle((discs[i][0] - map_center[0], discs[i][1] - map_center[1]), 
                                     REPRESENT_RADIUS,                               
                                     facecolor='grey', edgecolor='black', 
                                     zorder=500,  alpha = 0.5))                 
            

        else:
            ax.scatter(discs[i][0] - map_center[0], 
                       discs[i][1] - map_center[1], 
                       marker='o', color='red', s=2,zorder=1000)
            ax.add_artist(plt.Circle((discs[i][0] - map_center[0], discs[i][1] - map_center[1]), 
                                     REPRESENT_RADIUS,                               
                                     facecolor='grey', edgecolor='black', 
                                     zorder=500, alpha = 0.5))                       
             

    plt.xlabel('Easting [m]', fontsize=12)
    plt.ylabel('Northing [m]', fontsize=12)
    ax.legend(loc='lower left', fontsize=10)

    ax.set_xlim(-5,15)
    ax.set_ylim(-5,15)

    ax.set_aspect(1.0)
    plt.show()
    

map_center = [0,0]
REPRESENT_RADIUS = 2
np.random
points = np.array([[1,3,3],[2,1,4],[5,2,1],[0,7.4,1],[7.4,4,1]])
new = CPT()
new.set_utm_zone("32V")
new.add_measurements(measurements = points)
new.REP_RADIUS = REPRESENT_RADIUS
new.optimize_measurements()
# print(new.measurements_optimized)

discs = new.measurements_optimized
measurement_optimization_plt()

# print(new.measurements_initial)
# print(new.measurements_optimized)

# # discs, matrix = generating_disc_matrix(points, REPRESENT_RADIUS)
# # print(discs)
# # disc_covering(points, REPRESENT_RADIUS)
# # print(matrix.shape)
# # print(discs)

# disc_covering(points, REPRESENT_RADIUS)
# print(points)
# print("\n")
# print(array_difference(points, np.array([[1,3,3],[2,1,4],[2,2,2]])))


# mesh_center = None

# if mesh_center == None:
#     print('It is None!')  

