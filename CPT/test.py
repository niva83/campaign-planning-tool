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

    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)

    ax.set_aspect(1.0)
    plt.show()






# def generating_disc_matrix(points,radius):
    
#     points_combination = np.asarray(list(combinations(list(points[:,(0,1)]), 2)))    
#     discs = (points_combination[:,0] + points_combination[:,1]) / 2

#     temp = np.asarray(list(product(list(discs), list(points[:,(0,1)]))))
#     distances =  np.linalg.norm(temp[:,0] - temp[:,1], axis = 1)
#     distances = np.where(distances <= radius, 1, 0)
    
#     matrix = np.asarray(np.split(distances,len(discs)))
#     total_covered_points = np.sum(matrix,axis = 1)

#     matrix = matrix[(-total_covered_points).argsort()]
#     discs = discs[(-total_covered_points).argsort()]

#     # adding 0 m for elevation of each disc
#     discs = np.append(discs.T, np.array([np.zeros(len(discs))]),axis=0).T

#     return discs, matrix

# def disc_covering(points, radius):
#     discs, matrix = generating_disc_matrix(points,radius)
#     points_uncovered = points
#     points_covered_total = np.zeros((0,3))
#     discs_selected = np.zeros((0,3))
#     i = 0
#     j = len(points_uncovered)

#     while i <= (len(discs) - 1) and j > 0 :
#         indexes = np.where(matrix[i] == 1 )
#         points_covered = points[indexes]
#         points_new = array_difference(points_covered, points_covered_total)
#         if len(points_new) > 0:
#             points_covered_total = np.append(points_covered_total, points_new,axis=0)
#             discs_selected = np.append(discs_selected, np.array([discs[i]]),axis=0)
#         points_uncovered = array_difference(points_uncovered, points_covered)        
#         i += 1
#         j = len(points_uncovered)
#     discs_selected = np.append(discs_selected, points_uncovered, axis = 0)
    


#     print(points_uncovered)
#     print(points_covered_total)
#     # print(points_covered_total)
#     print("Selected discs:")
#     print(discs_selected)
        
    

map_center = [0,0]
REPRESENT_RADIUS = 5
points = np.array([[1,3,3],[2,1,4],[5,2,1],[0,7.4,1],[7.4,4,1]])
new = CPT()
new.set_utm_zone("31V")
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

