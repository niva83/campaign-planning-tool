from recast import CPT
import numpy as np
from pathlib import Path

layout = CPT()
layout.MESH_EXTENT = 5000 # in m
layout.MESH_RES = 100 # in m
layout.AVERAGE_RANGE = 4000 # in m 
layout.PULSE_LENGTH = 200 # in ns
layout.FFT_SIZE = 64 # no points
layout.ACCUMULATION_TIME = 1000 # in ms
layout.ZOOM = 5
# layout.REP_RADIUS = 10000

# setting path to the folder 
# where to store the output data
layout.set_path('/Users/niva/Desktop/recast-test', 
                path_type = 'output')

# setting path to the landcover data 
layout.set_path('/Volumes/Secondary_Drive/work/projects/campaign-planning-tool/data/input/landcover/g100_clc12_V18_5.tif', 
                path_type = 'landcover')

layout.set_utm_zone('33T')
layout.MAX_ELEVATION_ANGLE = 45

points = np.array([
[576697.34, 4845753, 395 + 80],
[576968, 4845595, 439 + 80],
[577215, 4845425, 423 + 80],
[577439, 4845219, 430 + 80],
[577752, 4845005, 446 + 80],
[577979, 4844819, 478 + 80],
[578400, 4844449, 453 + 80],
[578658, 4844287, 450 + 80],
[578838, 4844034, 430 + 80],
[578974, 4843842, 417 + 80],
[579121, 4844186, 413 + 80],
[579246, 4843915, 410 + 80]
])
layout.add_measurement_instances(points = points, points_id = 'initial')
layout.generate_mesh()
layout.optimize_measurements(points_id = 'initial')
# layout.plot_optimization()

layout.generate_combined_layer(points_id = 'optimized')

layout.add_lidar_instance(position = np.array([578886, 4847688, 179]),
                          lidar_id = 'brise')

# layout.plot_layer(layout.layer_selector('combined'), 
#                   title = 'Lidar placement map' , 
#                   legend_label = 'Reachable points []')

layout.generate_second_lidar_layer(lidar_id = 'brise')

layout.add_lidar_instance(position = np.array([580460, 4846018, 220]), 
                          lidar_id = 'sirocco')       

layout.plot_layer(layer_id = 'landcover', 
                  title = 'Lidar placement map')                                     

# layout.optimize_trajectory(lidar_ids = ['brise', 'sirocco'], points_id = 'optimized', sync = True)


# layout.export_kml(lidar_ids = ['brise', 'sirocco'], 
#                   layer_ids = ['combined', 
#                                'second_lidar_placement', 
#                                'range_contrained'])
# layout.export_measurement_scenario(lidar_ids = ['brise', 'sirocco'])

print(layout.legend_label)