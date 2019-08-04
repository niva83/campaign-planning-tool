from CPT.CPT import *
layout = CPT()
layout.REP_RADIUS = 1000
layout.MESH_EXTENT = 6000
layout.AVERAGE_RANGE = 500
layout.LANDCOVER_DATA_PATH = '/Volumes/Secondary_Drive/work/projects/campaign-planning-tool/data/input/landcover/g100_clc12_V18_5.tif'
layout.OUTPUT_DATA_PATH = '/Volumes/Secondary_Drive/work/projects/campaign-planning-tool/data/output/'
layout.set_utm_zone('36S')
layout.MAX_ELEVATION_ANGLE = 7
points = np.array([
[250596, 4231391, 80],
[250356, 4231711, 80],
[249476, 4231231, 80],
[248316, 4229751, 80],
[248356, 4229231, 80],
[248556, 4228631, 80],
[248316, 4227831, 80],
[248636, 4227311, 80],
[249156, 4227471, 80],
[249116, 4226911, 80],
[249196, 4226511, 80],
[249396, 4226151, 80],
[250396, 4226351, 80],
[250316, 4225711, 80],
[249876, 4225151, 80],
[250396, 4224751, 80],
[249796, 4224471, 80],
[250036, 4224071, 80],
[250476, 4223951, 80],
[250796, 4223751, 80],
[251236, 4224111, 80],
[251396, 4223631, 80]])
layout.add_measurements(measurements = points, points_type = 'initial')
layout.measurements_initial[:,2] = layout.measurements_initial[:,2] + layout.get_elevation('36S',layout.measurements_initial)
layout.optimize_measurements()
layout.generate_combined_layer(points_type = 'optimized')
layout.plot_optimization(points_type = 'initial')
print(layout.measurements_optimized)