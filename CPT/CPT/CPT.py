class CPT():
  LANDCOVER_DATA_PATH = ""
  GOOGLE_API_KEY = ""
  NO_LIDARS = 0
  NO_LAYOUTS = 0

  def __init__(self, *kwargs):
    # measurement positions
    self.measurements_initial = None
    self.measurements_optimized = None
    self.measurements_identified = None
    self.measurements_reachable = None
    self.utm_zone = None
    
    # lidar positions
    self.lidar_1_pos = None    
    self.lidar_2_pos = None

    # Lidar constraints
    if not 'max_elevation_angle' in kwargs:
      self.max_elevation_angle = 5 # in deg
    else:
      self.max_elevation_angle = kwargs['max_elevation_angle']

    if not 'min_elevation_angle' in kwargs:
      self.min_elevation_angle = 30 # in deg
    else:
      self.min_elevation_angle = kwargs['min_elevation_angle']

    if not 'average_range' in kwargs:
      self.average_range = 3000 # in m
    else:
      self.average_range = kwargs['average_range']

    if not 'max_acceleration' in kwargs:
      self.max_acceleration = 100 # in deg/s^2
    else:
      self.max_acceleration = kwargs['max_acceleration']
    
    # GIS layers
    self.topography_layer = None
    self.landcover_layer = None    
    self.exclusion_layer = None    
    self.elevation_angle_layer = None
    self.viewshed_layer = None
    self.range_layer = None
    self.combined_layer = None
    self.intersecting_angle_layer = None
    self. = None
    self.aerial_layer = None    

    self.flags = {'topography':False, 'landcover':False, 'exclusions': False,  
                  'viewshed':False, 'elevation_angle': False, 'range': False, 'intersecting_angle':False, 'measurements_optimized': False}    


    NO_LAYOUTS = NO_LAYOUTS + 1

  def add_measurements(self, *kwargs):
      """
      adding measurement points
      *kwargs:
      - measurement_points as triplet (Easting,Northing,Height) or (lon, lat, height)
      - UTM zone (if easting norhting height is given) or nothing
      """

    pass

  def add_lidars(self, *kwargs):
      """
      adding lidar positions points
      *kwargs:
      - lidar_pos_1
      - lidar_pos_2

      must find closes index for lidar position...
      """

    pass


  def optimize_measurements(self):
      """
      Disc covering problem  applied on the set of measurement points.
      """
    pass

  def generate_combined_layer(self):
      """
      Check flags for calculating other layers:
      - DEM layer
      - Landcover layer
      - Exclusion zone layer
      - LOS blockage layer
      - Elevation angle layer
      - Range layer
      - Aerial image???
      """
    pass

  def place_first_lidar(self):
      """
      Doc String
      """
    pass

  def place_second_lidar(self):
      """
      Doc String
      """
    pass

  def optimize_trajectory(self):
      """
      Doc String
      """
    pass

  def generate_trajectory(self):
      """
      Doc String
      """
    pass

  def find_measurements(self):
      """
      Doc String
      """
    pass

  def export_layout(self):
      """
      Doc String
      """
    pass    