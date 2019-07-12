class CPT():

  def __init__(self, **kwargs):
    if not 'measurement_points' in kwargs:
      self.measurement_points = None
    else:
      self.measurement_points = kwargs['measurement_points']          

    if not 'utm_zone' in kwargs:
      self.utm_zone = None
    else:
      self.utm_zone = kwargs['utm_zone']          

    if not 'lidar_1_pos' in kwargs:
      self.lidar_1_pos = None
    else:
      self.lidar_1_pos = kwargs['lidar_1_pos']            

    if not 'lidar_2_pos' in kwargs:
      self.lidar_2_pos = None
    else:
      self.lidar_2_pos = kwargs['lidar_2_pos']      
    
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
      
  def optimize_measurements(self):
      """
      Doc String
      """
    pass

  def generate_layers(self):
      """
      Doc String
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