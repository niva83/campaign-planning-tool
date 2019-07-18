import numpy as np
from itertools import combinations, product

def array_difference(A,B):
    """
    Finding which elements in array A are not present in
    array B.

    Parameters
    ----------
    A : ndarray
        nD array containing data with `float` or `int` type
    B : ndarray
        nD array containing data with `float` or `int` type
    """

    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
           'formats':ncols * [A.dtype]}

    C = np.intersect1d(A.view(dtype), B.view(dtype))
    if len(C)==0:
        return A

    D = np.setdiff1d(A.view(dtype), B.view(dtype))

    # This last bit is optional if you're okay with "C" being a structured array...
    # C = C.view(A.dtype).reshape(-1, ncols)
    if len(D) == 0:
        return np.array([])

    D = D.view(A.dtype).reshape(-1, ncols)
    return D

class CPT():
    LANDCOVER_DATA_PATH = ""
    GOOGLE_API_KEY = ""
    
    MESH_RES = 100 # in m
    MAP_EXTENT = 5000 # in m

    
    REP_RADIUS = 500 # in m
    MAX_ELEVATION_ANGLE = 5 # in deg
    MIN_INTERSECTING_ANGLE = 30 # in deg
    AVERAGE_RANGE = 3000 # in m
    MAX_ACCELERATION = 100 # deg / s^2

    NO_LIDARS = 0
    NO_LAYOUTS = 0

    def __init__(self, **kwargs):
        # flags
        # add missing flags as you code
        self.flags = {'topography':False, 'landcover':False, 'exclusions': False,    
                                    'viewshed':False, 'elevation_angle': False, 'range': False, 
                                    'intersecting_angle':False, 'measurements_optimized': False,
                                    'measurements_added': False,'lidar_1_pos':False,'lidar_2_pos':False,
                                    'map_center_added': False,
                                    'utm':False, 'input_check_pass': False}        


        # measurement positions
        self.measurements_initial = None
        self.measurements_optimized = None
        self.measurements_identified = None
        self.measurements_reachable = None


        if not 'map_center' in kwargs:
            self.map_center = None 
        else:
            self.map_center = kwargs['map_center']

        if not 'utm_zone' in kwargs:
            self.utm_zone = None
            self.grid_code = None
            self.epsg_code = None 
        else:
            if self.check_utm_zone(kwargs['utm_zone']):
                self.utm_zone = kwargs['utm_zone'][:-1]
                self.grid_code = kwargs['utm_zone'][-1].upper() 
                self.epsg_code = self.utm2epsg(kwargs['utm_zone']) 
                self.flags['utm'] = True
            else:
                self.utm_zone = None
                self.grid_code = None
                self.epsg_code = None                

        
        # lidar positions
        self.lidar_1_pos = None        
        self.lidar_2_pos = None

        
        # GIS layers
        self.map_center = None
        self.map_corners = None
        self.x = None
        self.y = None
        self.z = None
        self.mesh = None
        self.topography_layer = None
        self.landcover_layer = None        
        self.exclusion_layer = None        
        self.elevation_angle_layer = None
        self.los_layer = None
        self.range_layer = None
        self.combined_layer = None
        self.intersecting_angle_layer = None
        self.aerial_layer = None


        CPT.NO_LAYOUTS += 1
    
    def set_utm_zone(self, utm_zone):
        if self.check_utm_zone(utm_zone):
            self.utm_zone = utm_zone[:-1]
            self.grid_code = utm_zone[-1].upper() 
            self.epsg_code = self.utm2epsg(utm_zone) 
            self.flags['utm'] = True

    def utm_exists(self, **kwargs):
        if (self.utm_zone == None) and (not 'utm_zone' in kwargs):
            print('UTM zone not specified!')
            self.flags['utm'] = False
        elif 'utm_zone' in kwargs:
            if self.check_utm_zone(kwargs['utm_zone']):
                self.utm_zone = kwargs['utm_zone'][:-1]
                self.grid_code = kwargs['utm_zone'][-1].upper() 
                self.epsg_code = self.utm2epsg(kwargs['utm_zone']) 
                self.flags['utm'] = True
        
    def add_measurements(self, **kwargs):
        self.utm_exists(**kwargs)
        if self.flags['utm'] == False:
            print('Cannot add measurement points without specificing UTM zone!')
        if self.flags['utm'] and self.check_measurement_positions(kwargs['measurements']):
            if len(kwargs['measurements'].shape) == 2:
                    self.measurements_initial = np.unique(kwargs['measurements'], axis=0)
            else:
                    self.measurements_initial = np.array([kwargs['measurements']])
                    self.measurements_initial = np.unique(self.measurements_initial, axis=0)
            self.flags['measurements_added'] = True

    def generating_disc_matrix(self):
        if self.flags['measurements_added']:
            points_combination = np.asarray(list(combinations(list(self.measurements_initial[:,(0,1)]), 2)))    
            discs = (points_combination[:,0] + points_combination[:,1]) / 2

            temp = np.asarray(list(product(list(discs), list(self.measurements_initial[:,(0,1)]))))
            distances =  np.linalg.norm(temp[:,0] - temp[:,1], axis = 1)
            distances = np.where(distances <= self.REP_RADIUS, 1, 0)
            
            matrix = np.asarray(np.split(distances,len(discs)))
            total_covered_points = np.sum(matrix,axis = 1)

            matrix = matrix[(-1*total_covered_points).argsort()]
            discs = discs[(-1*total_covered_points).argsort()]

            # adding 0 m for elevation of each disc
            discs = np.append(discs.T, np.array([np.zeros(len(discs))]),axis=0).T

            return discs, matrix
        else:
            print("No measurement positions added, nothing to optimize!")

    def optimize_measurements(self):

        if self.flags['measurements_added']:
            discs, matrix = self.generating_disc_matrix()
            points_uncovered = self.measurements_initial
            points_covered_total = np.zeros((0,3))
            discs_selected = np.zeros((0,3))
            i = 0
            j = len(points_uncovered)

            while i <= (len(discs) - 1) and j > 0 :
                indexes = np.where(matrix[i] == 1 )
                points_covered = self.measurements_initial[indexes]
                points_new = array_difference(points_covered, points_covered_total)
                if len(points_new) > 0:
                    points_covered_total = np.append(points_covered_total, points_new,axis=0)
                    discs_selected = np.append(discs_selected, np.array([discs[i]]),axis=0)
                    print(discs_selected)
                points_uncovered = array_difference(points_uncovered, points_covered)        
                i += 1
                j = len(points_uncovered)
            if len(points_uncovered) > 0:
                self.measurements_optimized = np.append(discs_selected, points_uncovered, axis = 0)
            else:
                self.measurements_optimized = discs_selected
        else:
            print("No measurement positions added, nothing to optimize!")
            

    def add_lidars(self, **kwargs):
        if 'lidar_1_pos' in kwargs or 'lidar_2_pos' in kwargs:
            self.utm_exists(**kwargs)
            if self.flags['utm'] == False:
                print('Cannot add lidar positions without specificing UTM zone!')
            else:        
                if 'lidar_1_pos' in kwargs and self.check_lidar_position(kwargs['lidar_1_pos']):
                    self.lidar_1_pos = kwargs['lidar_1_pos']
                    self.flags['lidar_1_pos'] = True
                    print('Lidar 1 position added!')
                if 'lidar_2_pos' in kwargs and self.check_lidar_position(kwargs['lidar_2_pos']):
                    self.lidar_2_pos = kwargs['lidar_2_pos']
                    self.flags['lidar_2_pos'] = True
                    print('Lidar 2 position added!')
        else:
            print('Lidar position(s) not specified!')        

    def generate_mesh(self, **kwargs):
        """
        Generate equally spaced (measurement) points on a horizontal plane.

        Parameters
        ----------
        center : ndarray
                3D array containing data with `float` or `int` type
                corresponding to Easting, Northing and Height coordinates of the mesh center.
                3D array data are expressed in meters.
        map_extent : int
                map extent in Easting and Northing in meters.
        mesh_res : int
                mesh resolution for Easting and Northing in meters.
        """

        if 'map_extent' in kwargs:
            self.MAP_EXTENT = kwargs['map_extent']
        if 'map_center' in kwargs:
            self.map_center = kwargs['map_center']
            self.flags['map_center_added'] = True
        elif self.flags['measurements_added']:
            self.map_center = np.int_(np.mean(self.measurements_initial,axis = 0))
            self.flags['map_center_added'] = True
        else:
            print('Map center missing!')
            print('Mesh cannot be generated!')
            self.flags['map_center_added'] = False

        if self.flags['map_center_added']:
            # securing that the input parameters are int 
            self.map_center = np.int_(self.map_center)
            self.MAP_EXTENT = int(int(self.MAP_EXTENT / self.MESH_RES) * self.MESH_RES)
            self.map_corners = np.array([self.map_center[:2] - self.MAP_EXTENT, 
                                            self.map_center[:2] + self.MAP_EXTENT])

            self.x, self.y = np.meshgrid(
                    np.arange(self.map_corners[0][0], self.map_corners[1][0] + self.MAP_EXTENT, self.MESH_RES),
                    np.arange(self.map_corners[0][1], self.map_corners[1][1] + self.MAP_EXTENT, self.MESH_RES)
                            )
            
            self.z = np.full(self.x.shape, self.map_center[2])		
            self.mesh = np.array([self.x, self.y, self.z]).T.reshape(-1, 3)
    @staticmethod
    def check_measurement_positions(points):
        if(type(points).__module__ == np.__name__):
                if (len(points.shape) == 1 and points.shape[0] == 3) or (len(points.shape) == 2 and points.shape[1] == 3):
                    return True
                else:
                    print('Wrong dimensions!')
                    print('Measurement positions were not added')
                    return False
        else:
            print('Input is not numpy array!')
            print('Measurement positions were not added')
            return False

    @staticmethod
    def check_lidar_position(lidar_position):
        if(type(lidar_position).__module__ == np.__name__):
                if (len(lidar_position.shape) == 1 and lidar_position.shape[0] == 3):
                    return True
                else:
                    print('Wrong dimensions!\nLidar position is described by 3 parameters:\n(1)Easting\n(2)Northing\n(3)Height')
                    print('Lidar position was not added')
                    return False
        else:
            print('Input is not numpy array!')
            print('Lidar position was not added')
            return False      

    @staticmethod  
    def check_utm_zone(utm_zone):
        """
        Checks whether UTM zone with grid code is valid or not.

        Parameters
        ----------
        utm_zone : str
            A string containing UTM zone with grid code.
        
        Returns
        -------
        out : bool
            A boolean indicating True or False .
        
        Examples
        --------
        If UTM zone and grid code exist.
        >>> utm2epsg('31V') 
        True

        If UTM zone and/or grid code don't exist.
        >>> utm2epsg('61Z')
        False
        """ 
        flag = False
        grid_codes = ['C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X']
        try:

            grid_code = utm_zone[-1].upper() # in case users put lower case 
            utm_zone = int(utm_zone[:-1])
            if grid_code in grid_codes:
                print('Correct grid code!')
                flag = True
            else:
                print('Incorrect grid code!\nEnter a correct grid code!')
                flag = False
            
            if utm_zone >= 1 and utm_zone <= 60:
                print('Correct UTM zone!')
                flag = True and flag
            else:
                print('Incorrect UTM zone!\nEnter a correct UTM zone!')
                flag = False
        except:
            flag = False
            print('Wrong input!\nHint: there should not be spaces between UTM zone and grid code!')
        return flag

    @staticmethod
    def which_hemisphere(utm_zone):
        """
        Returns whether UTM zone belongs to the  Northern or Southern hemisphere. 

        Parameters
        ----------
        utm_zone : str
            A string containing UTM zone with grid code.
        
        Returns
        -------
        out : str
            A string indicating North or South hemisphere.
        
        Examples
        --------
        If UTM zone and grid code exist.
        >>> utm2epsg('31V') 
        'North'

        >>> utm2epsg('31C') 
        'South'        

        If UTM zone and/or grid code don't exist.
        >>> utm2epsg('61Z')
        None
        """
 
        grid_codes = ['C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X']
        grid_code = utm_zone[-1].upper() # in case users put lower case 
        if int(utm_zone[:-1]) >= 1 and int(utm_zone[:-1]) <= 60:
            if grid_code in grid_codes[10:]:
                return 'North'
            elif grid_code in grid_codes[:10]:
                return 'South'
            else:
                return None
        else:
            return None

    @staticmethod        
    def utm2epsg(utm_zone):
        """
        Converts UTM zone with grid code to EPSG code.

        Parameters
        ----------
        utm_zone : str
            A string representing an UTM zone with a grid code, containing digits (from 1 to 60) 
            indicating the UTM zone followed by a character ('C' to 'X' excluding 'O') for the grid code.
        
        Returns
        -------
        out : str
            A string containing EPSG code.
        
        Examples
        --------
        If UTM zone and grid code exist.
        >>> utm2epsg('31V') 
        '32631'

        If UTM zone and/or grid code don't exist.
        >>> utm2epsg('61Z')
        None
        """
        grid_codes = ['C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X']
        grid_code = utm_zone[-1].upper() # in case users put lower case 
        if int(utm_zone[:-1]) >= 1 and int(utm_zone[:-1]) <= 60:
            if grid_code in grid_codes[10:]:
                return '326' + utm_zone[:-1]
            elif grid_code in grid_codes[:10]:
                return '327' + utm_zone[:-1]
            else:
                return None
        else:
            return None

    # def optimize_measurements(self):
    #         """
    #         Disc covering problem    applied on the set of measurement points.
    #         """
    #     pass

    # def find_measurements(self):
    #         """
    #         Doc String
    #         """
    #     pass

    # def generate_topographic_layer(self):
    #         """
    #         Doc String
    #         """
    #     pass

    # def generate_landcover_layer(self):
    #         """
    #         Doc String
    #         """
    #     pass

    # def generate_los_layer(self):
    #         """
    #         Doc String
    #         """
    #     pass

    # def generate_range_layer(self):
    #         """
    #         Doc String
    #         """
    #     pass

    # def generate_elevation_layer(self):
    #         """
    #         Doc String
    #         """
    #     pass

    # def generate_combined_layer(self):
    #         """
    #         Check flags for calculating other layers:
    #         - DEM layer
    #         - Landcover layer
    #         - Exclusion zone layer
    #         - LOS blockage layer
    #         - Elevation angle layer
    #         - Range layer
    #         - Aerial image???
    #         """
    #     pass

    # def generate_intersecting_layer(self):
    #         """
    #         Doc String
    #         """
    #     pass

    # def generate_second_lidar_layer(self):
    #         """
    #         Doc String
    #         """
    #     pass

    # def optimize_trajectory(self):
    #         """
    #         Doc String
    #         """
    #     pass

    # def generate_trajectory(self):
    #         """
    #         Doc String
    #         """
    #     pass

    # def export_campaign_design(self):
    #         """
    #         Doc String
    #         """
    #     pass        