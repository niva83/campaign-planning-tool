import numpy as np

def check_utm_zone(utm_zone):
        flag = False
        grid_codes = ['C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X']
        try:

            grid_code = utm_zone[-1].upper() # in case users put lower case 
            utm_zone = int(utm_zone[:-1])
            if grid_code in grid_codes:
                flag = True
            else:
                print('Incorrect grid code!\nEnter a correct grid code!')
                flag = False
            
            if utm_zone >= 1 and utm_zone <= 60:
                flag = True and flag
            else:
                print('Incorrect UTM zone!\nEnter a correct UTM zone!')
                flag = False
        except:
            flag = False
            print('Wrong input!\nHint: there should not be spaces between UTM zone and grid code!')
        return flag


def which_hemisphere(utm_zone):
        grid_codes = ['C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X']
        
        if check_utm_zone(utm_zone):
            grid_code = utm_zone[-1].upper() # in case users put lower case 
            if grid_code in grid_codes[:10]:
                return 'South'
            else:
                return 'North'
                
        else:
            return None

        
def utm2epsg(utm_zone):
        if check_utm_zone(utm_zone):
            hemisphere = which_hemisphere(utm_zone)
                
            if hemisphere == 'North':
                return '326' + utm_zone[:-1]
            else:
                return '327' + utm_zone[:-1]
        else:
            return None        


def check_measurements(points):
        if(type(points).__module__ == np.__name__):
                if (len(points.shape) == 1 and points.shape[0] == 3) or (len(points.shape) == 2 and points.shape[1] == 3):
                    return True
                else:
                    print('Wrong dimensions!')
                    return False
        else:
            print('Input is not numpy array!')
            return False

class CPT():
    LANDCOVER_DATA_PATH = ""
    GOOGLE_API_KEY = ""
    
    MESH_RES = 100 # in m
    MAP_EXTENT = 5000 # in m

    
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
            if check_utm_zone(kwargs['utm_zone']):
                self.utm_zone = kwargs['utm_zone'][:-1]
                self.grid_code = kwargs['utm_zone'].upper() 
                self.epsg_code = utm2epsg(kwargs['utm_zone']) 
                self.flags['utm'] = True
            else:
                self.utm_zone = None
                self.grid_code = None
                self.epsg_code = None                

        
        # lidar positions
        self.lidar_1_pos = None        
        self.lidar_2_pos = None

        
        # GIS layers
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



    def generate_mesh(self, center):
            """
            Generate equally spaced (measurement) points on a horizontal plane.

            Parameters
            ----------
            center : ndarray
                    3D array containing data with `float` or `int` type
                    corresponding to x, y and z coordinates of the mesh center.
                    3D array data are expressed in meters.
            map_extent : int
                    map extent in x and y in meters.
            mesh_res : int
                    mesh resolution for x and y in meters.
            """
            map_corners = np.array([center[:2] - self.MAP_EXTENT, center[:2] + self.MAP_EXTENT])

            x, y = np.meshgrid(
                    np.arange(map_corners[0][0], map_corners[1][0] + self.MAP_EXTENT, self.MESH_RES),
                    np.arange(map_corners[0][1], map_corners[1][1] + self.MAP_EXTENT, self.MESH_RES)
                            )
            
            z = np.full(x.shape, center[2])		
            mesh = np.array([x, y, z]).T.reshape(-1, 3)
            return x, y, mesh

    def add_measurements(self, **kwargs):
        

        if (self.utm_zone == None) and (not 'utm_zone' in kwargs):
            print('UTM zone not specified!')
            self.flags['utm'] = False
        elif (self.utm_zone == None) and ('utm_zone' in kwargs):
            if check_utm_zone(kwargs['utm_zone']):
                self.utm_zone = kwargs['utm_zone'][:-1]
                self.grid_code = kwargs['utm_zone'].upper() 
                self.epsg_code = utm2epsg(kwargs['utm_zone']) 
                self.flags['utm'] = True
        
        if self.flags['utm'] and check_measurements(kwargs['measurements']):


            if len(kwargs['measurements']) == 2:
                    self.measurements_initial = np.unique(kwargs['measurements'], axis=0)
            else:
                    self.measurements_initial = np.array([kwargs['measurements']])
                    self.measurements_initial = np.unique(self.measurements_initial, axis=0)


    # def add_lidars(self, **kwargs):
    #         """
    #         adding lidar positions points
    #         *kwargs:
    #         - lidar_pos_1
    #         - lidar_pos_2

    #         must find closes index for lidar position...
    #         """

    #     pass

    # def generate_mesh(self, center, map_extent):
    #         """
    #         Generate equally spaced (measurement) points on a horizontal plane.

    #         Parameters
    #         ----------
    #         center : ndarray
    #                 3D array containing data with `float` or `int` type
    #                 corresponding to x, y and z coordinates of the mesh center.
    #                 3D array data are expressed in meters.
    #         map_extent : int
    #                 map extent in x and y in meters.
    #         mesh_res : int
    #                 mesh resolution for x and y in meters.
    #         """
    #         map_corners = np.array([center[:2] - map_extent, center[:2] + map_extent])

    #         x, y = np.meshgrid(
    #                 np.arange(map_corners[0][0], map_corners[1][0]+mesh_res, mesh_res),
    #                 np.arange(map_corners[0][1], map_corners[1][1]+mesh_res, mesh_res)
    #                         )
            
    #         z = np.full(x.shape, center[2])		
    #         mesh = np.array([x, y, z]).T.reshape(-1, 3)
    #         return x, y, mesh

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