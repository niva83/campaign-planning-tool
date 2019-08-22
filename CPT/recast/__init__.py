import numpy as np
import pandas as pd
from pathlib import Path
import os, shutil

from ._export import Export
from ._plot import Plot
from ._points_optimization import OptimizeMeasurements
from ._trajectory_optimization import OptimizeTrajectory
from ._generate_layers import LayersGIS


class CPT(Export, Plot, OptimizeMeasurements, OptimizeTrajectory, LayersGIS):
    """
    A class for designing scanning lidar measurement campaigns.

    Attributes
    ----------
    NO_LAYOUTS : int
        A number of layout instances generated.
    NO_DATA_VALUE : int
        A default value in case of missing data.
        Currently the default value is set to 0.
    LANDCOVER_DATA_PATH : str
        The path to a CORINE landcover dataset.
        A default value set to an empty string.
    OUTPUT_DATA_PATH : str
        The path to an existing folder where results will be saved.
    GOOGLE_API_KEY : str
        An API key to access Google Maps database.
        A default value is set to an empty string.
    FILE_EXTENSIONS : ndarray
        nD array of strings containing none-temporary file exentions.
    MESH_RES : int
        The resolution of the mesh used for GIS layers creation.
        The resolution is expressed in meters. 
        A default value is set to 100 m.
    MESH_EXTENT : int
        The mesh extent along x and y axes as a single value.
        The extent is expressed in meters. 
        A default value is set to 5000 m.
    REP_RADIUS : int
        MEASNET's representativness radius of measurements.
        The radius is expressed in meters.
        A default value is set to 500 m.
    POINTS_TYPE : ndarray
        nD array of strings indicating measurement point type.
        Five different types are preset and used in CPT.
    ACCUMULATION_TIME : int
        The laser pulse backscatter accumulation time.
        A value is given in ms.
        A default value is set to 1000 ms.
    AVERAGE_RANGE : int
        The average range of lidars.
        The range is expressed in m.
        A default value is set to 3000 m.  
    MAX_ACCELERATION : int
        The maximum acceleration of a scanner head.
        The acceleration is expressed in deg/s^2.
        A default value is set to 100 deg/s^2.
        Update the value according to the lidar specifications.
    MAX_ELEVATION_ANGLE : float
        The maximum allowed elevation angle for a beam steering.
        The angle is expressed in deg.
        A default value is set to 5 deg.
    MAX_NO_OF_GATES : int
        The maximum number of range gates along each LOS.
        A default value is set to 100.
        Update the value according to the lidar specifications.
    MIN_INTERSECTING_ANGLE : float
        The minimum intersecting angle between two beams.
        The angle is expressed in deg.
        A default value is set to 30 deg.
    PULSE_LENGTH : int
        The pulse length expressed in ms.
        A default value is set to 200 ms.
        Update the value according to the lidar configuration.
    FFT_SIZE :int
        A number of FFT points used to perform spectral analysis.
        A default value is set to 128 points.
        Update the value according to the lidar confguration.
    MY_DPI : int
        DPI for plots.
    FONT_SIZE : int
        Font size for plot labels.
    ZOOM : int
        Indicates how many times a GEOTiff should be enlarged.

    Methods
    --------
    set_path(**kwargs)
        Sets the file path to the landcover data and for results/data storage.
    add_measurement_instances(**kwargs)
        Adds measurement points to the measurement points dictionary.
    points_selector(points_id)
        Returns a numpy array of measurement points based on the points id.
    add_lidar_instance(**kwargs)
        Adds a lidar instance to the lidar dictionary.                
    update_lidar_instance(self, **kwargs):
        Updates a instance in the lidar dictionary with various information.
    find_mesh_points_index(point)
        For a given input point returns indexes of the closest point
        in the generated mesh.
    generate_beam_coords(lidar_pos, meas_pt_pos, opt=1):
        Generates beam steering coordinates in spherical coordinate system from 
        multiple lidar positions to a single measurement point and vice verse.
    generate_mesh(**kwargs)
        Generates a rectangular horizontal mesh containing equally spaced points.
    generate_combined_layer(**kwargs)
        Generates the combined layer which is used 
        for the positioning of the first lidars.
    generate_second_lidar_layer(**kwargs)
        Generates the combined layer which is used
        for the positioning of the first lidars.
    get_elevation(utm_zone, pts_utm)
        Fetch elevation from the SRTM database for 
        a number of points described by in the UTM coordinates.
    layer_selector(layer_id)
        Selects GIS layer according to the provided type.
    set_utm_zone(utm_zone)
        Sets EPSG code, latitudinal and longitudinal zones to the CPT instance. 
    utm2geo(points_utm, long_zone, hemisphere)
        Converts an array of points in the UTM coord system to
        an array of point in the GEO coord system.
    optimize_measurements(**kwargs)
        Optimizes measurement positions by solving disc covering problem.
    optimize_trajectory(**kwargs)
        Finding a shortest trajectory through the set of measurement points.
    generate_trajectory(lidar_pos, trajectory)
        Generates step-stare trajectory based on the lidar position and 
        trajectory points.
    plot_layer(**kwargs)
        Plots individual GIS layers and lidar positions.
    plot_optimization(**kwargs)
        Plots measurement point optimization result.
    plot_optimization(**kwargs)
        Plots measurement point optimization result.
    export_kml(**kwargs)
        Exports campaign design as as a Google compatible KML file.
    export_layer(**kwargs)
        Exports a specific GIS layer as GeoTIFF image.
    export_measurement_scenario(**kwargs)
        Exports measurement scenarios for given lidars.                
    """

    NO_LAYOUTS = 0

    NO_DATA_VALUE = 0
    LANDCOVER_DATA_PATH = ""
    OUTPUT_DATA_PATH = ""    
    GOOGLE_API_KEY = ""
    FILE_EXTENSIONS = np.array(['.tif', '.tiff', '.pdf', '.kml', '.png', '.pmc', '.xml' , '.yaml', '.csv'])

    MESH_RES = 100 # in m
    MESH_EXTENT = 5000 # in m
    REP_RADIUS = 500 # in m
    POINTS_TYPE = np.array(['initial', 'optimized', 'reachable', 'identified', 'misc'])
    LAYER_TYPE = [
                    'orography',
                    'landcover',
                    'canopy_height',
                    'topography',
                    'restriction_zones',
                    'elevation_angle_contrained',
                    'range_contrained',
                    'los_blockage',
                    'combined',
                    'intersecting_angle_contrained',
                    'second_lidar_placement',
                    'aerial_image',
                    'misc'
                ]
    COLOR_LIST = ['blue', 'green', 'red', 'purple', 
                  'brown', 'pink', 'gray', 'olive', 'cyan']

    __SPEC_LAYERS = ['elevation_angle_contrained', 
                     'range_contrained',
                     'los_blockage', 
                     'combined',
                     'intersecting_angle_contrained', 
                     'second_lidar_placement']
    
    ACCUMULATION_TIME = 1000 # in ms
    AVERAGE_RANGE = 3000 # in m
    MIN_RANGE = 50 # in m 
    MAX_RANGE = 6000 # in m
    MAX_ACCELERATION = 100 # in deg / s^2
    MAX_VELOCITY = 50 # in deg / s
    MAX_ELEVATION_ANGLE = 5 # in deg
    MAX_NO_OF_GATES = 100 # maximum number of range gates
    MIN_INTERSECTING_ANGLE = 30 # in deg
    PULSE_LENGTH = 400 # in ns
    FFT_SIZE = 128 # no points
    NO_DIGITS = 2 # number of decimal digits for positions and angles

    MY_DPI = 100
    FONT_SIZE = 10
    ZOOM = 10

    def __init__(self):
        # measurement positions / mesh / beam coords
        self.long_zone = None
        self.lat_zone = None
        self.epsg_code = None 
        self.hemisphere = None 
        self.measurements_dictionary = {}
        self.measurements_initial = None
        self.measurements_optimized = None
        self.measurements_identified = None
        self.measurements_reachable = None
        self.measurements_misc = None
        self.points_id = 'initial'
        self.beam_coords = None
        self.mesh_center = None
        self.flat_index_array = None 
        self.reachable_points = None
        self.trajectory = None
        self.motion_table = None
        self.motion_program = None
        self.range_gate_file = None
        self.legend_label = None
        
        # lidar positions
        self.lidar_dictionary = {}

        # GIS layers
        self.mesh_corners_utm = None
        self.mesh_corners_geo = None
        self.x = None
        self.y = None
        self.z = None
        self.mesh_utm = None
        self.mesh_geo = None
        self.orography_layer = None
        self.canopy_height_layer = None
        self.topography_layer = None
        self.landcover_layer = None        
        self.restriction_zones_layer = None       
        self.elevation_angle_layer = None
        self.los_blck_layer = None
        self.misc_layer = None
        self.range_layer = None
        self.combined_layer = None
        self.intersecting_angle_layer = None
        self.second_lidar_layer = None
        self.aerial_layer = None
        self.layers_info = {}
        

        # Flags as you code
        self.flags = {
                      'measurements_added' : False,
                      'measurements_optimized': False,
                      'measurements_reachable' : False,
                      'mesh_center_added' : False, 
                      'mesh_generated' : False,
                      'utm_set' : False, 
                      'input_check_pass' : False,
                      'output_path_set' : False,
                      'landcover_path_set' : False,
                      'landcover_map_clipped' : False,
                      'landcover_layer_generated' : False,
                      'canopy_height_generated' : False,
                      'restriction_zones_generated' : False,
                      'landcover_layers_generated' : False,
                      'orography_layer_generated' : False,
                      'topography_layer_generated' : False,
                      'beam_coords_generated' : False,
                      'measurements_exported' : False,
                      'topography_exported': False,
                      'viewshed_performed' : False,
                      'viewshed_analyzed' : False,
                      'los_blck_layer_generated' : False,
                      'combined_layer_generated' : False,
                      'intersecting_angle_layer_generated' : False,
                      'second_lidar_layer' : False,
                      'trajectory_optimized' : False,
                      'motion_table_generated' : False,
                     }
  
        CPT.NO_LAYOUTS += 1

    def set_path(self, path_str, **kwargs):
        """
        Sets file paths to the landcover data and for data/results storage.
        """

        if kwargs['path_type'] == 'landcover':
            try:
                self.LANDCOVER_DATA_PATH = Path(r'%s' %path_str)
                if self.LANDCOVER_DATA_PATH.exists():
                    if self.LANDCOVER_DATA_PATH.is_file():
                        print('Path ' + str(self.LANDCOVER_DATA_PATH) + ' set for landcover data')
                        self.flags['landcover_path_set'] = True
                    else:
                        print('Provided path does not point to the landcover data!')
                        self.flags['landcover_path_set'] = False
                else:
                    print('Provided path does not exist!')
                    self.flags['landcover_path_set'] = False
            except:
                print('Uppsss something went wrong!!!')
        elif kwargs['path_type'] == 'output':
            try:
                self.OUTPUT_DATA_PATH = Path(r'%s' %path_str)
                if self.OUTPUT_DATA_PATH.exists():
                    if self.OUTPUT_DATA_PATH.is_dir():
                        print('Path ' + str(self.OUTPUT_DATA_PATH) + ' set for storing CPT outputs')
                        self.flags['output_path_set'] = True
                    else:
                        print('Provided path does not point to directory!')
                        self.flags['output_path_set'] = True
                else:
                    print('Provided path does not exist!')
                    self.flags['landcover_path_set'] = False
            except:
                print('Uppsss something went wrong!!!')
        else: 
            print('Wrong inputs!')   

    @staticmethod
    def __inside_mesh(mesh_corners, point):
        """
        Checks whether the point is inside the mesh.
        """
        diff = mesh_corners - point
        if np.all(diff[0,(0,1)] <= 0) and np.all(diff[1,(0,1)] >= 0):
            return True
        return False

    @staticmethod
    def __check_lidar_position(lidar_position):
        """
        Validates a lidar position
        
        Parameters
        ----------
        lidar_position : ndarray
            nD array containing data with `float` or `int` type
            corresponding to x, y and z coordinates of a lidar.
            nD array data are expressed in meters.
        
        Returns
        -------
            True / False

        See also
        --------
        add_lidar_instance() : adds lidar to the lidar dictionary 

        """        
        if(type(lidar_position).__module__ == np.__name__):
                if (len(lidar_position.shape) == 1 and lidar_position.shape[0] == 3):
                        return True
                else:
                    print('Wrong dimensions!\nLidar position is described by 3 parameters:\n(1)Easting\n(2)Northing\n(3)Height!')
                    print('Lidar position was not added!')
                    return False
        else:
            print('Input is not numpy array!')
            print('Lidar position was not added!')
            return False      
        
    def add_measurement_instances(self, **kwargs):
        """
        Adds measurement points, provided as UTM coordinates,
        to the measurement points dictionary.
        
        Parameters
        ----------
            **kwargs : see below

        Keyword Arguments
        -----------------
        points_id : str
            A string indicating what type of measurements are 
            added to the measurements dictionary.
        points : ndarray, required
            nD array containing data with `float` or `int` type
            corresponding to UTM coordinates of measurement points.
            Shape of points should be (n,3) where n is number of points.
            nD array data are expressed in meters.
        
        Notes
        -----
        UTM zone must be set before calling this method!

        """
        rules = np.array([
            self.flags['utm_set'],
            'points_id' in kwargs,
            (kwargs['points_id'] in self.POINTS_TYPE 
            if 'points_id' in kwargs else False),                          
            'points' in kwargs,
            ((len(kwargs['points'].shape) == 2 and kwargs['points'].shape[1] == 3)
            if 'points' in kwargs else False)])

        print_statements = np.array(['- UTM zone is not set',
                                     '- points_id is not in kwargs',
                                     '- given points_id is not in self.POINTS_TYPE',
                                     '- points is not in kwargs',
                                     '- incorrect shape of points'])
        if np.all(rules):
            points_pd = pd.DataFrame(kwargs['points'], 
                                    columns = ["Easting [m]", 
                                               "Northing [m]",
                                               "Height asl [m]"])

            points_pd.insert(loc=0, column='Point no.', 
                             value=np.array(range(1,len(points_pd) + 1)))
            pts_dict = {kwargs['points_id']: points_pd}
            self.measurements_dictionary.update(pts_dict)

            print('Measurement points \'' 
                  + kwargs['points_id'] 
                  + '\' added to the measurements dictionary!')

            print('Measurements dictionary contains ' 
                   + str(len(self.measurements_dictionary)) 
                   + ' different measurement type(s).')

            self.flags['measurements_added'] = True
            self.points_id = kwargs['points_id']
        else:
            print('Measurement points not added to the dictionary because:')
            print(*print_statements[np.where(rules == False)], sep = "\n")     

    def points_selector(self, points_id):
        """
        Returns a numpy array of measurement points based on the points id.

        Parameters
        ----------
        points_id : str
            A string indicating which measurement points to be returned

        Returns
        -------
        ndarray
            Depending on the points id this method returns one
            of the following measurement points:(1) initial, 
            (2) optimized, (3) reachable, (4) identified or 
            (5) None .

        Notes
        -----
        This method is used during the generation of the beam steering coordinates.
        """        

        if points_id == 'initial':
            return np.asarray(self.measurements_dictionary['initial'].values[:, 1:].tolist())
        elif points_id == 'optimized':
            return np.asarray(self.measurements_dictionary['optimized'].values[:, 1:].tolist())
        elif points_id == 'reachable':
            return np.asarray(self.measurements_dictionary['reachable'].values[:, 1:].tolist())
        elif points_id == 'identified':
            return np.asarray(self.measurements_dictionary['identified'].values[:, 1:].tolist())
        elif points_id == 'misc':
            return np.asarray(self.measurements_dictionary['misc'].values[:, 1:].tolist())
        else:
            return None

    def add_lidar_instance(self, **kwargs):
        """
        Adds a lidar instance, containing lidar position in UTM 
        coordinates and unique lidar id, to the lidar dictionary.
        
        Parameters
        ----------
            **kwargs : see below

        Keyword Arguments
        -----------------
        lidar_id : str, required
            String which identifies lidar.
        position : ndarray, required
            nD array containing data with `float` or `int` type corresponding 
            to Northing, Easting and Height coordinates of the second lidar.
            nD array data are expressed in meters.
        layer_id : str, optional

        points_id : str, optional
        
        Returns
        -------

        Notes
        --------
        Lidar positions can be added one at time.

        """
        if self.flags['utm_set']:
            if 'lidar_id' in kwargs:
                if 'position' in kwargs:
                    if self.__check_lidar_position(kwargs['position']):
                        lidar_dict = {kwargs['lidar_id']:{
                                                'position': kwargs['position'],
                                                'inside_mesh' : False,
                                                'points_id' : None,
                                                'points_position' : None,
                                                'layer_id': None,
                                                'linked_lidars' : [],
                                                'reachable_points' : None,
                                                'trajectory' : None,
                                                'probing_coordinates' : None,
                                                'emission_config': None,      
                                                'motion_config': None,
                                                'acqusition_config': None,
                                                'data_config': None
                                                         }
                                     }
                        self.lidar_dictionary.update(lidar_dict)
                        print('Lidar \'' + kwargs['lidar_id'] 
                              + '\' added to the lidar dictionary,'
                              +' which now contains ' 
                              + str(len(self.lidar_dictionary)) 
                              + ' lidar instance(s).')

                        if ('layer_id' in kwargs):
                            self.update_lidar_instance(**kwargs)
                        elif ('points_id' in kwargs):
                            self.update_lidar_instance(**kwargs)
                    else:
                        print('Incorrect position information, cannot add lidar!')
                else:
                    print('Lidar position not specified, cannot add lidar!')
            else:
                print('Lidar id not provided, cannot add lidar!')
        else:
            print('UTM zone not specified, cannot add lidar!')

    def update_lidar_instance(self, **kwargs):
        """
        Updates an instance in the lidar dictionary with measurement points,
        trajectory, laser configuration, etc.
        
        Parameters
        ----------
        see keyword arguments

        Keyword Arguments
        -----------------
        lidar_id : str, required
            String which identifies the lidar instance to be updated.
        points_id : str, optional
            Indicates which points to be used to update the lidar instance.
        layer_id : str, optional
            String indicating which GIS layer to use
            for the instance update.
            The argument value can be either 'combined or 'second_lidar'.
        trajectory : pd df, optional
            panda dataframe containing trajectory described by UTM coordinates.

        Notes
        --------
        Either provide layer_id or points_id, if both are provided the method
        will consider information related to layer_id when it updated the
        lidar dictionary!

        """
        # checking if lidar_id is in kwargs and
        if ('lidar_id' in kwargs 
            and kwargs['lidar_id'] in self.lidar_dictionary):
            lidar_id = kwargs['lidar_id']
            if ('layer_id' in kwargs and 
                kwargs['layer_id'] in self.layers_info):
                layer_id = kwargs['layer_id']

                self.lidar_dictionary[lidar_id]['layer_id'] = layer_id

                points_id = self.layers_info[layer_id]['points_id']                
                print('Updating lidar instance \'' 
                        + kwargs['lidar_id'] 
                        + '\' considering GIS layer \'' 
                        + layer_id + '\'.')                

                lidar_position = self.lidar_dictionary[lidar_id]['position']

                inside_mesh = self.__inside_mesh(self.mesh_corners_utm, lidar_position)
                self.lidar_dictionary[lidar_id]['inside_mesh'] = inside_mesh

                self.lidar_dictionary[lidar_id]['points_id'] = points_id

                self.lidar_dictionary[lidar_id]['points_position'] = self.measurements_dictionary[points_id]
                if layer_id in self.__SPEC_LAYERS and inside_mesh:
                    layer = self.layer_selector(layer_id)
                    i, j = self.find_mesh_point_index(lidar_position)
                    self.lidar_dictionary[lidar_id]['reachable_points'] = layer[i,j,:]
                    if layer_id == 'second_lidar_placement':
                        linked_lidar = self.layers_info['second_lidar_placement']['lidars_id']
                        self.lidar_dictionary[lidar_id]['linked_lidars'] = (
                                    self.lidar_dictionary[lidar_id]['linked_lidars']
                                    + [linked_lidar])
                        self.lidar_dictionary[linked_lidar]['linked_lidars'] = (
                                    self.lidar_dictionary[linked_lidar]['linked_lidars']
                                    + [lidar_id])
                self.lidar_dictionary[kwargs['lidar_id']]['emission_config'] = {'pulse_length': self.PULSE_LENGTH}
                self.lidar_dictionary[kwargs['lidar_id']]['acqusition_config'] = {'fft_size': self.FFT_SIZE}
            elif ('points_id' in kwargs and 
            kwargs['points_id'] in self.POINTS_TYPE and 
            kwargs['points_id'] in self.measurements_dictionary
            ):  
                points_id = kwargs['points_id']
                print('Updating lidar instance \'' 
                        + kwargs['lidar_id'] 
                        + '\' considering points id \'' 
                        + points_id + '\'.')            
                self.lidar_dictionary[lidar_id]['points_id'] = points_id
                points = self.measurements_dictionary[points_id]
                self.lidar_dictionary[lidar_id]['points_position'] = points
                self.lidar_dictionary[kwargs['lidar_id']]['emission_config'] = {'pulse_length': self.PULSE_LENGTH}
                self.lidar_dictionary[kwargs['lidar_id']]['acqusition_config'] = {'fft_size': self.FFT_SIZE}

            if 'trajectory' in kwargs and kwargs['trajectory'] is not None:
                self.lidar_dictionary[lidar_id]['trajectory'] = kwargs['trajectory']
                    # calculate probing coordinates
                probing_coords = self.generate_beam_coords(
                         self.lidar_dictionary[kwargs['lidar_id']]['position'],
                         kwargs['trajectory'].values[:, 1:], 0)

                probing_coords = pd.DataFrame(
                                      np.round(probing_coords, self.NO_DIGITS), 
                                      columns = ["Azimuth [deg]", 
                                                 "Elevation [deg]", 
                                                 "Range [m]"])
                probing_coords.insert(loc=0, column='Point no.',
                              value=np.array(range(1,len(probing_coords) + 1)))
                
                self.lidar_dictionary[lidar_id]['probing_coordinates'] = probing_coords
                    # calculate motion config table
                self.lidar_dictionary[lidar_id]['motion_config'] = self.generate_trajectory(
                        self.lidar_dictionary[lidar_id]['position'], 
                        self.lidar_dictionary[lidar_id]['trajectory'].values[:,1:])
        else:
            print('lidar_id not provided')
            print('Aborting the operation!')


    # def update_lidar_dictionary(self, **kwargs):
    #     """
    #     Updates all instances in lidar dictionary with 
    #     measurement points, trajectory and lidar configuration.
        
    #     Parameters
    #     ----------
    #         **kwargs : see below

    #     Keyword Arguments
    #     -----------------
    #     use_reachable_points : boolean, optional
    #         Indicates whether to update the lidar instance
    #         only considering the reachable points.
    #     layer_id : str, optional
    #         String indicating which GIS layer to use
    #         for the instance update.
    #         The argument value can be either 'combined or 'second_lidar'.
    #     use_optimized_trajectory: boolean, optional
    #         Indicates whether to use the optimized  trajectory for
    #         to update the lidar dictionary. 
    #     motion_type : str, optional
    #         String indicating which type of motion should be used to 
    #         generate trajetory between measurement points.
    #         The argument takes either 'step-stare' or 'sweep' value.

    #     Returns
    #     -------

    #     Notes
    #     --------
    #     If 'only_reachable_points' is not provided, the method
    #     will consider all the measurement points during the instance update.

    #     If 'only_reachable_points' is set to True, the method requires that the
    #     'layer_id' points to either 'combined' or 'second_lidar' layer. If
    #     'layer_id' is not provided the method will use 'combined' layer.

    #     If 'use_optimized_trajectory' is set to True, it is required that the 
    #     method self.optimize_trajectory was run prior the current method, 
    #     otherwise the current method will update the lidar instance considering
    #     the order of measurement points as is.

    #     Currently the method only support step-stare trajectory, so the argument
    #     wheter on not set 'motion_type' it will not impact the trajectory calculation.


    #     Examples
    #     --------

    #     """

    #     if ('points_id' in kwargs and 
    #         kwargs['points_id'] in self.POINTS_TYPE and 
    #         kwargs['points_id'] in self.measurements_dictionary
    #         ):
    #         kwargs.update({'points_id' : kwargs['points_id']})
    #         kwargs.update({'lidar_id' : ''})


    #         for lidar in self.lidar_dictionary:
    #             kwargs['lidar_id'] = lidar
    #             self.update_lidar_instance(**kwargs)

    #     else:
    #         print('Either the points_id was not provided or no points exists for the given points_id!')
    #         print('Halting the current operation!')


    


    