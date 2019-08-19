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

    Methods
    --------
    add_lidars(self, **kwargs)
        Adds lidars positions to the CPT class instance.   
    add_measurements(self, **kwargs)
        Adds measurement positions to the CPT class instance.
    optimize_measurements(self)
        Optimizes measurement positions by solving disc covering problem.
    set_utm_zone(utm_zone)
        Sets UTM grid zone and corresponding EPSG code to the CPT instance.
    which_hemisphere(utm_zone)
        Returns whether UTM grid zone belongs to the Northern or Southern hemisphere.
    utm2epsg(utm_zone)
        Converts UTM grid zone to EPSG code.
    utm2geo(points_utm, long_zone, hemisphere)
        Converts UTM to GEO coordinates.                
    generate_mesh(self, **kwargs)
        Generates a rectangular horizontal mesh containing equally spaced points.
    generate_beam_coords_mesh(self, **kwargs):
        Generates beam steering coordinates from every mesh point to every measurement point.
    generate_combined_layer(self, **kwargs):
        Generates the combined GIS layer which used for the lidar positioning.
    generate_elevation_restriction_layer
        Generates elevation restricted GIS layer.
    generate_range_restriction_layer
        Generates range restricted GIS layer.
    generate_los_blck_layer(self, **kwargs):
        Generates the los blockage GIS layer.
    generate_topographic_layer(self)
        Generates topographic GIS layer (i.e., canopy + orography height).
    generate_orography_layer(self)
        Generates orography GIS layer.
    generate_landcover_layer(self):
        Generates restriction zones and canopy height 
        GIS layers based on the CORINE landcover data.
    plot_layer(self, layer, **kwargs)
        Plots an individual GIS layer.
    plot_optimization(self, **kwargs)
        Plots results of measurement point optimization.
    """

    NO_LAYOUTS = 0

    NO_DATA_VALUE = 0
    LANDCOVER_DATA_PATH = ""
    OUTPUT_DATA_PATH = ""    
    GOOGLE_API_KEY = ""
    FILE_EXTENSIONS = np.array(['.tif', '.tiff', '.pdf', '.kml', '.png', '.pmc', '.xml' , '.yaml'])

    MESH_RES = 100 # in m
    MESH_EXTENT = 5000 # in m
    REP_RADIUS = 500 # in m
    POINTS_TYPE = np.array(['initial', 'optimized', 'reachable', 'identified', 'misc'])
    LAYER_TYPE = np.array([
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
    ])
    COLOR_LIST = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
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
        self.measurements_selector = 'initial'
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
        self.lidar_pos_1 = None        
        self.lidar_pos_2 = None

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
        self.layer_creation_info = {}
        

        # Flags as you code
        self.flags = {
                      'measurements_added' : False,
                      'measurements_optimized': False,
                      'measurements_reachable' : False,
                      'lidar_pos_1' : False,
                      'lidar_pos_2' : False,
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
        
    def add_measurement_instances(self, **kwargs):
        """
        Adds measurement points, given in as UTM coordinates,
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
            nD array data are expressed in meters.
        
        Returns
        -------

        Notes
        --------

        Examples
        --------
        >>> layout = CPT()
        >>> layout.set_utm_zone('33T')
        Correct latitudinal zone!
        Correct longitudinal zone!
        UTM zone set
        >>> layout.add_measurement_instances(points = np.array([[576697, 4845753, 395 + 80], 
        [577979, 4844819, 478 + 80],]), points_id = 'initial')
        Measurement points 'initial' added to the measurements dictionary!
        Measurements dictionary contains 1 different measurement type(s).

        """
        rules = np.array([self.flags['utm_set'],
                          'points_id' in kwargs,
                          kwargs['points_id'] in self.POINTS_TYPE,
                          'points' in kwargs])
        print_statements = np.array(['- UTM zone is not set',
                                     '- points_id is not in kwargs',
                                     '- points_id is not in self.POINTS_TYPE',
                                     '- points is not in kwargs'])
        if np.all(rules):
            if len(kwargs['points'].shape) == 2 and kwargs['points'].shape[1] == 3:

                points_pd = pd.DataFrame(kwargs['points'], 
                                        columns = ["Easting [m]", "Northing [m]","Height asl [m]"])

                points_pd.insert(loc=0, column='Point no.', value=np.array(range(1,len(points_pd) + 1)))
                pts_dict = {kwargs['points_id']: points_pd}
                self.measurements_dictionary.update(pts_dict)

                print('Measurement points \'' + kwargs['points_id'] + '\' added to the measurements dictionary!')
                print('Measurements dictionary contains ' + str(len(self.measurements_dictionary)) + ' different measurement type(s).')
                self.flags['measurements_added'] = True
                self.measurements_selector = kwargs['points_id']
            else:
                print('Incorrect position information, cannot add measurements!')
                print('Input measurement points must be a numpy array of shape (n,3) where n is number of points!')
        else:
            print('Measurement points not added to the dictionary because:')
            print(*print_statements[np.where(rules == False)], sep = "\n")     
            

        # if self.flags['utm_set']:
        #     if 'points_id' in kwargs:
        #         if kwargs['points_id'] in self.POINTS_TYPE:
        #             if 'points' in kwargs:
        #                 if len(kwargs['points'].shape) == 2 and kwargs['points'].shape[1] == 3:

        #                     points_pd = pd.DataFrame(kwargs['points'], 
        #                                             columns = ["Easting [m]", "Northing [m]","Height asl [m]"])

        #                     points_pd.insert(loc=0, column='Point no.', value=np.array(range(1,len(points_pd) + 1)))
        #                     pts_dict = {kwargs['points_id']: points_pd}
        #                     self.measurements_dictionary.update(pts_dict)

        #                     print('Measurement points \'' + kwargs['points_id'] + '\' added to the measurements dictionary!')
        #                     print('Measurements dictionary contains ' + str(len(self.measurements_dictionary)) + ' different measurement type(s).')
        #                     self.flags['measurements_added'] = True
        #                     self.measurements_selector = kwargs['points_id']
        #                 else:
        #                     print('Incorrect position information, cannot add measurements!')
        #                     print('Input measurement points must be a numpy array of shape (n,3) where n is number of points!')
        #             else:
        #                 print('Measurement points not specified, cannot add points!')
        #         else:
        #             print('Measurement point type not permitted!')
        #             print('Allowed types are: \'initial\', \'optimized\', \'reachable\', \'identified\' and \'misc\'')
        #     else:
        #         print('Measurement points\' type not provided, cannot add measurement points!')
        # else:
        #     print('UTM zone not specified, cannot add measurement points!')

    def measurement_type_selector(self, points_id):
        """
        Selects measurement type.

        Parameters
        ----------
        points_id : str
            A string indicating which measurement points to be returned

        Returns
        -------
        measurement_points : ndarray
            Depending on the input type this method returns one
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

    @staticmethod
    def check_measurement_positions(points):
        """
        Validates input points.
        
        Parameters
        ----------
        points : ndarray
            nD array containing data with `float` or `int` type
            corresponding to x, y and z coordinates of measurement points.
            nD array data are expressed in meters.

        
        Returns
        -------
            True / False

        See also
        --------
        add_measurements() : adding measurement points to the CPT class instance 

        Examples
        --------
        >>> layout = CPT()        
        >>> layout.check_measurement_positions(np.array([1,2]))
        Wrong dimensions!
        Measurement positions were not added
        False      
        >>> layout.check_measurement_positions(np.array([1,2,1]))
        True              
        >>> layout.check_measurement_positions([1,2,1])
        Input is not numpy array!
        Measurement positions were not added
        False        
        """
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

    def update_reachable_points(self, **kwargs):

        
        if ('points_id' in kwargs and 
            kwargs['points_id'] in self.POINTS_TYPE and 
            self.measurements_dictionary[kwargs['points_id']] is not None
        ):
            if ('lidar_ids' in kwargs and set(kwargs['lidar_ids']).issubset(self.lidar_dictionary)):
                flag = True
                measurement_id = []

                for lidar in kwargs['lidar_ids']:
                    reachable_pts = self.lidar_dictionary[lidar]['reachable_points']
                    if reachable_pts is None:
                        flag = False

                    else:
                        measurement_id = measurement_id + [self.lidar_dictionary[lidar]['measurement_id']]

                if flag:
                    if all(x == measurement_id[0] for x in measurement_id):
                        flag = True
                    else:
                        flag = False
                        print('One or more lidar instances was not updated with the same measurement points!')
                        print('Halting operation!')
                else:
                    print('One or more lidar instances was not updated!')
                    print('Halting operation!')
                    print('Run self.update_lidar_dictionary() to update all lidar instances!')
                    print('Run self.update_lidar_instance(lidar_id = name) to a single lidar instance!')

                if flag:
                    if kwargs['points_id'] == measurement_id[0]:
                        print('Finding reachable points which are common for lidar instances:' + str(kwargs['lidar_ids']))
                        measurement_pts = self.measurement_type_selector(kwargs['points_id'])
                        self.measurement_selector = 'reachable'
                        all_ones = np.full(len(measurement_pts),1)
                        for lidar in kwargs['lidar_ids']:
                            reachable_pts = self.lidar_dictionary[lidar]['reachable_points']
                            all_ones = all_ones * reachable_pts            
                        pts_ind = np.where(all_ones == 1)
                        print('Updating self.measurements_dictionary instance \'reachable\' with common reachable points')
                        self.add_measurement_instances(points = measurement_pts[pts_ind], 
                                                       points_id = 'reachable')
                        print('Optimizing trajectory through the common reachable points for lidar instances:' + str(kwargs['lidar_ids']))
                        self.optimize_trajectory(points_id = 'reachable', 
                                                 lidar_ids = kwargs['lidar_ids'])
                                              
                        print('Lidar instances:' + str(kwargs['lidar_ids']) + ' will be updated with the common reachable points and optimized trajectory')
                        for lidar in kwargs['lidar_ids']:
                            self.update_lidar_instance(lidar_id = lidar, use_optimized_trajectory = True, points_id = 'reachable')

                        self.sync_trajectory(**kwargs)                            
                        
                        # should run to sync timing between windscanners!

                    else:
                        print('Lidar instances \'measurement_id\' does not match \'points_id\' keyword paramater')
                        print('Halting operation!')
            else:
                print('One or more lidar ids don\'t exist in the lidar dictionary')
                print('Available lidar ids: ' + str(list(self.lidar_dictionary.keys())))
        else:
            print('Either point type id does not exist or for the corresponding measurement dictionary instance there are no points!')


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
        
        Returns
        -------

        Notes
        --------
        Lidar positions can be added one at time.

        Examples
        --------
        >>> layout = CPT()
        >>> layout.set_utm_zone('33T')
        Correct latitudinal zone!
        Correct longitudinal zone!
        UTM zone set
        >>> layout.add_lidar(position = np.array([580600,4845700,100]), lidar_id = 'koshava')
        Lidar 'koshava' added to the class instance!      
        """
        if self.flags['utm_set']:
            if 'lidar_id' in kwargs:
                if 'position' in kwargs:
                    if self.check_lidar_position(kwargs['position']):
                        lidar_dict = {kwargs['lidar_id']:{
                                                      'position': kwargs['position'],
                                                      'lidar_inside_mesh' : False,
                                                      'measurement_id' : None,
                                                      'measurement_points' : None,
                                                      'layer_id': None,
                                                      'linked_lidar' : None,
                                                      'reachable_points' : None,
                                                      'trajectory' : None,
                                                      'probing_coordinates' : None,
                                                      'emission_config': None,      
                                                      'motion_config': None,
                                                      'acqusition_config': None,
                                                      'data_config': None}
                                     }
                        self.lidar_dictionary.update(lidar_dict)
                        print('Lidar \'' + kwargs['lidar_id'] + '\' added to the lidar dictionary, which now contains ' + str(len(self.lidar_dictionary)) + ' lidar instance(s).')
                        # print('Lidar dictionary contains ' + str(len(self.lidar_dictionary)) + ' lidar instance(s).')
                    else:
                        print('Incorrect position information, cannot add lidar!')
                else:
                    print('Lidar position not specified, cannot add lidar!')
            else:
                print('Lidar id not provided, cannot add lidar!')
        else:
            print('UTM zone not specified, cannot add lidar!')

    def update_lidar_dictionary(self, **kwargs):
        """
        Updates all instances in lidar dictionary with 
        measurement points, trajectory and lidar configuration.
        
        Parameters
        ----------
            **kwargs : see below

        Keyword Arguments
        -----------------
        use_reachable_points : boolean, optional
            Indicates whether to update the lidar instance
            only considering the reachable points.
        layer_id : str, optional
            String indicating which GIS layer to use
            for the instance update.
            The argument value can be either 'combined or 'second_lidar'.
        use_optimized_trajectory: boolean, optional
            Indicates whether to use the optimized  trajectory for
            to update the lidar dictionary. 
        motion_type : str, optional
            String indicating which type of motion should be used to 
            generate trajetory between measurement points.
            The argument takes either 'step-stare' or 'sweep' value.

        Returns
        -------

        Notes
        --------
        If 'only_reachable_points' is not provided, the method
        will consider all the measurement points during the instance update.

        If 'only_reachable_points' is set to True, the method requires that the
        'layer_id' points to either 'combined' or 'second_lidar' layer. If
        'layer_id' is not provided the method will use 'combined' layer.

        If 'use_optimized_trajectory' is set to True, it is required that the 
        method self.optimize_trajectory was run prior the current method, 
        otherwise the current method will update the lidar instance considering
        the order of measurement points as is.

        Currently the method only support step-stare trajectory, so the argument
        wheter on not set 'motion_type' it will not impact the trajectory calculation.


        Examples
        --------

        """

        if ('points_id' in kwargs and 
            kwargs['points_id'] in self.POINTS_TYPE and 
            kwargs['points_id'] in self.measurements_dictionary
            ):
            kwargs.update({'points_id' : kwargs['points_id']})
            kwargs.update({'lidar_id' : ''})


            for lidar in self.lidar_dictionary:
                kwargs['lidar_id'] = lidar
                self.update_lidar_instance(**kwargs)

        else:
            print('Either the points_id was not provided or no points exists for the given points_id!')
            print('Halting the current operation!')


    def update_lidar_instance(self, **kwargs):
        """
        Updates a instance(s) in lidar dictionary with
        measurement points lidar instance, containing lidar position in UTM 
        coordinates and unique lidar id, to the lidar dictionary.
        
        Parameters
        ----------
            **kwargs : see below

        Keyword Arguments
        -----------------
        lidar_id : str, required
            String which identifies the lidar instance to be updated.
        points_id : str, optional
            Indicates which points to be used to update the lidar instace.
        use_reachable_points : boolean, optional
            Indicates whether to update the lidar instance
            only considering the reachable points.
        layer_id : str, optional
            String indicating which GIS layer to use
            for the instance update.
            The argument value can be either 'combined or 'second_lidar'.
        use_optimized_trajectory: boolean, optional
            Indicates whether to use the optimized  trajectory for
            to update the lidar dictionary. 
        motion_type : str, optional
            String indicating which type of motion should be used to 
            generate trajetory between measurement points.
            The argument takes either 'step-stare' or 'sweep' value.

        Returns
        -------

        Notes
        --------
        If 'only_reachable_points' is not provided, the method
        will consider all the measurement points during the instance update.

        If 'only_reachable_points' is set to True, the method requires that the
        'layer_id' points to either 'combined' or 'second_lidar' layer. If
        'layer_id' is not provided the method will use 'combined' layer.

        If 'use_optimized_trajectory' is set to True, it is required that the 
        method self.optimize_trajectory was run prior the current method, 
        otherwise the current method will update the lidar instance considering
        the order of measurement points as is.

        Currently the method only support step-stare trajectory, so the argument
        wheter on not set 'motion_type' it will not impact the trajectory calculation.


        Examples
        --------

        """
        if ('points_id' in kwargs and 
            kwargs['points_id'] in self.POINTS_TYPE and 
            kwargs['points_id'] in self.measurements_dictionary
            ):
            measurement_pts = self.measurement_type_selector(kwargs['points_id'])
            self.measurements_selector = kwargs['points_id']

            if len(measurement_pts) > 0:
                if 'lidar_id' in kwargs and kwargs['lidar_id'] in self.lidar_dictionary:
                    # selects the according lidar
                    # sets measurement_id
                    print('Updating lidar instance \'' + kwargs['lidar_id'] + '\' considering measurement type \'' + self.measurements_selector + '\'.') 
                    self.lidar_dictionary[kwargs['lidar_id']]['measurement_id'] = self.measurements_selector
    
                    self.lidar_dictionary[kwargs['lidar_id']]['measurement_points'] = self.measurements_dictionary[self.measurements_selector]
    
                    if self.flags['mesh_generated']:
                        lidar_position = self.lidar_dictionary[kwargs['lidar_id']]['position']
                        self.lidar_dictionary[kwargs['lidar_id']]['lidar_inside_mesh'] = self.inside_mesh(self.mesh_corners_utm, lidar_position)
    
                        
                        if  (
                                self.lidar_dictionary[kwargs['lidar_id']]['lidar_inside_mesh'] and
                                'layer_id' in kwargs and
                                (kwargs['layer_id'] == 'combined' or 
                                kwargs['layer_id'] == 'second_lidar_placement') and
                                self.layer_selector(kwargs['layer_id']) is not None
                            ):
                            layer = self.layer_selector(kwargs['layer_id'])
                            i, j = self.find_mesh_point_index(self.lidar_dictionary[kwargs['lidar_id']]['position'])
                            self.lidar_dictionary[kwargs['lidar_id']]['reachable_points'] = layer[i,j,:]
                            self.lidar_dictionary[kwargs['lidar_id']]['layer_id'] =  kwargs['layer_id']
                            if kwargs['layer_id'] == 'second_lidar_placement':
                                linked_lidar = self.layer_creation_info['second_lidar_placement']['lidars_id']

                                self.lidar_dictionary[kwargs['lidar_id']]['linked_lidar'] = linked_lidar

                        elif (
                            self.lidar_dictionary[kwargs['lidar_id']]['lidar_inside_mesh'] and 
                            self.layer_selector('combined') is not None
                             ):
                            layer = self.layer_selector('combined')
                            i, j = self.find_mesh_point_index(self.lidar_dictionary[kwargs['lidar_id']]['position'])
                            self.lidar_dictionary[kwargs['lidar_id']]['reachable_points'] = self.combined_layer[i,j,:]                     
                        else:
                            self.lidar_dictionary[kwargs['lidar_id']]['reachable_points'] = np.full(len(measurement_pts),0.0)                    
                    
                    if  (
                          'use_optimized_trajectory' in kwargs and 
                          kwargs['use_optimized_trajectory'] and
                          self.trajectory is not None
                        ):
                        self.lidar_dictionary[kwargs['lidar_id']]['trajectory'] = self.trajectory
    
                    elif (
                          'use_reachable_points' in kwargs and 
                          kwargs['use_reachable_points']
                       ):
                        reachable_pts = self.lidar_dictionary[kwargs['lidar_id']]['reachable_points']
    
                        pts_subset = measurement_pts[np.where(reachable_pts > 0)]
                        pts_subset = pd.DataFrame(pts_subset, columns = ["Easting [m]", 
                                                                        "Northing [m]", 
                                                                        "Height asl [m]"])
    
                        pts_subset.insert(loc=0, column='Point no.', value=np.array(range(1,len(pts_subset) + 1)))                                    
                        self.lidar_dictionary[kwargs['lidar_id']]['trajectory'] = pts_subset
                    else:
                        self.lidar_dictionary[kwargs['lidar_id']]['trajectory'] = self.lidar_dictionary[kwargs['lidar_id']]['measurement_points']
                    
                    # calculate probing coordinates
                    probing_coords = self.generate_beam_coords(self.lidar_dictionary[kwargs['lidar_id']]['position'],
                                                               self.lidar_dictionary[kwargs['lidar_id']]['trajectory'].values[:, 1:],
                                                               0)
    
                    probing_coords = pd.DataFrame(np.round(probing_coords, self.NO_DIGITS), 
                                                    columns = ["Azimuth [deg]", 
                                                               "Elevation [deg]", 
                                                               "Range [m]"])
                    probing_coords.insert(loc=0, column='Point no.', value=np.array(range(1,len(probing_coords) + 1)))
                    
                    self.lidar_dictionary[kwargs['lidar_id']]['probing_coordinates'] = probing_coords
    
                    # calculate motion config table
                    self.lidar_dictionary[kwargs['lidar_id']]['motion_config'] = self.generate_trajectory(
                        self.lidar_dictionary[kwargs['lidar_id']]['position'], 
                        self.lidar_dictionary[kwargs['lidar_id']]['trajectory'].values[:,1:])
                    
                    # calculate range gate table
                    self.lidar_dictionary[kwargs['lidar_id']]['emission_config'] = {'pulse_length': self.PULSE_LENGTH}
                    self.lidar_dictionary[kwargs['lidar_id']]['acqusition_config'] = {'fft_size': self.FFT_SIZE}                    
                else:
                    print('The provided lidar_id does not match any lidar instance in lidar dictionary!')
            else:
                print('There are no measurement points -> halting lidar instance/dictionary update!')

        else:
            print('Either the points_id was not provided or no points exists for the given points_id!')
            print('Halting the current operation!')



    
    @staticmethod
    def inside_mesh(mesh_corners, point):
        diff = mesh_corners - point
        if np.all(diff[0,(0,1)] <= 0) and np.all(diff[1,(0,1)] >= 0):
            return True
        return False

    @staticmethod
    def check_lidar_position(lidar_position):
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
        add_lidars() : adding lidar positions to the CPT class instance 

        Examples
        --------
        >>> layout = CPT()
        >>> layout.check_lidar_position(np.array([1,2,100]))
        True
        >>> layout.check_lidar_position(np.array([1,2]))
        Wrong dimensions!
        Lidar position is described by 3 parameters:
        (1)Easting
        (2)Northing
        (3)Height
        Lidar position was not added
        False
        >>> layout.check_lidar_position([1,2])
        Input is not numpy array!
        Lidar position was not added
        False        
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


    