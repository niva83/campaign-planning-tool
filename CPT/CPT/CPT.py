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

    Returns
    -------
    out : ndarray
        nD array containing data with 'float' or 'int' type
    
    Examples
    --------
    >>> A = np.array([[1,2,3],[1,1,1]])
    >>> B = np.array([[3,3,3],[3,2,3],[1,1,1]])
    >>> array_difference(A, B)
    array([[1, 2, 3]])

    >>> A = np.array([[1,2,3],[1,1,1]])
    >>> B = np.array([[1,2,3],[3,3,3],[3,2,3],[1,1,1]])
    >>> array_difference(A,B)
    array([], dtype=float64)
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
    """
    A class for designing scanning lidar measurement campaigns.

    ...

    Attributes
    ----------
    LANDCOVER_DATA_PATH : str
        The path to a CORINE landcover dataset.
        A default value set to an empty string.
    GOOGLE_API_KEY : str
        An API key to access Google Maps database.
        A default value is set to an empty string.
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
    MAX_ELEVATION_ANGLE : float
        The maximum allowed elevation angle for a beam steering.
        The angle is expressed in deg.
        A default value is set to 5 deg.
    MIN_INTERSECTING_ANGLE : float
        The minimum intersecting angle between two beams.
        The angle is expressed in deg.
        A default value is set to 30 deg.
    AVERAGE_RANGE : int
        The average range of lidars.
        The range is expressed in m.
        A default value is set to 3000 m.    
    MAX_ACCELERATION : int
        The maximum acceleration of scanner head.
        The acceleration is expressed in deg/s^2.
        A default value is set to 100 deg/s^2.
    NO_LAYOUTS : int
        A number of layout instances generated.
    ...need to add all the self.attributes!!!!

    Methods
    --------
    set_utm_zone(utm_zone)
        Sets UTM grid zone and EPSG code to the CPT instance.
    check_utm_zone(utm_zone)
        Checks whether UTM grid zone is valid or not.
    which_hemisphere(utm_zone)
        Returns whether UTM grid zone belongs to the Northern or Southern hemisphere.
    utm2epsg(utm_zone)
        Converts UTM grid zone to EPSG code.
    add_measurements(self, **kwargs)
        Adds measurement positions to the CPT class instance.
    generate_disc_matrix(self)
        Generates inputs for optimize_measurements() method
    optimize_measurements(self)
        Optimizes measurement positions by solving disc covering problem.
    add_lidars(self, **kwargs)
        Adds lidars positions to the CPT class instance.
    generate_mesh(self, **kwargs)
        Generates a rectangular horizontal mesh containing equally spaced points.
    check_measurement_positions(points)
        Validates input measurement points.
    check_lidar_position(lidar_position)
        Validates input lidar positions.
    """
    INPUT_DATA_PATH = ""
    OUTPUT_DATA_PATH = ""
    LANDCOVER_DATA_PATH = ""
    GOOGLE_API_KEY = ""
    
    MESH_RES = 100 # in m
    MESH_EXTENT = 5000 # in m

    
    REP_RADIUS = 500 # in m
    MAX_ELEVATION_ANGLE = 5 # in deg
    MIN_INTERSECTING_ANGLE = 30 # in deg
    AVERAGE_RANGE = 3000 # in m
    MAX_ACCELERATION = 100 # deg / s^2

    NO_LAYOUTS = 0

    def __init__(self, **kwargs):
        # flags
        # add missing flags as you code
        self.flags = {'topography':False, 'landcover':False, 'exclusions': False,    
                                    'viewshed':False, 'elevation_angle': False, 'range': False, 
                                    'intersecting_angle':False, 'measurements_optimized': False,
                                    'measurements_added': False,'lidar_pos_1':False,'lidar_pos_2':False,
                                    'mesh_center_added': False,
                                    'utm':False, 'input_check_pass': False}        


        # measurement positions
        self.measurements_initial = None
        self.measurements_optimized = None
        self.measurements_identified = None
        self.measurements_reachable = None


        if not 'mesh_center' in kwargs:
            self.mesh_center = None 
        else:
            self.mesh_center = kwargs['mesh_center']

        if not 'utm_zone' in kwargs:
            self.long_zone = None
            self.lat_zone = None
            self.epsg_code = None 
        else:
            if self.check_utm_zone(kwargs['utm_zone']):
                self.long_zone = kwargs['utm_zone'][:-1]
                self.lat_zone = kwargs['utm_zone'][-1].upper() 
                self.epsg_code = self.utm2epsg(kwargs['utm_zone']) 
                self.flags['utm'] = True
            else:
                self.long_zone = None
                self.lat_zone = None
                self.epsg_code = None                

        
        # lidar positions
        self.lidar_pos_1 = None        
        self.lidar_pos_2 = None

        
        # GIS layers
        self.mesh_corners = None
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
        """
        Sets UTM grid zone and EPSG code to the CPT instance. 
        
        Parameters
        ----------
        utm_zone : str, optional
            A string representing an UTM grid zone, containing digits (from 1 to 60) 
            indicating the longitudinal zone followed by a character ('C' to 'X' excluding 'O') 
            corresponding to the latitudinal zone.

        Returns
        -------
        self.long_zone : str
            A string representing longitudinal zone of the UTM grid zone.
        self.lat_zone : str
            A character representing latitudinal zone of the UTM grid zone.
        self.epsg_code : str
            A string representing EPSG code.
        self.flags['utm'] : bool
            Sets the key 'utm' in the flag dictionary to True.                
        """
        if self.check_utm_zone(utm_zone):
            self.long_zone = utm_zone[:-1]
            self.lat_zone = utm_zone[-1].upper() 
            self.epsg_code = self.utm2epsg(utm_zone) 
            self.flags['utm'] = True
            return print('UTM zone set')
        else:
            return print('UTM zone not set')
        
    def add_measurements(self, **kwargs):
        """
        Adds measurement positions, provided as 
        UTM coordinate triplets, to the CPT class.
        
        Parameters
        ----------
            **kwargs : see below

        Keyword Arguments
        -----------------
        measurements : ndarray, required
            nD array containing data with `float` or `int` type
            corresponding to x, y and z coordinates of measurement points.
            nD array data are expressed in meters.
        utm_zone : str, optional
            A string representing an UTM grid zone, containing digits (from 1 to 60) 
            indicating the longitudinal zone followed by a character ('C' to 'X' excluding 'O') 
            corresponding to the latitudinal zone.
        
        Returns
        -------
        self.measurements_initial : ndarray 
            nD array containing data with `float` or `int` type corresponding to 
            Northing, Easting and Height coordinates of measurement points.
        self.flags['measurements_added'] : bool
            Sets the key 'measurements_added' in the flag dictionary to True.
        self.long_zone : str
            A string representing longitudinal zone of the UTM grid zone.
        self.lat_zone : str
            A character representing latitudinal zone of the UTM grid zone.
        self.epsg_code : str
            A string representing EPSG code.
        self.flags['utm'] : bool
            Sets the key 'utm' in the flag dictionary to True.
        """
        if self.flags['measurements_added']:
            print('Existing measurement points will be overwritten!')

        if 'utm_zone' in kwargs:
            self.set_utm_zone(kwargs['utm_zone'])

        if self.flags['utm'] == False:
            print('Cannot add measurement points without specificing UTM zone!')

        if self.flags['utm'] and self.check_measurement_positions(kwargs['measurements']):
            if len(kwargs['measurements'].shape) == 2:
                    self.measurements_initial = np.unique(kwargs['measurements'], axis=0)
            else:
                    self.measurements_initial = np.array([kwargs['measurements']])
                    self.measurements_initial = np.unique(self.measurements_initial, axis=0)
            self.flags['measurements_added'] = True

    def generate_disc_matrix(self):
        """
        Generates mid points between any combination of two measurement points 
        which act as disc centers. The mid points are tested which measurement
        points they are covering producing so-called disc-covering matrix used
        in the measuremen point optimization method.

        Parameters
        ----------
        self.measurements_initial : ndarray
            An initial set of measurement points provided as nD array.
        self.REP_RADIUS : int
            MEASNET's representativness radius of measurements.
            The radius is expressed in meters.
            A default value is set to 500 m.
        Returns
        -------
        discs : ndarray
            An array of mid points between all combinations of two measurement points.
        matrix : ndarray
            A binary matrix indicating which measurement points are covered by each disc.
            The matrix shape is (len(discs), len(measurements_initial)).
            The value 1 indicates that a measurement point is covered or by a disc.
            The value 0 indicates that a measurement point is not covered or by a disc.
            The matrix is sorted decending order, having a row with a maximum number of 1
            positioned at the top.

        See also
        --------
        optimize_measurements : implementation of disc covering problem

        Notes
        --------
        generate_disc_matrix() method is used to generate necessary inputs for the
        greedy implementation of the disc covering problem which optimizes the 
        measurement points for the field campaign. It is required that measurement
        points are added to the class instance before calling this method.

        References
        ----------
        .. [1] Ahmad Biniaz, Paul Liu, Anil Maheshwari and Michiel Smid, 
           Approximation algorithms for the unit disk cover problem in 2D and 3D,
           Computational Geometry, Volume 60, Pages 8-18, 2017,
           https://doi.org/10.1016/j.comgeo.2016.04.002.

        Examples
        --------
        >>> layout = CPT()
        >>> layout.set_utm_zone('31V')
        Correct latitudinal zone!
        Correct longitudinal zone!
        UTM zone set        
        >>> layout.REP_RADIUS = 2
        >>> measurements_initial = np.array([[1,3,3],[2,1,4],[5,2,1],[0,7.4,1],[7.4,4,1]]) 
        >>> layout.add_measurements(measurements = measurements_initial)
        >>> layout.generate_disc_matrix()
        (array([[1.5, 2. , 0. ],
            [3.5, 1.5, 0. ],
            [6.2, 3. , 0. ],
            [1. , 4.2, 0. ],
            [3. , 2.5, 0. ],
            [4.2, 3.5, 0. ],
            [4.7, 2.5, 0. ],
            [0.5, 5.2, 0. ],
            [2.5, 4.7, 0. ],
            [3.7, 5.7, 0. ]]), array([[0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]))

        """
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
            return print("No measurement positions added, nothing to optimize!")

    def optimize_measurements(self):
        """
        Optimizes measurement positions by solving disc covering problem.
        
        Parameters
        ----------
        
        Returns
        -------
        self.measurements_optimized : ndarray
            An nD array of optimized measurements positions.
        
        See also
        --------
        generate_disc_matrix : method which calculates inputs for optimize_measurements()

        Notes
        --------
        A greedy implementation of the disc covering problem for a set of measurement points.

        References
        ----------
        .. [1] Ahmad Biniaz, Paul Liu, Anil Maheshwari and Michiel Smid, 
           Approximation algorithms for the unit disk cover problem in 2D and 3D,
           Computational Geometry, Volume 60, Pages 8-18, 2017,
           https://doi.org/10.1016/j.comgeo.2016.04.002.

        Examples
        --------
        >>> layout = CPT()
        >>> layout.set_utm_zone('31V')
        Correct latitudinal zone!
        Correct longitudinal zone!
        UTM zone set
        >>> layout.REP_RADIUS = 2
        >>> measurements_initial = np.array([[1,3,3],[2,1,4],[5,2,1],[0,7.4,1],[7.4,4,1]]) 
        >>> layout.add_measurements(measurements = measurements_initial)
        >>> layout.optimize_measurements()
        >>> layout.measurements_optimized
        array([[1.5, 2. , 0. ],
            [3.5, 1.5, 0. ],
            [6.2, 3. , 0. ],
            [0. , 7.4, 1. ]])

        """

        if self.flags['measurements_added']:
            discs, matrix = self.generate_disc_matrix()
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
        """
        Adds lidars positions, provided as 
        UTM coordinate triplets, to the CPT class.
        
        Parameters
        ----------
            **kwargs : see below

        Keyword Arguments
        -----------------
        lidar_pos_1 : ndarray, required
            nD array containing data with `float` or `int` type corresponding 
            to Northing, Easting and Height coordinates of the first lidar.
            nD array data are expressed in meters.
        lidar_pos_2 : ndarray, required
            nD array containing data with `float` or `int` type corresponding 
            to Northing, Easting and Height coordinates of the second lidar.
            nD array data are expressed in meters.
        utm_zone : str, optional
            A string representing an UTM grid zone, containing digits (from 1 to 60) 
            indicating the longitudinal zone followed by a character ('C' to 'X' excluding 'O') 
            corresponding to the latitudinal zone.
        
        Returns
        -------
        self.lidar_pos_1 : ndarray 
            nD array containing data with `float` or `int` type corresponding to 
            Northing, Easting and Height coordinates of the first lidar.
        self.lidar_pos_2 : ndarray 
            nD array containing data with `float` or `int` type corresponding to 
            Northing, Easting and Height coordinates of the second lidar.            
        self.flags['lidar_pos_1'] : bool
            Sets the key 'lidar_pos_1' in the flag dictionary to True.
        self.flags['lidar_pos_2'] : bool
            Sets the key 'lidar_pos_1' in the flag dictionary to True.
        self.long_zone : str
            A string representing longitudinal zone of the UTM grid zone.
        self.lat_zone : str
            A character representing latitudinal zone of the UTM grid zone.
        self.epsg_code : str
            A string representing EPSG code.
        self.flags['utm'] : bool
            Sets the key 'utm' in the flag dictionary to True.

        Notes
        --------
        Lidar positions can be added one at time.

        Examples
        --------
        >>> layout = CPT()
        >>> layout.set_utm_zone('31V')
        Correct latitudinal zone!
        Correct longitudinal zone!
        UTM zone set
        >>> layout.add_lidars(lidar_pos_1 = np.array([1,20,200]))
        Lidar 1 position added!

        >>> layout.add_lidars(lidar_pos_1 = np.array([1,20,200]), lidar_pos_2 = np.array([-20,1, 200]))
        Lidar 1 position added!
        Lidar 2 position added!        
        """

        if 'utm_zone' in kwargs:
            self.set_utm_zone(kwargs['utm_zone'])

        if 'lidar_pos_1' in kwargs or 'lidar_pos_2' in kwargs:
            if self.flags['utm'] == False:
                print('Cannot add lidar positions without specificing UTM zone!')
            else:        
                if 'lidar_pos_1' in kwargs and self.check_lidar_position(kwargs['lidar_pos_1']):
                    self.lidar_pos_1 = kwargs['lidar_pos_1']
                    self.flags['lidar_pos_1'] = True
                    print('Lidar 1 position added!')
                if 'lidar_pos_2' in kwargs and self.check_lidar_position(kwargs['lidar_pos_2']):
                    self.lidar_pos_2 = kwargs['lidar_pos_2']
                    self.flags['lidar_pos_2'] = True
                    print('Lidar 2 position added!')
        else:
            print('Lidar position(s) not specified!')        

    def generate_mesh(self, **kwargs):
        """
        Generates a rectangular horizontal mesh containing equally spaced points.
        
        Parameters
        ----------
            **kwargs : see below

        Keyword Arguments
        -----------------
        mesh_center : ndarray, optional
            3D array containing data with `float` or `int` type
            corresponding to Easting, Northing and Height coordinates of the mesh center.
            3D array data are expressed in meters.
        mesh_extent : int, optional
            mesh extent in Easting and Northing in meters.
        
        Returns
        -------
        self.mesh_corners : ndarray
            ndarray containing lower left and upper right corner of the mesh
        self.mesh : ndarray
            ndarray of mesh points

        Notes
        --------
        In case mesh center is not provided, but initial measurement points are
        this method will find the barycenter of measurement points and consider it
        as the mesh center. If mesh extent is not provided, a default value of 5000
        meters will be considered. 

        Examples
        --------
        >>> layout = CPT()
        >>> layout.generate_mesh(mesh_center = np.array([0,0,0]), mesh_extent = 1000)
        >>> layout.mesh
        array([[-1000, -1000,     0],
            [-1000,  -900,     0],
            [-1000,  -800,     0],
            ...,
            [ 1900,  1700,     0],
            [ 1900,  1800,     0],
            [ 1900,  1900,     0]])
        >>> layout.mesh_corners
        array([[-1000, -1000],
            [ 1000,  1000]])            
        """

        if 'mesh_extent' in kwargs:
            self.MESH_EXTENT = kwargs['mesh_extent']
        if 'mesh_center' in kwargs:
            self.mesh_center = kwargs['mesh_center']
            self.flags['mesh_center_added'] = True
        elif self.flags['measurements_added']:
            self.mesh_center = np.int_(np.mean(self.measurements_initial,axis = 0))
            self.flags['mesh_center_added'] = True
        else:
            print('Mesh center missing!')
            print('Mesh cannot be generated!')
            self.flags['mesh_center_added'] = False

        if self.flags['mesh_center_added']:
            # securing that the input parameters are int 
            self.mesh_center = np.int_(self.mesh_center)
            self.MESH_EXTENT = int(int(self.MESH_EXTENT / self.MESH_RES) * self.MESH_RES)
            self.mesh_corners = np.array([self.mesh_center[:2] - self.MESH_EXTENT, 
                                            self.mesh_center[:2] + self.MESH_EXTENT])

            self.x, self.y = np.meshgrid(
                    np.arange(self.mesh_corners[0][0], self.mesh_corners[1][0] + self.MESH_EXTENT, self.MESH_RES),
                    np.arange(self.mesh_corners[0][1], self.mesh_corners[1][1] + self.MESH_EXTENT, self.MESH_RES)
                            )
            
            self.z = np.full(self.x.shape, self.mesh_center[2])		
            self.mesh = np.array([self.x, self.y, self.z]).T.reshape(-1, 3)

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
        Checks whether UTM grid zone is valid or not.

        Parameters
        ----------
        utm_zone : str
            A string containing UTM grid zone with grid code.
        
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
        lat_zones = ['C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X']
        try:

            lat_zone = utm_zone[-1].upper() # in case users put lower case 
            long_zone = int(utm_zone[:-1])
            if lat_zone in lat_zones:
                print('Correct latitudinal zone!')
                flag = True
            else:
                print('Incorrect latitudinal zone!\nEnter a correct latitudinal zone!')
                flag = False
            
            if long_zone >= 1 and long_zone <= 60:
                print('Correct longitudinal zone!')
                flag = True and flag
            else:
                print('Incorrect longitudinal zone!\nEnter a correct longitudinal zone!')
                flag = False
        except:
            flag = False
            print('Wrong input!\nHint: there should not be spaces between longitudinal and latitudinal zones when expressing the UTM zone!')
        return flag

    @staticmethod
    def which_hemisphere(utm_zone):
        """
        Returns whether UTM grid zone belongs to the Northern or Southern hemisphere. 

        Parameters
        ----------
        utm_zone : str
            A string representing an UTM grid zone, containing digits (from 1 to 60) 
            indicating the longitudinal zone followed by a character ('C' to 'X' excluding 'O') 
            corresponding to the latitudinal zone.
        
        Returns
        -------
        out : str
            A string indicating North or South hemisphere.
        
        Examples
        --------
        If UTM grid zone exists:
        >>> utm2epsg('31V') 
        'North'

        >>> utm2epsg('31C') 
        'South'        

        If UTM grid zone doesn't exist:
        >>> utm2epsg('61Z')
        'Mars?'
        """
 
        lat_zones = ['C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X']
        lat_zone = utm_zone[-1].upper() # in case users put lower case 
        if int(utm_zone[:-1]) >= 1 and int(utm_zone[:-1]) <= 60:
            if lat_zone in lat_zones[10:]:
                return 'North'
            elif lat_zone in lat_zones[:10]:
                return 'South'
            else:
                return 'Mars?'
        else:
            return 'Mars?'

    @staticmethod        
    def utm2epsg(utm_zone):
        """
        Converts UTM grid zone to EPSG code.

        Parameters
        ----------
        utm_zone : str
            A string representing an UTM grid zone, containing digits (from 1 to 60) 
            indicating the longitudinal zone followed by a character ('C' to 'X' excluding 'O') 
            corresponding to the latitudinal zone.        
        Returns
        -------
        out : str
            A string containing EPSG code.
        
        Examples
        --------
        If UTM grid zone exists:
        >>> utm2epsg('31V') 
        '32631'

        If UTM grid zone doesn't exist:
        >>> utm2epsg('61Z')
        None
        """
        lat_zones = ['C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X']
        lat_zone = utm_zone[-1].upper() # in case users put lower case 
        if int(utm_zone[:-1]) >= 1 and int(utm_zone[:-1]) <= 60:
            if lat_zone in lat_zones[10:]:
                return '326' + utm_zone[:-1]
            elif lat_zone in lat_zones[:10]:
                return '327' + utm_zone[:-1]
            else:
                return 'Wrong latitudinal zone'
        else:
            return 'Wrong longitudinal zone'

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