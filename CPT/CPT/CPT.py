import numpy as np
from itertools import combinations, product
from pyproj import Proj

from osgeo import gdal, osr, ogr, gdal_array
import rasterio
import srtm

import pandas as pd
import geopandas
from shapely.geometry import Point
import whitebox

import matplotlib.pyplot as plt
import os, shutil


def del_folder_content(folder, exclude_file_extensions = None):
    """
    Deletes all files in a folder except specific file extensions.
    
    Parameters
    ----------
    folder : str
        A path to the folder which files will be deleted.
    exclude_file_extensions : array
        A array containing strings representing file extensions
        which will not be deleted.
    """
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        _, file_extension = os.path.splitext(file)
        try:
            if exclude_file_extensions is not None:
                if os.path.isfile(file_path) and file_extension not in exclude_file_extensions:
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)                    
            else:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def array_difference(A,B):
    """
    Finding which elements in array A are not present in array B. 

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


def azimuth2vector(azimuth):
    '''
    Converts azimuth angle to Cartesian angle.
    
    Parameters
    ----------
    azimuth : int, float, or ndarray
        Azimuth angle(s) given in degrees from North.
        
    Returns
    ------
    vector_angle : int, float or ndarray 
        Corresponding Cartesian angle(s) in degrees.
    
    '''
    y = np.cos(azimuth * (np.pi / 180))
    x = np.sin(azimuth * (np.pi / 180))
    vector_angle = np.arctan2(y, x) * (180 / np.pi)
    
    return vector_angle


def between_beams_angle(azimuth_1, azimuth_2):
    '''
    Find an intersecting angle between two laser beams which
    beam direction is described by azimuth angles.
    
    Parameters
    ----------
    azimuth_1 : int, float, or ndarray
        Azimuth angle(s) given in degrees from North for the first laser beam.

    azimuth_1 : int, float, or ndarray
        Azimuth angle(s) given in degrees from North for the second laser bea,.


    Returns
    ------
    bba : int, float or ndarray 
        Corresponding between beam angle in degrees.
    
    '''
    bba = abs(azimuth2vector(azimuth_1) - azimuth2vector(azimuth_2)) % 180
    
    return bba


class CPT():
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
    MAX_NO_OF_RANGES : int
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
    FILE_EXTENSIONS = np.array(['.tif', '.tiff', '.pdf', '.kml', '.png'])

    MESH_RES = 100 # in m
    MESH_EXTENT = 5000 # in m
    REP_RADIUS = 500 # in m
    POINTS_TYPE = np.array(['initial', 'optimized', 'reachable', 'identified', 'misc'])
    
    ACCUMULATION_TIME = 1000 # in ms
    AVERAGE_RANGE = 3000 # in m
    MAX_ACCELERATION = 100 # deg / s^2
    MAX_ELEVATION_ANGLE = 5 # in deg
    MAX_NO_OF_RANGES = 100 # maximum number of range gates
    MIN_INTERSECTING_ANGLE = 30 # in deg
    PULSE_LENGTH = 400 # in ns
    FFT_SIZE = 128 # no points

    MY_DPI = 96
    FONT_SIZE = 12

    def __init__(self):
        # measurement positions / mesh / beam coords
        self.long_zone = None
        self.lat_zone = None
        self.epsg_code = None 
        self.hemisphere = None 
        self.measurements_initial = None
        self.measurements_optimized = None
        self.measurements_identified = None
        self.measurements_reachable = None
        self.measurements_misc = None
        self.measurements_selector = 'initial'
        self.beam_coords = None
        self.mesh_center = None 
        
        # lidar positions
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
        self.exclusion_layer = None        
        self.elevation_angle_layer = None
        self.los_blck_layer = None
        self.range_layer = None
        self.combined_layer = None
        self.intersecting_angle_layer = None
        self.second_lidar_layer = None
        self.aerial_layer = None

        # Flags as you code
        self.flags = {
                      'measurements_added' : False,
                      'measurements_optimized': False,
                      'lidar_pos_1' : False,
                      'lidar_pos_2' : False,
                      'mesh_center_added' : False, 
                      'mesh_generated' : False,
                      'utm_set' : False, 
                      'input_check_pass' : False,
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
                      'combined_layer_generated' : False
                     }
  
        CPT.NO_LAYOUTS += 1

    def plot_layer(self, layer, **kwargs):
        """
        Plots individual GIS layers.
        
        Parameters
        ----------
        layer : ndarray
            nD array containing data with `float` or `int` type 
            corresponding to a specific GIS layer.
        **kwargs : see below

        Keyword Arguments
        -----------------
        title : str
            The plot title.
        legend_label : str
            The legend label indicating what parameter is plotted.
        levels : ndarray
            Predetermined levels for the plotted parameter.
        save_plot : bool
            Indicating whether to save the plot as PDF.
        input_type : str

        
        Returns
        -------
        plot : matplotlib
        
        Examples
        --------
        >>> layout.plot_GIS_layer(layout.orography_layer, levels = np.array(range(0,510,10)), title = 'Orography', legend_label = 'Height asl [m]' , save_plot = True)

        """
        if 'levels' in kwargs:
            levels = kwargs['levels']
        else:
            levels = np.linspace(np.min(layer), np.max(layer), 20)

        if len(layer.shape) > 2:
            levels = np.array(range(-1,layer.shape[-1] + 1, 1))
            layer = np.sum(layer, axis = 2)

        fig, ax = plt.subplots(sharey = True, figsize=(600/self.MY_DPI, 600/self.MY_DPI), dpi=self.MY_DPI)
        cmap = plt.cm.RdBu
        cs = plt.contourf(self.x, self.y, layer, levels=levels, cmap=cmap, alpha = 0.75)


        cbar = plt.colorbar(cs,orientation='vertical',fraction=0.047, pad=0.01)
        if 'legend_label' in kwargs:
            cbar.set_label(kwargs['legend_label'], fontsize = self.FONT_SIZE)
        
        if self.lidar_pos_1 is not None:
            ax.scatter(self.lidar_pos_1[0], self.lidar_pos_1[1], marker='o', 
            facecolors='black', edgecolors='white', s=30, zorder=2000, label = "lidar_1")
        if self.lidar_pos_2 is not None:
            ax.scatter(self.lidar_pos_2[0], self.lidar_pos_2[1], marker = 'o', 
            facecolors='white', edgecolors='black',s=30,zorder=2000, label = "lidar_2")

        if 'points_type' in kwargs and kwargs['points_type'] in self.POINTS_TYPE:
            measurement_pts = self.measurement_type_selector(kwargs['points_type'])
        else:
            measurement_pts = self.measurement_type_selector(self.measurements_selector)        

        if measurement_pts is not None:
            for i, pts in enumerate(measurement_pts):
                if i == 0:
                    ax.scatter(pts[0], pts[1], marker='o', 
                    facecolors='red', edgecolors='black', 
                    s=30,zorder=1500, label = 'measurements_' + self.measurements_selector)                    
                else:
                    ax.scatter(pts[0], pts[1], marker='o',
                    facecolors='red', edgecolors='black', 
                    s=30,zorder=1500)

        if self.lidar_pos_1 is not None or self.lidar_pos_2 is not None or measurement_pts is not None:
            ax.legend(loc='lower right', fontsize = self.FONT_SIZE)    

        plt.xlabel('Easting [m]', fontsize = self.FONT_SIZE)
        plt.ylabel('Northing [m]', fontsize = self.FONT_SIZE)

        if 'title' in kwargs:
            plt.title(kwargs['title'], fontsize = self.FONT_SIZE)

        ax.set_aspect(1.0)
        plt.show()

        if 'title' in kwargs and 'save_plot' in kwargs and kwargs['save_plot']:
                fig.savefig(self.OUTPUT_DATA_PATH + kwargs['title'] + '.pdf', bbox_inches='tight')


    def plot_optimization(self, **kwargs):
        """
        Plots measurement point optimization result.
        
        Parameters
        ----------
        **kwargs : see below

        Keyword Arguments
        -----------------
        save_plot : bool
            Indicating whether to save the plot as PDF.

        See also
        --------
        optimize_measurements : implementation of disc covering problem
        add_measurements : method for adding initial measurement points

        Notes
        -----
        To generate the plot it is required that 
        the measurement optimization was performed.

        Returns
        -------
        plot : matplotlib
        
        """
        if 'points_type' in kwargs and kwargs['points_type'] in self.POINTS_TYPE:
            measurement_pts = self.measurement_type_selector(kwargs['points_type'])
            self.measurements_selector = kwargs['points_type']
        else:
            measurement_pts = self.measurement_type_selector(self.measurements_selector)  


        if measurement_pts is not None and self.measurements_optimized is not None:
            fig, ax = plt.subplots(sharey = True, figsize=(600/self.MY_DPI, 600/self.MY_DPI), dpi=self.MY_DPI)

            for i,pt in enumerate(measurement_pts):
                if i == 0:
                    ax.scatter(pt[0], pt[1],marker='o', 
                        facecolors='red', edgecolors='black', 
                        s=30,zorder=1500, label = "original")
                else:
                    ax.scatter(pt[0], pt[1],marker='o', 
                                        facecolors='red', edgecolors='black', 
                                        s=30,zorder=1500,)            


            for i,pt in enumerate(self.measurements_optimized):
                if i == 0:
                    ax.scatter(pt[0], pt[1],marker='o', 
                        facecolors='white', edgecolors='black', 
                        s=30,zorder=1500, label = "optimized")
                    ax.add_artist(plt.Circle((pt[0], pt[1]), 
                                            self.REP_RADIUS,                               
                                            facecolor='grey', edgecolor='black', 
                                            zorder=500,  alpha = 0.5))                 
                else:
                    ax.scatter(pt[0], pt[1],marker='o', 
                        facecolors='white', edgecolors='black', 
                        s=30,zorder=1500)
                    ax.add_artist(plt.Circle((pt[0], pt[1]), 
                                            self.REP_RADIUS,                               
                                            facecolor='grey', edgecolor='black', 
                                            zorder=500,  alpha = 0.5))                 
    
                    

            plt.xlabel('Easting [m]', fontsize = self.FONT_SIZE)
            plt.ylabel('Northing [m]', fontsize = self.FONT_SIZE)
            ax.legend(loc='lower right', fontsize = self.FONT_SIZE)


            ax.set_xlim(np.min(self.x),np.max(self.x))
            ax.set_ylim(np.min(self.y),np.max(self.y))

            ax.set_aspect(1.0)
            plt.show()
            if 'save_plot' in kwargs and kwargs['save_plot']:
                fig.savefig(self.OUTPUT_DATA_PATH + 'measurements_optimized.pdf', bbox_inches='tight')




    def set_utm_zone(self, utm_zone):
        """
        Sets latitudinal and longitudinal zones and EPSG code to the CPT instance. 
        
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
        self.hemisphere : str
            A string indicating north or south hemisphere.            
        self.flags['utm_set'] : bool
            Sets the key 'utm' in the flag dictionary to True.                
        """
        if self.check_utm_zone(utm_zone):
            self.long_zone = utm_zone[:-1]
            self.lat_zone = utm_zone[-1].upper() 
            self.epsg_code = self.utm2epsg(utm_zone)
            self.hemisphere = self.which_hemisphere(utm_zone) 
            self.flags['utm_set'] = True
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
        points_type : str
            A string indicating to what variable 
            measurement points should be added.
            A default value is set to 'initial'.
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
        self.flags['measurements_added'] : bool
            Sets the key 'measurements_added' in the flag dictionary to True.
        self.long_zone : str
            A string representing longitudinal zone of the UTM grid zone.
        self.lat_zone : str
            A character representing latitudinal zone of the UTM grid zone.
        self.epsg_code : str
            A string representing EPSG code.
        self.hemisphere : str
            A string indicating north or south hemisphere.            
        self.flags['utm_set'] : bool
            Sets the key 'utm' in the flag dictionary to True.
        """
        if 'utm_zone' in kwargs:
            self.set_utm_zone(kwargs['utm_zone'])

        if self.flags['utm_set'] == False:
            print('Cannot add measurement points without specificing UTM zone first!')

        if self.flags['utm_set'] and self.check_measurement_positions(kwargs['measurements']):
            if 'points_type' in kwargs and kwargs['points_type'] in self.POINTS_TYPE:
                self.measurements_selector = kwargs['points_type']
            print('Adding ' + self.measurements_selector + ' measurement points!')

            if len(kwargs['measurements'].shape) == 2:
                if self.measurements_selector == 'initial':
                    self.measurements_initial = np.unique(kwargs['measurements'], axis=0)
                elif self.measurements_selector == 'optimized':
                    self.measurements_optimized = np.unique(kwargs['measurements'], axis=0)
                elif self.measurements_selector == 'reachable':
                    self.measurements_reachable = np.unique(kwargs['measurements'], axis=0)
                elif self.measurements_selector == 'identified':
                    self.measurements_reachable = np.unique(kwargs['measurements'], axis=0)
                else:
                    self.measurements_misc = np.unique(kwargs['measurements'], axis=0)
            else:
                if self.measurements_selector == 'initial':
                    self.measurements_initial = np.unique(np.array([kwargs['measurements']]), axis=0)
                elif self.measurements_selector == 'optimized':
                    self.measurements_optimized = np.unique(np.array([kwargs['measurements']]), axis=0)
                elif self.measurements_selector == 'reachable':
                    self.measurements_reachable = np.unique(np.array([kwargs['measurements']]), axis=0)
                elif self.measurements_selector == 'identified':
                    self.measurements_reachable = np.unique(np.array([kwargs['measurements']]), axis=0)
                else:
                    self.measurements_misc = np.unique(np.array([kwargs['measurements']]), axis=0)

            self.flags['measurements_added'] = True

    def generate_disc_matrix(self, **kwargs):
        """
        Generates mid points between any combination of two measurement points 
        which act as disc centers. The mid points are tested which measurement
        points they are covering producing so-called disc-covering matrix used
        in the measurement point optimization method.

        Parameters
        ----------
        self.measurements_initial : ndarray
            An initial set of measurement points provided as nD array.
        self.REP_RADIUS : int
            MEASNET's representativness radius of measurements.
            The radius is expressed in meters.
            A default value is set to 500 m.

        Keyword arguments
        -----------------
        points_type : str
            ...

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
        if 'points_type' in kwargs and kwargs['points_type'] in self.POINTS_TYPE:
            measurement_pts = self.measurement_type_selector(kwargs['points_type'])
            self.measurements_selector = kwargs['points_type']
        else:
            measurement_pts = self.measurement_type_selector(self.measurements_selector)        

        if measurement_pts is not None:
            points_combination = np.asarray(list(combinations(list(measurement_pts[:,(0,1)]), 2)))    
            discs = (points_combination[:,0] + points_combination[:,1]) / 2

            temp = np.asarray(list(product(list(discs), list(measurement_pts[:,(0,1)]))))
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
            return print("No measurement points -> nothing to optimize!")

    def optimize_measurements(self, **kwargs):
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
        if 'points_type' in kwargs and kwargs['points_type'] in self.POINTS_TYPE:
            measurement_pts = self.measurement_type_selector(kwargs['points_type'])
            self.measurements_selector = kwargs['points_type']
        else:
            measurement_pts = self.measurement_type_selector(self.measurements_selector)        

        if measurement_pts is not None:
            print('Optimizing ' + self.measurements_selector + ' measurement points!')
            discs, matrix = self.generate_disc_matrix()
            points_uncovered = measurement_pts
            points_covered_total = np.zeros((0,3), measurement_pts.dtype)
            discs_selected = np.zeros((0,3))
            i = 0
            j = len(points_uncovered)

            while i <= (len(discs) - 1) and j > 0 :
                indexes = np.where(matrix[i] == 1 )
                points_covered = measurement_pts[indexes]
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
            if len(self.measurements_optimized) == len(measurement_pts):
                self.measurements_optimized = measurement_pts
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
        self.hemisphere : str
            A string indicating north or south hemisphere.            
        self.flags['utm_set'] : bool
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
            if self.flags['utm_set'] == False:
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
        points_type : str, optional

        
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
        if 'points_type' in kwargs and kwargs['points_type'] in self.POINTS_TYPE:
            measurement_pts = self.measurement_type_selector(kwargs['points_type'])
            self.measurements_selector = kwargs['points_type']
        else:
            measurement_pts = self.measurement_type_selector(self.measurements_selector)

        if 'mesh_extent' in kwargs:
            self.MESH_EXTENT = kwargs['mesh_extent']
        if 'mesh_center' in kwargs:
            self.mesh_center = kwargs['mesh_center']
            self.flags['mesh_center_added'] = True
        elif self.flags['measurements_added']:
            self.mesh_center = np.int_(np.mean(measurement_pts,axis = 0))
            self.flags['mesh_center_added'] = True
        else:
            print('Mesh center missing!')
            print('Mesh cannot be generated!')
            self.flags['mesh_center_added'] = False

        if self.flags['mesh_center_added']:
            # securing that the input parameters are int 
            self.mesh_center = np.int_(self.mesh_center)
            self.MESH_EXTENT = int(int(self.MESH_EXTENT / self.MESH_RES) * self.MESH_RES)
            # self.mesh_corners_utm = np.array([self.mesh_center[:2] - self.MESH_EXTENT, 
            #                                 self.mesh_center[:2] + self.MESH_EXTENT])
            self.mesh_corners_utm = np.array([self.mesh_center - self.MESH_EXTENT, 
                                            self.mesh_center + self.MESH_EXTENT])
            self.mesh_corners_geo = self.utm2geo(self.mesh_corners_utm, self.long_zone, self.hemisphere)                                 

            self.x, self.y = np.meshgrid(
                    np.arange(self.mesh_corners_utm[0][0], self.mesh_corners_utm[1][0] + self.MESH_RES, self.MESH_RES),
                    np.arange(self.mesh_corners_utm[0][1], self.mesh_corners_utm[1][1] + self.MESH_RES, self.MESH_RES)
                            )
            
            self.z = np.full(self.x.shape, self.mesh_center[2])		
            self.mesh_utm = np.array([self.x, self.y, self.z]).T.reshape(-1, 3)
            self.mesh_geo = self.utm2geo(self.mesh_utm, self.long_zone, self.hemisphere)            
            self.flags['mesh_generated'] = True

    def generate_intersecting_angle_layer(self):
        if self.flags['lidar_pos_1'] :
            measurement_pts = self.measurement_type_selector(self.measurements_selector)
            nrows, ncols = self.x.shape
            no_pts = len(measurement_pts)

            azimuths_1 = (self.generate_beam_coords(measurement_pts, self.lidar_pos_1)[:,0] - 180) % 360
            azimuths_2 = self.azimuth_angle_array

            self.intersecting_angle_layer = np.empty((nrows, ncols, no_pts), dtype=float)

            for i in range(0,no_pts):
                azimuth_1 =  azimuths_1[i]
                azimuth_2 =  azimuths_2[:,:, i]
                tmp =  between_beams_angle(azimuth_1, azimuth_2)
                tmp[np.where(tmp >= 90)] = 180 - tmp[np.where(tmp >= 90)]
                tmp[np.where(tmp < 30)] = 0
                tmp[np.where(tmp >= 30)] = 1
                self.intersecting_angle_layer[:,:,i] = tmp            
        else:
            print('Lidar 1 position not set!')

    def generate_combined_layer(self, **kwargs):
        """
        Generates the combined layer which is used
        for the positioning of lidars.
        
        Notes
        --------
        Initial measurement positions must be added before
        calling this method. The method calls sequentially
        generation of mesh, topographic layer,
        beam steering coordinates, range restriction layer, 
        elevation restriction layer and los blockage layer.

        See also
        --------
        add_measurements() : adding measurement points to the CPT class instance 
        """

        if 'points_type' in kwargs and kwargs['points_type'] in self.POINTS_TYPE:
            self.measurements_selector = kwargs['points_type']
        else:
            self.measurements_selector = 'initial'

        if self.measurement_type_selector(self.measurements_selector) is not None:
            print('Generating combined layer for ' + self.measurements_selector + ' measurement points!')
            self.generate_mesh()
            self.generate_topographic_layer()
            self.generate_beam_coords_mesh()
            self.generate_range_restriction_layer()
            self.generate_elevation_restriction_layer()
            self.generate_los_blck_layer()

            nrows, ncols = self.x.shape
            self.combined_layer = self.elevation_angle_layer * self.range_layer * self.los_blck_layer
            self.combined_layer = self.combined_layer * self.restriction_zones_layer.reshape((nrows,ncols,1))
            self.flags['combined_layer_generated'] = True
        else:
            print('Variable self.measurements_'+ self.measurements_selector + ' is empty!')
            print('Combined layer was not generated!')

    def generate_los_blck_layer(self, **kwargs):
        """
        Generates the los blockage layer by performing 
        viewshed analysis for the selected site.
        
        Notes
        --------
        Initial measurement positions must be added, output data
        folder set and mesh and topography layer generated before 
        calling this method.
        
        The method makes sequential calls to methods that exports
        measurement points and topography layer as shape files. 
        This files are temporary and are removed once the viewshed
        analysis and processing are executed.

        See also
        --------
        add_measurements() : adding measurement points to the CPT class instance 
        """
        if 'points_type' in kwargs and kwargs['points_type'] in self.POINTS_TYPE:
            self.measurements_selector = kwargs['points_type']

        if self.measurement_type_selector(self.measurements_selector) is not None:        
            self.export_measurements()
            self.export_topography()
            self.viewshed_analysis()
            self.viewshed_processing()
            self.flags['los_blck_layer_generated'] = True
            del_folder_content(self.OUTPUT_DATA_PATH, self.FILE_EXTENSIONS)
        else:
            print('Variable self.measurements_'+ self.measurements_selector + ' is empty!')
            print('LOS blockage layer was not generated!')

    def viewshed_processing(self):
        """
        Performs the viewshed data processing for the given site.
        
        Notes
        --------
        The method which performs viewshed analysis for the given 
        site must be called first before calling this method.
        
        The method loads the shapefiles corresponding to the viewshed
        analysis of the site for each individual measurement point.
        The loaded data are converted into the los blockage GIS layer.

        See also
        --------
        viewshed_analysis() : viewshed analysis of the site
        """        

        if self.flags['viewshed_analyzed']:
            measurement_pts = self.measurement_type_selector(self.measurements_selector)
            nrows, ncols = self.x.shape
            no_pts = len(measurement_pts)

            self.los_blck_layer = np.empty((nrows, ncols, no_pts), dtype=float)


            for i in range(0,len(measurement_pts)):
                los_blck_tmp  = np.loadtxt(self.OUTPUT_DATA_PATH + "los_blockage_" +str(i+1)+".asc", skiprows=6)
                los_blck_tmp  = np.flip(los_blck_tmp, axis = 0)
                self.los_blck_layer[:,:,i] = los_blck_tmp
            
            self.flags['viewshed_performed'] = True

    def viewshed_analysis(self):
        """
        Performs the viewshed analysis for the given site.
        
        Notes
        --------
        The shapefiles corresponding to the topography layer and
        measurement points must be generated before calling this method.
        
        The method loads the shapefiles and calls a method of the 
        whitebox library[1] which performs the viewshed analysis.

        References
        ----------
        .. [1] Missing reference ...        

        See also
        --------
        export_topography() : topography layer shapefile exporter
        export_measurements() : measurement points shapefile exporter
        """

        if os.path.exists(self.OUTPUT_DATA_PATH + 'topography.asc') and self.flags['topography_exported'] and self.flags['measurements_exported'] and self.flags['measurements_added'] and self.measurement_type_selector(self.measurements_selector) is not None:
            measurement_pts = self.measurement_type_selector(self.measurements_selector)

            terrain_height = self.get_elevation(self.long_zone + self.lat_zone, measurement_pts)
            measurement_height = measurement_pts[:,2]
            height_diff = measurement_height - terrain_height

            for i in range(0,len(measurement_pts)):
                wbt = whitebox.WhiteboxTools()
                wbt.set_working_dir(self.OUTPUT_DATA_PATH)
                wbt.verbose = False
                wbt.viewshed('topography.asc',"measurement_pt_" +str(i+1)+".shp","los_blockage_" +str(i+1)+".asc",height_diff[i])
            self.flags['viewshed_analyzed'] = True

    def export_topography(self):
        """
        Exports the topography layer a ASCI shapefile.
        
        Notes
        --------
        The topography layer must be generated and output data
        folder set before calling this method.
        
        The method writes out a ASCI shape file. The shapefile is 
        used in the site viewshed analysis.

        See also
        --------
        add_measurements() : adding measurement points to the CPT class instance 
        viewshed_analysis() : the site viewshed analysis
        """        
        if os.path.exists(self.OUTPUT_DATA_PATH) and self.flags['topography_layer_generated']:
            topography_array = np.flip(self.topography_layer,axis=0)
            f = open(self.OUTPUT_DATA_PATH + 'topography.asc', 'w')
            f.write("ncols " + str(topography_array.shape[0]) + "\n")
            f.write("nrows " + str(topography_array.shape[1]) + "\n")
            f.write("xllcorner " + str(self.mesh_corners_utm[0][0]) + "\n")
            f.write("yllcorner " + str(self.mesh_corners_utm[0][1]) + "\n")
            f.write("cellsize " + str(self.MESH_RES) + "\n")
            f.write("NODATA_value " + str(self.NO_DATA_VALUE) + "\n")
            np.savetxt(f, topography_array, fmt='%.1f')
            f.close()
            self.flags['topography_exported'] = True
        else:
            print('The output data path does not exist!')

    def export_measurements(self):
        """
        Exports the measurement points as ESRI shapefile.
        
        Notes
        --------
        Initial measurement positions must be added and output data
        folder set before calling this method.
        
        The method creates Geopanda dataframe which is then exported
        as a ESRI shapefile. The shapefile is used in the site viewshed analysis.

        See also
        --------
        add_measurements() : adding measurement points to the CPT class instance 
        viewshed_analysis() : the site viewshed analysis
        """

        if os.path.exists(self.OUTPUT_DATA_PATH) and self.flags['measurements_added']: 
            pts = self.measurement_type_selector(self.measurements_selector)

            pts_dict=[]
            for i,pt in enumerate(pts)  :
                pts_dict.append({'Name': "MP_" + str(i), 'E': pt[0], 'N': pt[1]})
                pts_df = pd.DataFrame(pts_dict)
                pts_df['geometry'] = pts_df.apply(lambda x: Point((float(x.E), float(x.N))), axis=1)
                pts_df = geopandas.GeoDataFrame(pts_df, geometry='geometry')
                pts_df.crs= "+init=epsg:" + self.epsg_code
                pts_df.to_file(self.OUTPUT_DATA_PATH + 'measurement_pt_' + str(i + 1) + '.shp', driver='ESRI Shapefile')

                pts_dict=[]
            self.flags['measurements_exported'] = True


    def generate_range_restriction_layer(self):
        """
        Generates range restricted GIS layer.

        Notes
        -----
        The beams coordinates for every mesh point must be
        generated (self.generate_beam_coords_mesh()) before
        calling this method.
        """        
        if self.flags['beam_coords_generated'] == True:
            self.elevation_angle_layer = np.copy(self.elevation_angle_array)
            self.elevation_angle_layer[np.where((self.elevation_angle_layer <= self.MAX_ELEVATION_ANGLE))] = 1
            self.elevation_angle_layer[np.where((self.elevation_angle_layer > self.MAX_ELEVATION_ANGLE))] = 0
        else:
            print('No beams coordinated generated, run self.gerate_beam_coords_mesh(str) first!')    

    def generate_elevation_restriction_layer(self):
        """
        Generates elevation restricted GIS layer.

        Notes
        -----
        The beams coordinates for every mesh point must be
        generated (self.generate_beam_coords_mesh()) before
        calling this method.
        """
        if self.flags['beam_coords_generated'] == True:
            self.range_layer = np.copy(self.range_array)
            self.range_layer[np.where((self.range_layer <= self.AVERAGE_RANGE))] = 1
            self.range_layer[np.where((self.range_layer > self.AVERAGE_RANGE))] = 0          
        else:
            print('No beams coordinated generated!\n Run self.gerate_beam_coords_mesh() first!')

    def generate_beam_coords_mesh(self, **kwargs):
        """
        Generates beam steering coordinates from every mesh point 
        to every measurement point.

        Parameters
        ----------
        **kwargs : see below

        Keyword Arguments
        -----------------
        points_type : str
            A string indicating which measurement points to be
            used for the beam steering coordinates calculation.
            A default value is set to 'initial'.

        Notes
        --------
        The measurement points must exists and mesh generated 
        before calling this method.

        """
        # measurement point selector:
        
        if 'points_type' in kwargs and kwargs['points_type'] in self.POINTS_TYPE:
            measurement_pts = self.measurement_type_selector(kwargs['points_type'])
            self.measurements_selector = kwargs['points_type']
        else:
            measurement_pts = self.measurement_type_selector(self.measurements_selector)        

        if measurement_pts is not None:
            try:

                array_shape = (measurement_pts.shape[0], )  + self.mesh_utm.shape
                self.beam_coords = np.empty(array_shape, dtype=float)

                for i, pts in enumerate(measurement_pts):
                    self.beam_coords[i] = self.generate_beam_coords(self.mesh_utm, pts)
                self.flags['beam_coords_generated'] = True

                # splitting beam coords to three arrays 
                nrows, ncols = self.x.shape
                self.azimuth_angle_array = self.beam_coords[:,:,0].T.reshape(nrows,ncols,len(measurement_pts), order='F')                
                self.elevation_angle_array = self.beam_coords[:,:,1].T.reshape(nrows,ncols,len(measurement_pts), order='F')
                self.range_array = self.beam_coords[:,:,2].T.reshape(nrows,ncols,len(measurement_pts), order='F')                

            except:
                print('Something went wrong! Check measurement points')
        else:
            print('Variable self.measurements_'+ self.measurements_selector + ' is empty!')
            print('Beam steering coordinates not generated!')

    def measurement_type_selector(self, points_type):
        """
        Selects measurement type.

        Parameters
        ----------
        points_type : str
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

        if points_type == 'initial':
            return self.measurements_initial
        elif points_type == 'optimized':
            return self.measurements_optimized
        elif points_type == 'reachable':
            return self.measurements_reachable
        elif points_type == 'identified':
            return self.measurements_reachable
        elif points_type == 'misc':
            return self.measurements_misc            
        else:
            return None


    def store_points(self, points_type, points):
        """
        Store measurement points to the measurement point
        variable specified by the type.

        Parameters
        ----------
        points_type : str
            A string indicating measurement points type
        points : ndarray
            nD array containing measurement points

        Notes
        -----
        ...
        """        

        if points_type == 'initial':
            self.measurements_initial = points
        elif points_type == 'optimized':
            self.measurements_optimized = points
        elif points_type == 'reachable':
            self.measurements_reachable = points
        elif points_type == 'identified':
            self.measurements_reachable = points
        else:
            self.measurements_misc = points

    def generate_topographic_layer(self):
        """
        Generates topographic layer.

        Notes
        -----
        It is required that the landcover data are provided 
        and that computer running this code has an access to
        the Internet in order to obtain SRTM DEM data.

        The method itself sequentially calls generation of the
        landcover and orography layers, which is followed with 
        the summation of the orography and canopy heights for 
        each individual mesh point.

        See also
        --------
        self.generate_orography_layer() : orography layer generation
        self.generate_landcover_layer() : landcover layer generation
        """        

        if self.flags['mesh_generated']:
            self.generate_orography_layer()
            self.generate_landcover_layer()
            if self.flags['orography_layer_generated'] == True:
                self.topography_layer = self.canopy_height_layer + self.orography_layer

                if self.flags['landcover_layers_generated'] == False:
                    print('Topography layer only generated using orography height since canopy height is not provided!')
                else:
                    print('Topography layer generated using orography and canopy height.')
                self.flags['topography_layer_generated'] = True
            else:
                print('Cannot generate topography layer following layers are missing:')
                if self.flags['landcover_layers_generated'] == False:
                    print('Canopy height')
                if self.flags['orography_layer_generated'] == False:
                    print('Orography height')
        else:
            print('Mesh not generated -> topographic layer cannot be generated ')

    def generate_orography_layer(self):
        """
        Generates orography layer.

        Notes
        -----
        It is required that the computer running this method 
        has an access to the Internet since the terrain height
        information are obtained from the SRTM DEM database.

        The mesh must be generated before running this method.

        The method builds on top of the existing SRTM library [1].

        References
        ----------
        .. [1] Missing reference ...

        See also
        --------
        self.generate_mesh() : mesh generation
        """        
                
        if self.flags['mesh_generated']:        
            nrows, ncols = self.x.shape
            elevation_data = srtm.get_data()

            self.mesh_utm[:,2] = np.asarray([elevation_data.get_elevation(x[0],x[1]) if elevation_data.get_elevation(x[0],x[1]) != None and elevation_data.get_elevation(x[0],x[1]) != np.nan else 0 for x in self.mesh_geo])

            self.mesh_geo[:,2] = self.mesh_utm[:,2]
            self.orography_layer = self.mesh_utm[:,2].reshape(nrows, ncols).T
            self.flags['orography_layer_generated'] = True
        else:
            print('Mesh not generated -> orography layer cannot be generated ')

    def generate_landcover_layer(self):
        """
        Generates restriction zones and canopy height 
        layers based on the CORINE landcover data.

        Notes
        --------
        It is necessary that the path to the landcover data 
        is set to the corresponding class attributed.

        Currently the method only works with the CORINE data [1].

        It is necessary that the mesh has been generated before 
        calling this method.

        The method crops the landcover data according to the mesh 
        corners, saves the cropped data, and based on the cropped
        data generates canopy height and restriction zone layers.

        References
        ----------
        .. [1] Missing reference ....

        See also
        --------
        self.generate_mesh() : mesh generation
        self.crop_landcover_data() : cropping landcover data
        self.import_landcover_data() : importing cropped landcover data
        self.generate_canopy_height() : canopy height generation
        self.generate_restriction_zones() : restriction zone generation
        """   
        if len(self.LANDCOVER_DATA_PATH) > 0:
            try:
                self.crop_landcover_data()
                self.import_landcover_data()
                self.generate_canopy_height()
                self.generate_restriction_zones()
                self.flags['landcover_layers_generated'] = True
            except:
                print('Seems that the path to the landcover data or landcover data is not valid!')
                self.flags['landcover_layers_generated'] = False
        else:
            print('Path to landcover data not provided!')
            self.flags['landcover_layers_generated'] = False

    def generate_restriction_zones(self):
        """
        Converts specific CORINE landcover CLC codes
        to the restriction zones layer.

        Notes
        --------
        It is necessary that the base landcover layer is generated.

        Currently the method only works with the CORINE data [1].

        The method converts specific CLC codes[2], corresponding to specific 
        landcover types such as for example water body, to the zones which
        are restricted for the lidar installation. 

        References
        ----------
        .. [1] Missing reference ....
        .. [2] Missing reference ....

        See also
        --------
        self.crop_landcover_data() : cropping landcover data
        self.import_landcover_data() : importing cropped landcover data
        """           
        if self.flags['landcover_layer_generated']:        
            self.restriction_zones_layer = np.copy(self.landcover_layer)
            self.restriction_zones_layer[np.where((self.restriction_zones_layer < 23))] = 1
            self.restriction_zones_layer[np.where((self.restriction_zones_layer > 25) &
                                                (self.restriction_zones_layer < 35))] = 1
            self.restriction_zones_layer[np.where((self.restriction_zones_layer > 44))] = 1

            self.restriction_zones_layer[np.where((self.restriction_zones_layer >= 23) & 
                                                (self.restriction_zones_layer <= 25))] = 0

            self.restriction_zones_layer[np.where((self.restriction_zones_layer >= 35) & 
                                                (self.restriction_zones_layer <= 44))] = 0
            self.flags['restriction_zones_generated'] = True
        else:
            print('No landcover layer generated -> exclusion zones layer not generated!')

    def generate_canopy_height(self):
        """
        Converts specific CORINE landcover CLC codes
        to the canopy height layer.

        Notes
        --------
        It is necessary that the base landcover layer is generated.

        Currently the method only works with the CORINE data [1].

        The method converts specific CLC codes[2], corresponding to forest, 
        to the canopy height. 
        
        It simply adds 20 m for CLC codes correspoding to forest.

        References
        ----------
        .. [1] Missing reference ....
        .. [2] Missing reference ....

        See also
        --------
        self.crop_landcover_data() : cropping landcover data
        self.import_landcover_data() : importing cropped landcover data
        """              

        if self.flags['landcover_layer_generated']:
            self.canopy_height_layer = np.copy(self.landcover_layer)
            self.canopy_height_layer[np.where(self.canopy_height_layer < 23)] = 0
            self.canopy_height_layer[np.where(self.canopy_height_layer == 23)] = 20
            self.canopy_height_layer[np.where(self.canopy_height_layer == 24)] = 20
            self.canopy_height_layer[np.where(self.canopy_height_layer == 25)] = 20
            self.canopy_height_layer[np.where(self.canopy_height_layer >  25)] = 0
            self.flags['canopy_height_generated'] = True
        else:
            print('No landcover layer generated -> canopy height layer not generated!')


    def import_landcover_data(self):
        """
        Generates landcover layer based on the CORINE landcover data.

        Notes
        --------
        It is necessary that the CORINE landcover data are cropped
        to the area correspoding to the previously generated mesh.

        Currently the method only works with the CORINE data [1].

        References
        ----------
        .. [1] Missing reference ....

        See also
        --------
        self.crop_landcover_data() : cropping landcover data
        self.generate_mesh() : mesh generation
        """         
        if self.flags['landcover_map_clipped']:
            nrows, ncols = self.x.shape
            with rasterio.open(self.OUTPUT_DATA_PATH + 'landcover_cropped_utm.tif') as src:
                land_cover_array = src.read()
                # header_information = src.profile
            land_cover_array = np.flip(land_cover_array.reshape(nrows, ncols),axis=0)
            # land_cover_array = np.flip(land_cover_array,axis=0)
            self.landcover_layer = land_cover_array
            self.flags['landcover_layer_generated'] = True
        else:
            print('Landcover map not clipped!')

    def crop_landcover_data(self):
        """
        Crops the CORINE landcover data to the mesh area.

        Notes
        --------
        It is necessary that the CORINE landcover data path is provided
        and that the mesh is generated before calling this method.

        Currently the method only works with the CORINE data [1].

        References
        ----------
        .. [1] Missing reference ....

        See also
        --------
        self.generate_mesh() : mesh generation
        """               
        if self.flags['mesh_generated']:  
            if len(self.LANDCOVER_DATA_PATH) > 0:
                if len(self.OUTPUT_DATA_PATH)> 0:               
                    input_image = gdal.Open(self.LANDCOVER_DATA_PATH, gdal.GA_ReadOnly)
                    # projection = input_image.GetProjectionRef()
                    # print(projection)

                    clipped_map = gdal.Warp(self.OUTPUT_DATA_PATH + 'landcover_cropped_utm.tif', 
                                input_image,format = 'GTiff',
                                outputBounds=[self.mesh_corners_utm[0,0], self.mesh_corners_utm[0,1],
                                            self.mesh_corners_utm[1,0], self.mesh_corners_utm[1,1]],
                                dstSRS='EPSG:'+self.epsg_code, 
                                width=int(1 + 2 * self.MESH_EXTENT / self.MESH_RES), 
                                height=int(1 + 2 * self.MESH_EXTENT / self.MESH_RES))
                    clipped_map = None # Close dataset
                    self.flags['landcover_map_clipped'] = True
                else:
                    print('No output data folder provided!')
            else:
                print('No landcover data provided!')
        else:
            print('Mesh not generated -> landcover map cannot be clipped!')            

    @classmethod
    def get_elevation(cls, utm_zone, pts_utm):
        """
        Fetch elevation from the SRTM database for 
        a number of points described by in the UTM coordinates.

        Parameters
        ----------
        utm_zone : str
            A string representing an UTM grid zone, containing digits (from 1 to 60) 
            indicating the longitudinal zone followed by a character ('C' to 'X' excluding 'O') corresponding to the latitudinal zone.
        pts_utm : ndarray
            nD array containing data with `float` or `int` type corresponding 
            to Easting and Northing coordinates of points.
            nD array data are expressed in meters.            

        Returns
        -------
        elevation : ndarray
            nD array containing elevations for each point in pts_utm array

        Notes
        --------
        It is required that the computer running this method 
        has an access to the Internet since the terrain height
        information are obtained from the SRTM DEM database [1, 2].

        References
        ----------
        .. [1] Missing reference ....

        See also
        --------
        self.which_hemisphere(utm_zone) : returns hemisphere
        self.utm2geo(pts_utm, long_zone, hemisphere) : converts utm to geo
        srtm : library for simple access to the SRTM DEM database
        """             
        if cls.check_utm_zone(utm_zone):
            hemisphere = cls.which_hemisphere(utm_zone)
            long_zone = utm_zone[:-1]
            pts_geo = cls.utm2geo(pts_utm, long_zone, hemisphere)
            elevation_data = srtm.get_data()

            # this works if multiple points are provided but fails if single one is given
            elevation = np.asarray([elevation_data.get_elevation(pt[0],pt[1]) if elevation_data.get_elevation(pt[0],pt[1]) != None and elevation_data.get_elevation(pt[0],pt[1]) != np.nan else 0 for pt in pts_geo])
            elevation[np.isnan(elevation)] = cls.NO_DATA_VALUE
            return elevation
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
            A string representing an UTM grid zone, containing digits (from 1 to 60) 
            indicating the longitudinal zone followed by a character ('C' to 'X' excluding 'O') corresponding to the latitudinal zone.
        
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
        """
 
        lat_zones = ['C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X']
        lat_zone = utm_zone[-1].upper() # in case users put lower case 
        if int(utm_zone[:-1]) >= 1 and int(utm_zone[:-1]) <= 60:
            if lat_zone in lat_zones[10:]:
                return 'north'
            elif lat_zone in lat_zones[:10]:
                return 'south'
            else:
                return None
        else:
            return None

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

    @staticmethod
    def utm2geo(points_utm, long_zone, hemisphere):
        """
        Converts an array of points in the UTM coord system to
        an array of point in the GEO coord system.
        
        Parameters
        ----------
        points_utm : ndarray
            nD array containing data with `float` or `int` type corresponding 
            to Northing, Easting and Height coordinates of points.
            nD array data are expressed in meters.
        long_zone : str
            A string representing longitudinal zone of the UTM grid zone.
        hemisphere : str
            A string indicating north or south hemisphere.            

        Returns
        -------
        points_geo : ndarray
            nD array containing data with `float` or `int` type corresponding 
            to latitude, longitude and height coordinates of points.
        
        Examples
        --------
        >>> points_utm = np.array([[317733, 6175124, 100], [316516, 6175827, 100], [316968, 6174561, 100]])
        >>> utm2geo(points_utm, '33', 'north')
        array([[ 55.68761863,  12.10043705, 100.        ],
            [ 55.69346874,  12.0806343 , 100.        ],
            [ 55.68227857,  12.08866043, 100.        ]])        
        """
        geo_projection = Proj("+proj=utm +zone=" + long_zone + " +" + hemisphere + " +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

        points_geo = np.array(list(reversed(geo_projection(points_utm[:,0], points_utm[:,1],inverse=True))))
        points_geo = np.append(points_geo, np.array([points_utm[:,2]]),axis = 0).transpose()

        return points_geo

    @staticmethod
    def generate_beam_coords(mesh_utm, measurement_pt):
        """
        Generates beam steering coordinates in spherical coordinate system
        from multiple lidar positions and single measurement point. 

        Parameters
        ----------
        mesh_utm : ndarray
            nD array containing data with `float` or `int` type
            corresponding to x, y and z coordinates of multiple lidar positions.
            nD array data are expressed in meters.
        meas_pt_pos : ndarray
            3D array containing data with `float` or `int` type
            corresponding to x, y and z coordinates of a measurement point.
            3D array data are expressed in meters.
        """
        # testing if  meas_pt has single or multiple measurement points
        if len(mesh_utm.shape) == 2:
            x_array = mesh_utm[:, 0]
            y_array = mesh_utm[:, 1]
            z_array = mesh_utm[:, 2]
        else:
            x_array = np.array([mesh_utm[0]])
            y_array = np.array([mesh_utm[1]])
            z_array = np.array([mesh_utm[2]])


        # calculating difference between lidar_pos and meas_pt_pos coordiantes
        dif_xyz = np.array([x_array - measurement_pt[0], y_array - measurement_pt[1], z_array - measurement_pt[2]])    

        # distance between lidar and measurement point in space
        distance_3D = np.sum(dif_xyz**2,axis=0)**(1./2)

        # distance between lidar and measurement point in a horizontal plane
        distance_2D = np.sum(np.abs([dif_xyz[0],dif_xyz[1]])**2,axis=0)**(1./2)

        # in radians
        azimuth = np.arctan2(measurement_pt[0] - x_array, measurement_pt[1] - y_array)
        # conversion to metrological convention
        azimuth = (360 + azimuth * (180 / np.pi)) % 360

        # in radians
        elevation = np.arccos(distance_2D / distance_3D)
        # conversion to metrological convention
        elevation = np.sign(measurement_pt[2] - z_array) * (elevation * (180 / np.pi))

        return np.transpose(np.array([azimuth, elevation, distance_3D]))  


    # def find_measurements(self):
    #         """
    #         Doc String
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