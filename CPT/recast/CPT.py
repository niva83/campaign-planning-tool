import numpy as np
from itertools import combinations, product
from pyproj import Proj


from osgeo import gdal, osr, ogr, gdal_array
import srtm

import pandas as pd
import geopandas
from shapely.geometry import Point
import whitebox
import simplekml

import matplotlib.pyplot as plt
from PIL import Image
import os, shutil

from random import shuffle

# for converting YAML to XML
import yaml # installation via conda
from xml.dom.minidom import parseString
import dicttoxml # installation via conda available

# for working with files and folders
from pathlib import Path
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

    _ , ncols = A.shape
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
    __yaml_template = {'skeleton': 
"""# This is the config file for the trajectory generation
# for long-range WindScanners written in YAML
#
# YAML config template author: Nikola Vasiljevic
# Template usage license: CC-BY-NC


# Header
coordinate_system: 'insertCOORDSYS'
utm_zone: 'insertUTMzone'
epsg_code: 'insertEPSGcode'
number_of_lidars: insertNOlidars
numer_of_measurement_scenarios: insertMOscenarios

# Lidar position description
lidar_positions:
insertLIDARdetails

# Measurement scenario description
measurement_scenarios:
    # Individual measurement scenario header
insertMEASUREMENTscenarios""", 'scenario':
"""
    - scenario_id: 'insertScenarioID'
      author: 'Unknown Artist'
      number_of_transects: insertNOtransects
      lidar_ids: insertLIDARids
      synchronized: insertSYNC # 0 - no, 1 - yes
      scan_type: insertSCANtype 
      fft_size: insertFFTsize # no points
      pulse_length: insertPULSElenght # in ns
      accumulation_time: insertACCtime # in ms
      min_range: insertMINrange # first range gate in m
      max_range: insertMAXrange # last range gate in m
insertTransects""", 'transect': 
"""
      transects:
       #Individual transect header
        - transect_no: 1 
          transect_points : 
insertTransectPoints""", 
     'points': 
"""
            - point : insertPTid
              easting : insertPTeasting 
              northing: insertPTnorthing 
              height: insertPTheight 
              move_time: insertPTtiming 
""", 'lidars': 
"""
lidar_positions:
insertLIDARS
""", 'lidar': 
"""
    - lidar_id: 'insertLIDARid' 
      easting: insertLIDAReasting 
      northing: insertLIDARnorthing 
      height: insertLIDARheight 
"""}
    __rg_template = """LidarMode	insertMODE
MaxDistance	insertMaxRange
FFTSize	insertFFTSize
insertRangeGates"""

    __pmc_template =  {'skeleton' : 
"""CLOSE
END GATHER
DELETE GATHER
DELETE TRACE


OPEN PROG 1983 CLEAR
P1988=0
P1983=0
P1000=1
M372=0
P1015=0
M1000=0
P1001=-3000
P1007=1
P1004=1
I5192=1
I5187=1
I5188=0
I322=insertPRF

CMD"#3HMZ"

#1->-8192X
#2->8192X+8192Y


IF (P1005=1)
    I5111=10000*8388608/I10
        WHILE(I5111>0)
                IF(P1004=1)
                    IF (P1008=0)
                        F(P1011)
                        X(1st_azimuth)Y(1st_elevation)
                    ENDIF
                END IF
            P1004=0
        END WHILE
ENDIF
P1000=2
WHILE(P1001!=-1999)
    insertMeasurements

    IF(P1001=-78)
        CMD"#3j/"
        P1001=-1999
    ENDIF

ENDWHILE
CMD"#3j/"
F30
X0Y0

WHILE(M133=0 OR M233=0)
END WHILE

; CLEARING ALL VARIABLES
P1000=0
M372=0
P1015=0
M1000=0
P1007=0
P1004=0
P1002=0
P1003=0
P1011=0
P1001=0
P1013=0
P1008=0
P1010=0
P1005=0

CLOSE""",
    "motion":
    """
    I5111 = (insertMotionTime)*8388608/I10
    TA(insertHalfMotionTime)TM(1)
    X(insertAzimuth)Y(insertElevation)
    WHILE(I5111>0)
    END WHILE
    WHILE(M133=0 OR M233=0)
    END WHILE

    CMD"#3j^insertTriggers"
    I5111 = (insertAccTime)*8388608/I10
    WHILE(I5111>0)
    END WHILE

    WHILE(M333=0)
    END WHILE

    Dwell(1)
    IF(P1983>0)
        I5111=P1983*8388608/I10
        WHILE(I5111>0)
        END WHILE
        P1984=P1983
        P1988=P1988+1
        P1983=0
    END IF
    """}

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
        if layer is not None:
            if len(np.unique(layer)) > 1:
                if 'levels' in kwargs:
                    levels = kwargs['levels']
                else:
                    min_value = np.min(layer)
                    max_value = np.max(layer)
                    increment = abs(max_value - min_value)/20
                    min_value = min_value
                    
                    levels = np.linspace(min_value, max_value, 20)
                    boundaries = np.linspace(min_value - increment/2, max_value + increment/2, 21)          
            
                if len(layer.shape) > 2:
                    layer = np.sum(layer, axis = 2)
                    levels = np.array(range(-1,int(np.max(layer)) + 1, 1))
                    boundaries = levels + 0.5
            
                fig, ax = plt.subplots(sharey = True, figsize=(800/self.MY_DPI, 800/self.MY_DPI), dpi=self.MY_DPI)
                cmap = plt.cm.RdBu_r
                cs = plt.pcolormesh(self.x, self.y, layer, cmap=cmap, alpha = 1)
            
            
                cbar = plt.colorbar(cs,orientation='vertical', ticks=levels, boundaries=boundaries,fraction=0.047, pad=0.01)
                if 'legend_label' in kwargs:
                    cbar.set_label(kwargs['legend_label'], fontsize = self.FONT_SIZE)
                    
                
                if 'points_type' in kwargs and kwargs['points_type'] in self.POINTS_TYPE:
                    measurement_pts = self.measurement_type_selector(kwargs['points_type'])
                else:
                    measurement_pts = self.measurement_type_selector(self.measurements_selector)        
            
                if measurement_pts is not None:
                    for i, pts in enumerate(measurement_pts):
                        if i == 0:
                            ax.scatter(pts[0], pts[1], marker='o', 
                            facecolors='yellow', edgecolors='black', 
                            s=80,zorder=1500, label = 'points: ' + self.measurements_selector)                    
                        else:
                            ax.scatter(pts[0], pts[1], marker='o',
                            facecolors='yellow', edgecolors='black', 
                            s=80,zorder=1500)
            
                if self.reachable_points is not None:
                    visible_points = measurement_pts[np.where(self.reachable_points>0)]
                    for i in range(0,len(visible_points)):
                        if i == 0:
                            ax.scatter(visible_points[i][0], visible_points[i][1], 
                                    marker='+', color='black', s=80,zorder=2000, label = "reachable")
                        else:
                            ax.scatter(visible_points[i][0], visible_points[i][1], 
                                    marker='+', color='black', s=80,zorder=2000)
            
                plt.xlabel('Easting [m]', fontsize = self.FONT_SIZE)
                plt.ylabel('Northing [m]', fontsize = self.FONT_SIZE)
            
                if 'title' in kwargs:
                    plt.title(kwargs['title'], fontsize = self.FONT_SIZE)
            
                ax.set_aspect(1.0)
            
                if len(self.lidar_dictionary) > 0:
                    for i, lidar in enumerate(self.lidar_dictionary):
                        lidar_pos = self.lidar_dictionary[lidar]['position']
                        ax.scatter(lidar_pos[0], lidar_pos[1], 
                                    marker='s', 
                                    facecolors=self.COLOR_LIST[i], edgecolors='white',linewidth='2',
                                    s=100, zorder=2000, label = 'lidar: ' + lidar)
            
                if self.lidar_pos_1 is not None or self.lidar_pos_2 is not None or measurement_pts is not None:
                    ax.legend(loc='lower right', fontsize = self.FONT_SIZE)    
            
            
                
                plt.show()
            
                if 'title' in kwargs and 'save_plot' in kwargs and kwargs['save_plot']:
                        fig.savefig(self.OUTPUT_DATA_PATH + kwargs['title'] + '.pdf', bbox_inches='tight')
            else:
                print('Provided layer elements all have the same value equal to: ' + str(np.unique(layer)[0]))
        else:
            print('Provided layer does not exist!')

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
            pts_str = kwargs['points_type']
        else:
            measurement_pts = self.measurement_type_selector('initial')
            pts_str = 'initial'


        if measurement_pts is not None and self.measurement_type_selector('optimized') is not None:
            fig, ax = plt.subplots(sharey = True, figsize=(800/self.MY_DPI, 800/self.MY_DPI), dpi=self.MY_DPI)

            for i,pt in enumerate(measurement_pts):
                if i == 0:
                    ax.scatter(pt[0], pt[1],marker='o', 
                        facecolors='red', edgecolors='black', 
                        s=10,zorder=1500, label = "points: " + pts_str)
                else:
                    ax.scatter(pt[0], pt[1],marker='o', 
                                        facecolors='red', edgecolors='black', 
                                        s=10,zorder=1500,)            


            for i,pt in enumerate(self.measurement_type_selector('optimized') ):
                if i == 0:
                    ax.scatter(pt[0], pt[1],marker='o', 
                        facecolors='white', edgecolors='black', 
                        s=10,zorder=1500, label = "points: optimized")
                    ax.add_artist(plt.Circle((pt[0], pt[1]), 
                                            self.REP_RADIUS,                               
                                            facecolor='grey', edgecolor='black', 
                                            zorder=0,  alpha = 0.5))                 
                else:
                    ax.scatter(pt[0], pt[1],marker='o', 
                        facecolors='white', edgecolors='black', 
                        s=10,zorder=1500)
                    ax.add_artist(plt.Circle((pt[0], pt[1]), 
                                            self.REP_RADIUS,                               
                                            facecolor='grey', edgecolor='black', 
                                            zorder=0,  alpha = 0.5))                 
    
                    

            plt.xlabel('Easting [m]', fontsize = self.FONT_SIZE)
            plt.ylabel('Northing [m]', fontsize = self.FONT_SIZE)
            ax.legend(loc='lower right', fontsize = self.FONT_SIZE)


            ax.set_xlim(np.min(self.x),np.max(self.x))
            ax.set_ylim(np.min(self.y),np.max(self.y))

            ax.set_aspect(1.0)
            plt.show()
            if 'save_plot' in kwargs and kwargs['save_plot']:
                fig.savefig(self.OUTPUT_DATA_PATH + 'measurements_optimized.pdf', bbox_inches='tight')


    def plot_layout(self, **kwargs):
        """
        Plots campaign layout.
        
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
        if self.flags['trajectory_optimized']:

            # levels = np.array(range(-1,self.combined_layer.shape[-1] + 1, 1))
            # layer = np.sum(self.combined_layer, axis = 2)
            levels = np.linspace(np.min(self.orography_layer), np.max(self.orography_layer), 20)

            fig, ax = plt.subplots(sharey = True, figsize=(800/self.MY_DPI, 800/self.MY_DPI), dpi=self.MY_DPI)
            cmap = plt.cm.Greys
            cs = plt.contourf(self.x, self.y, self.orography_layer, levels=levels, cmap=cmap, alpha = 0.4)

            cbar = plt.colorbar(cs,orientation='vertical',fraction=0.047, pad=0.01)
            cbar.set_label('Height asl [m]', fontsize = self.FONT_SIZE)
            
            ax.scatter(self.lidar_pos_1[0], self.lidar_pos_1[1], marker='o', 
            facecolors='black', edgecolors='white', s=60, zorder=2000, label = "lidar_1")

            ax.scatter(self.lidar_pos_2[0], self.lidar_pos_2[1], marker = 'o', 
            facecolors='white', edgecolors='black',s=60,zorder=2000, label = "lidar_2")

            if 'points_type' in kwargs and kwargs['points_type'] in self.POINTS_TYPE:
                measurement_pts = self.measurement_type_selector(kwargs['points_type'])
            else:
                measurement_pts = self.measurement_type_selector(self.measurements_selector)        

            if measurement_pts is not None:
                for i, pts in enumerate(measurement_pts):
                    if i == 0:
                        ax.scatter(pts[0], pts[1], marker='o', 
                        facecolors='yellow', edgecolors='black', 
                        s=60,zorder=1500, label = 'measurements_' + self.measurements_selector)                    
                    else:
                        ax.scatter(pts[0], pts[1], marker='o',
                        facecolors='yellow', edgecolors='black', 
                        s=60,zorder=1500)

            if self.measurements_reachable is not None:
                for i in range(0,len(self.measurements_reachable)):
                    if i == 0:
                        ax.scatter(self.measurements_reachable[i][0], self.measurements_reachable[i][1], 
                                marker='+', 
                                color='red',  edgecolors='black', 
                                s=80,zorder=2000, label = "measurements_reachable")
                    else:
                        ax.scatter(self.measurements_reachable[i][0], self.measurements_reachable[i][1], 
                                marker='+', 
                                facecolors='red', 
                                s=80,zorder=2000)

                ax.plot(self.trajectory[:,0],self.trajectory[:,1],
                        color='blue', linestyle='--',linewidth=1, zorder=3000,label='trajectory')

                ax.scatter(self.trajectory[0,0], self.trajectory[0,1], 
                       marker='o', facecolors='white', edgecolors='green',s=120,zorder=1400,label = "trajectory start")



            ax.legend(loc='lower right', fontsize = self.FONT_SIZE)    

            plt.xlabel('Easting [m]', fontsize = self.FONT_SIZE)
            plt.ylabel('Northing [m]', fontsize = self.FONT_SIZE)


            plt.title('Campaign layout', fontsize = self.FONT_SIZE)

            ax.set_aspect(1.0)
            plt.show()

            if 'save_plot' in kwargs and kwargs['save_plot']:
                    fig.savefig(self.OUTPUT_DATA_PATH + 'campaign_layout' + '.pdf', bbox_inches='tight')
        else:
            print('Trajectory not optimized -> nothing to plot')

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
                    self.measurements_initial = kwargs['measurements']
                elif self.measurements_selector == 'optimized':
                    self.measurements_initial = kwargs['measurements']
                elif self.measurements_selector == 'reachable':
                    self.measurements_initial = kwargs['measurements']
                elif self.measurements_selector == 'identified':
                    self.measurements_initial = kwargs['measurements']
                else:
                    self.measurements_initial = kwargs['measurements']
            else:
                if self.measurements_selector == 'initial':
                    self.measurements_initial = np.array([kwargs['measurements']])
                elif self.measurements_selector == 'optimized':
                    self.measurements_optimized = np.array([kwargs['measurements']])
                elif self.measurements_selector == 'reachable':
                    self.measurements_reachable = np.array([kwargs['measurements']])
                elif self.measurements_selector == 'identified':
                    self.measurements_reachable = np.array([kwargs['measurements']])
                else:
                    self.measurements_misc = np.array([kwargs['measurements']])

            self.flags['measurements_added'] = True

    def add_measurement_instances(self, **kwargs):
        """
        Adds measurement points, given in as UTM coordinates,
        to the measurement points dictionary.
        
        Parameters
        ----------
            **kwargs : see below

        Keyword Arguments
        -----------------
        points_type : str
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
        [577979, 4844819, 478 + 80],]), points_type = 'initial')
        Measurement points 'initial' added to the measurements dictionary!
        Measurements dictionary contains 1 different measurement type(s).

        """
        if self.flags['utm_set']:
            if 'points_type' in kwargs:
                if kwargs['points_type'] in self.POINTS_TYPE:
                    if 'points' in kwargs:
                        if len(kwargs['points'].shape) == 2 and kwargs['points'].shape[1] == 3:

                            points_pd = pd.DataFrame(kwargs['points'], 
                                                    columns = ["Easting [m]", "Northing [m]","Height asl [m]"])

                            points_pd.insert(loc=0, column='Point no.', value=np.array(range(1,len(points_pd) + 1)))
                            pts_dict = {kwargs['points_type']: points_pd}
                            self.measurements_dictionary.update(pts_dict)

                            print('Measurement points \'' + kwargs['points_type'] + '\' added to the measurements dictionary!')
                            print('Measurements dictionary contains ' + str(len(self.measurements_dictionary)) + ' different measurement type(s).')
                            self.flags['measurements_added'] = True
                            self.measurements_selector = kwargs['points_type']
                        else:
                            print('Incorrect position information, cannot add measurements!')
                            print('Input measurement points must be a numpy array of shape (n,3) where n is number of points!')
                    else:
                        print('Measurement points not specified, cannot add points!')
                else:
                    print('Measurement point type not permitted!')
                    print('Allowed types are: \'initial\', \'optimized\', \'reachable\', \'identified\' and \'misc\'')
            else:
                print('Measurement points\' type not provided, cannot add measurement points!')
        else:
            print('UTM zone not specified, cannot add measurement points!')

    def update_reachable_points(self, **kwargs):

        
        if ('points_type' in kwargs and 
            kwargs['points_type'] in self.POINTS_TYPE and 
            self.measurements_dictionary[kwargs['points_type']] is not None
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
                    if kwargs['points_type'] == measurement_id[0]:
                        print('Finding reachable points which are common for lidar instances:' + str(kwargs['lidar_ids']))
                        measurement_pts = self.measurement_type_selector(kwargs['points_type'])
                        self.measurement_selector = 'reachable'
                        all_ones = np.full(len(measurement_pts),1)
                        for lidar in kwargs['lidar_ids']:
                            reachable_pts = self.lidar_dictionary[lidar]['reachable_points']
                            all_ones = all_ones * reachable_pts            
                        pts_ind = np.where(all_ones == 1)
                        print('Updating self.measurements_dictionary instance \'reachable\' with common reachable points')
                        self.add_measurement_instances(points = measurement_pts[pts_ind], 
                                                       points_type = 'reachable')
                        print('Optimizing trajectory through the common reachable points for lidar instances:' + str(kwargs['lidar_ids']))
                        self.optimize_trajectory(points_type = 'reachable', 
                                                 lidar_ids = kwargs['lidar_ids'])
                                              
                        print('Lidar instances:' + str(kwargs['lidar_ids']) + ' will be updated with the common reachable points and optimized trajectory')
                        for lidar in kwargs['lidar_ids']:
                            self.update_lidar_instance(lidar_id = lidar, use_optimized_trajectory = True, points_type = 'reachable')

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

    def sync_trajectory(self, **kwargs):
        if ('lidar_ids' in kwargs and set(kwargs['lidar_ids']).issubset(self.lidar_dictionary)):
            print('Synchronizing trajectories for lidar instances:' + str(kwargs['lidar_ids']))                                                 
            sync_time = []
            try:
                for lidar in kwargs['lidar_ids']:
                    motion_table = self.lidar_dictionary[lidar]['motion_config']
                    timing = motion_table.loc[:, 'Move time [ms]'].values
                    sync_time = sync_time + [timing]

                sync_time = np.max(np.asarray(sync_time).T, axis = 1)

                
                for lidar in kwargs['lidar_ids']:
                    self.lidar_dictionary[lidar]['motion_config']['Move time [ms]'] = sync_time
            except:
                print('Number of trajectory points for lidar instances don\'t match!')
                print('Aborting the operation!')

        else: 
            print('One or more lidar ids don\'t exist in the lidar dictionary')
            print('Available lidar ids: ' + str(list(self.lidar_dictionary.keys())))
            print('Aborting the operation!')
                
            
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

            # removing discs which cover same points
            unique_discs = np.unique(matrix, return_index= True, axis = 0)
            matrix = unique_discs[0]
            discs = discs[unique_discs[1]]

            # remove discs which cover only one point
            ind = np.where(np.sum(matrix,axis = 1) > 1)
            matrix = matrix[ind]
            discs = discs[ind]


            total_covered_points = np.sum(matrix,axis = 1)

            matrix = matrix[(-1*total_covered_points).argsort()]
            discs = discs[(-1*total_covered_points).argsort()]



            # adding 0 m for elevation of each disc
            discs = np.append(discs.T, np.array([np.zeros(len(discs))]),axis=0).T

            self.discs = discs
            self.matrix = matrix

            return discs, matrix
        else:
            return print("No measurement points -> nothing to optimize!")

    @staticmethod
    def find_unique_indexes(matrix):

        unique_indexes = []
        none_unique_indexes = []
        for i in range(0,len(matrix)):
            sub_matrix = np.delete(matrix, i, axis = 0)
            sum_rows = np.sum(sub_matrix, axis = 0)
            sum_rows[np.where(sum_rows>0)] = 1
            product_rows = sum_rows * matrix[i]
            product_rows = product_rows + matrix[i]
            product_rows[np.where(product_rows>1)] = 0

            if np.sum(product_rows) > 0:
                unique_indexes = unique_indexes + [i]
            else:
                none_unique_indexes = none_unique_indexes + [i]
                
        return unique_indexes, none_unique_indexes

    @classmethod
    def minimize_discs(cls, matrix,disc):
        unique_indexes, none_unique_indexes = cls.find_unique_indexes(matrix)
        if len(none_unique_indexes) > 0:

            disc_unique = disc[unique_indexes]
            matrix_unique = matrix[unique_indexes]

            disc_none_unique = disc[none_unique_indexes]
            matrix_none_unique = matrix[none_unique_indexes]



            row_sum = np.sum(matrix_unique, axis = 0)
            # coverting all elements > 0 to 1
            row_sum[np.where(row_sum > 0)] = 1

            # removing all covered elements
            matrix_test = matrix_none_unique* (1 - row_sum)

            # removing discs that cover the same uncovered points
            unique_elements = np.unique(matrix_test, return_index= True, axis = 0)
            remaining_indexes = unique_elements[1]
            matrix_test = matrix_test[remaining_indexes]
            disc_test = disc_none_unique[remaining_indexes]

            # sorting by the number of covered points prior test
            total_covered_points = np.sum(matrix_test,axis = 1)
            matrix_test = matrix_test[(-1*total_covered_points).argsort()]
            disc_test = disc_test[(-1*total_covered_points).argsort()]

            covered_pts_ind = np.unique(np.where(matrix_unique > 0)[1])
            new_indexes = [] 
            for i, row in enumerate(matrix_test):
                covered_pts_ind_new = np.where(row > 0)[0]
                
                uncovered_pts_ind = np.setdiff1d(covered_pts_ind_new, covered_pts_ind)
                if len(uncovered_pts_ind):
                    covered_pts_ind = np.append(covered_pts_ind, uncovered_pts_ind)
                    new_indexes = new_indexes + [i]
                    
            if len(new_indexes) > 0:
                disc_unique = np.append(disc_unique, disc_test[new_indexes], axis = 0)
            
            return disc_unique
        else:
            return disc[unique_indexes]

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
        measure_pt_height = abs(measurement_pts[:,2] -  self.get_elevation(self.long_zone + self.lat_zone, measurement_pts))


        if measurement_pts is not None:
            print('Optimizing ' + self.measurements_selector + ' measurement points!')
            discs, matrix = self.generate_disc_matrix()


            points_uncovered = measurement_pts
            points_covered_total = np.zeros((0,3), measurement_pts.dtype)
            # discs_selected = np.zeros((0,3))
            i = 0
            j = len(points_uncovered)
            disc_indexes = []
            while i <= (len(discs) - 1) and j > 0 :
                indexes = np.where(matrix[i] == 1 )
                # matrix = matrix * (1 - matrix[i])
                points_covered = measurement_pts[indexes]
                points_new = array_difference(points_covered, points_covered_total)

                if len(points_new) > 0:
                    points_covered_total = np.append(points_covered_total, points_new,axis=0)
                    # discs_selected = np.append(discs_selected, np.array([discs[i]]),axis=0)
                    disc_indexes = disc_indexes + [i]
                points_uncovered = array_difference(points_uncovered, points_covered)
                i += 1
                j = len(points_uncovered)

            # makes subset of discs and matrix
            discs_selected = discs[disc_indexes]
            matrix_selected = matrix[disc_indexes]

            # minimize number of discs
            if len(discs_selected) > 1:
                discs_selected = self.minimize_discs(matrix_selected,discs_selected)
            self.disc_temp = discs_selected
            # if we don't cover all the points
            # remaining uncovered points must be
            # added to the array
            self.uncovered = points_uncovered
            self.covered = points_covered_total
            if len(points_uncovered) > 0:
                measurements_optimized = np.append(discs_selected, points_uncovered, axis = 0)
                terrain_height = self.get_elevation(self.long_zone + self.lat_zone, measurements_optimized)
                measurements_optimized[:, 2] = terrain_height + np.average(measure_pt_height)
                self.add_measurement_instances(points = measurements_optimized, points_type = 'optimized')

            # if we cover all the points then
            # the optimized measurements are
            # are equal to the disc centers
            else:
                measurements_optimized = discs_selected
                terrain_height = self.get_elevation(self.long_zone + self.lat_zone, measurements_optimized)
                measurements_optimized[:, 2] = terrain_height + np.average(measure_pt_height)
                self.add_measurement_instances(points = measurements_optimized, points_type = 'optimized')

            # in case when none of the measurement
            # points are covered by this method than
            # the optimized points should be equal to
            # the original measurements points
            # if len(measurements_optimized) == len(measurement_pts):
            #     self.add_measurement_instances(points = measurement_pts, points_type = 'optimized')
                
        else:
            print("No measurement positions added, nothing to optimize!")

    def optimize_trajectory(self, **kwargs):
        """
        Finding a shortest trajectory through the set of measurement points.
        
        Parameters
        ----------
        see kwargs

        
        Keyword paramaters
        ------------------
        points_type : str, required
            A string indicating which measurement points
            should be consider for the trajectory optimization.
        lidar_ids : list of str, required
            A list of strings containing lidar ids.
        sync : bool, optional
            Indicates whether to sync trajectories or not
        
        Returns
        -------
        self.trajectory : ndarray
            An ordered nD array containing trajectory points.
        
        See also
        --------
        self.tsp : adapted traveling salesman problem for scanning lidars
        self.generate_trajectory : generation of synchronized trajectories

        Notes
        --------
        The optimization of the trajectory is performed by applying the adapted 
        traveling salesman problem to the measurement point set while varing the
        starting point of the trajectory. This secures the shortest trajectory. 

        References
        ----------
        .. [1] Nikola Vasiljevic, Andrea Vignaroli, Andreas Bechmann and 
            Rozenn Wagner: Digitalization of scanning lidar measurement
            campaign planning, https://www.wind-energ-sci-discuss.net/
            wes-2019-13/#discussion, 2017.
        Examples
        --------
        """        
        # selecting points which will be used for optimization
        if ('points_type' in kwargs and
             kwargs['points_type'] in self.POINTS_TYPE and
             kwargs['points_type'] in self.measurements_dictionary and
             len(self.measurements_dictionary[kwargs['points_type']]) > 0
            ):
            if ('lidar_ids' in kwargs and
                 set(kwargs['lidar_ids']).issubset(self.lidar_dictionary)
                ):

                measurement_pts = self.measurements_dictionary[kwargs['points_type']].values[:, 1:].tolist()
                self.measurements_selector = kwargs['points_type']
                sync_time_list = []
                for i in range(0,len(measurement_pts)):
    
                    trajectory = self.tsp(i, **kwargs)

                    # needs to record each lidar timing for each move
                    # and then 'if we want to keep them in syn
                    sync_time = []
                    for lidar in kwargs['lidar_ids']:

                        motion_table = self.generate_trajectory(self.lidar_dictionary[lidar]['position'], 
                                                                trajectory)
                        timing = motion_table.loc[:, 'Move time [ms]'].values
                        sync_time = sync_time + [timing]
                    sync_time = np.sum(np.max(np.asarray(sync_time).T, axis = 1))
                    sync_time_list = sync_time_list + [sync_time]
                    
                        # if i == 0:
                        #     total_time.update({lidar:{i : timing}})
                        # else:
                        #     total_time[lidar].update({i : timing})

                sync_time_list = np.asarray(sync_time_list)
                self.temp = sync_time_list
                # this returns tuple, and sometimes by chance there 
                # are two min values we are selecting first one!
                # first 0 means to select the array from the tuple, 
                # while second 0 results in selecting the first min value
                min_traj_ind = np.where(sync_time_list == np.min(sync_time_list))[0][0]
                trajectory = self.tsp(min_traj_ind, **kwargs)
                
                trajectory = pd.DataFrame(trajectory, columns = ["Easting [m]", 
                "Northing [m]", 
                "Height asl [m]"])

                trajectory.insert(loc=0, column='Point no.', value=np.array(range(1,len(trajectory) + 1)))
                self.trajectory = trajectory
                self.flags['trajectory_optimized'] = True   
             
                print('Lidar instances:' + str(kwargs['lidar_ids']) + ' will be updated with the optimized trajectory')
                for lidar in kwargs['lidar_ids']:
                    self.update_lidar_instance(lidar_id = lidar, 
                                            use_optimized_trajectory = True, 
                                            points_type = kwargs['points_type'])

                if 'sync' in kwargs and kwargs['sync']:
                    self.sync_trajectory(**kwargs)                            


            else: 
                print('One or more lidar ids don\'t exist in the lidar dictionary')
                print('Available lidar ids: ' + str(list(self.lidar_dictionary.keys())))

        else:
            print('Either point type id does not exist or for the corresponding measurement dictionary instance there are no points!')


    def tsp(self, start=None, **kwargs):
        """
        Solving a travelening salesman problem for a set of points and 
        two lidar positions.
        
        Parameters
        ----------
        start : int
            Presetting the trajectory starting point.
            A default value is set to None.
        
        Keyword paramaters
        ------------------
        points_type : str
            A string indicating which measurement points
            should be consider for the trajectory optimization.
        lidar_ids : list of str
            A list of strings containing lidar ids.
        
        Returns
        -------
        trajectory : ndarray
            An ordered nD array containing optimized trajectory points.
        
        See also
        --------
        self.generate_trajectory : generation of synchronized trajectories

        Notes
        --------
        The optimization of the trajectory is performed through the adaptation
        of the Nearest Neighbor Heuristics solution for the traveling salesman
        problem [1,2]. 

        References
        ----------
        .. [1] Nikola Vasiljevic, Andrea Vignaroli, Andreas Bechmann and 
            Rozenn Wagner: Digitalization of scanning lidar measurement
            campaign planning, https://www.wind-energ-sci-discuss.net/
            wes-2019-13/#discussion, 2017.
        .. [2] Reinelt, G.: The Traveling Salesman: Computational Solutions 
            for TSP Applications, Springer-Verlag, Berlin, Heidelberg, 1994.

        Examples
        --------
        """
        if ('points_type' in kwargs and
             kwargs['points_type'] in self.POINTS_TYPE and
             kwargs['points_type'] in self.measurements_dictionary and
             len(self.measurements_dictionary[kwargs['points_type']]) > 0
            ):
            if ('lidar_ids' in kwargs and
                 set(kwargs['lidar_ids']).issubset(self.lidar_dictionary)
                ):
                points = self.measurements_dictionary[kwargs['points_type']].values[:, 1:].tolist()

                if start is None:
                    shuffle(points)
                    start = points[0]
                else:
                    start = points[start]

                unvisited_points = points
                # sets first trajectory point        
                trajectory = [start]
                # removes that point from the points list
                unvisited_points.remove(start)

                # lidar list
                lidars = []
                for lidar in kwargs['lidar_ids']:
                    lidars = lidars + [self.lidar_dictionary[lidar]['position']]

                while unvisited_points:
                    last_point = trajectory[-1]
                    max_angular_displacements = []

                    # calculates maximum angular move from the last
                    # trajectory point to any other point which is not
                    # a part of the trajectory            
                    for next_point in unvisited_points:
                        max_displacement = self.calculate_max_move(last_point, next_point, lidars)
                        max_angular_displacements = max_angular_displacements + [max_displacement]

                    # finds which displacement is shortest
                    # and the corresponding index in the list
                    min_displacement = min(max_angular_displacements)
                    index = max_angular_displacements.index(min_displacement)

                    # next trajectory point is added to the trajectory
                    # and removed from the list of unvisited points
                    next_trajectory_point = unvisited_points[index]
                    trajectory.append(next_trajectory_point)
                    unvisited_points.remove(next_trajectory_point)
                trajectory = np.asarray(trajectory)
                return trajectory
            else: 
                print('One or more lidar ids don\'t exist in the lidar dictionary')
                print('Available lidar ids: ' + str(list(self.lidar_dictionary.keys())))
                trajectory = None
                return trajectory
        else:
            print('Either point type id does not exist or selected there are no points!')
            trajectory = None
            return trajectory

    @staticmethod
    def displacement2time(displacement, Amax, Vmax):

        time = np.empty((len(displacement),), dtype=float)
        # find indexes for which the scanner head 
        # will reach maximum velocity (i.e. rated speed)
        index_a = np.where(displacement > (Vmax**2) / Amax)

        # find indexes for which the scanner head 
        # will not reach maximum velocity (i.e. rated speed)
        index_b = np.where(displacement <= (Vmax**2) / Amax)

        time[index_a] = displacement[index_a] / Vmax + Vmax / Amax
        time[index_b] = 2 * np.sqrt(displacement[index_b] / Amax)


        return time

    @staticmethod
    def generate_range_gate_file(template_str, no_los, range_gates, lidar_mode, fft_size, accumulation_time):
        range_gate_file = template_str
        range_gate_file = range_gate_file.replace("insertMODE", str(lidar_mode))
        range_gate_file = range_gate_file.replace("insertMaxRange", str(max(range_gates)))
        range_gate_file = range_gate_file.replace("insertFFTSize", str(fft_size))

        rows = ""
        range_gate_row = "\t".join(list(map(str, range_gates)))

        for i in range(0, no_los):
            row_temp = str(i+1) + '\t' + str(accumulation_time) + '\t'
            row_temp = row_temp + range_gate_row

            if i < no_los - 1:
                row_temp = row_temp + '\n'
            rows = rows + row_temp

        range_gate_file = range_gate_file.replace("insertRangeGates", rows)

        return range_gate_file

    
    def generate_trajectory(self, lidar_pos, trajectory):

        _, angles_stop, angular_displacement =  self.trajectory2displacement(lidar_pos, trajectory)


        move_time = self.displacement2time(np.max(angular_displacement, axis = 1),
                                           self.MAX_ACCELERATION, 
                                           self.MAX_VELOCITY)


        timing = np.ceil(move_time * 1000)

        matrix = np.array([angles_stop[:,0],
                           angles_stop[:,1],
                           timing]).T

        motion_table = pd.DataFrame(matrix, columns = ["Azimuth [deg]", "Elevation [deg]", "Move time [ms]"])
        first_column = []

        for i in range(0, len(angular_displacement)):
            if i == 0:
                insert_str = str(len(angular_displacement)) + '->' + str(i + 1)
                first_column = first_column + [insert_str]
            else:
                insert_str = str(i) + '->' + str(i+1) 
                first_column = first_column + [insert_str]

            # # old way
            # if i != len(angular_displacement) - 1:
            #     insert_str = str(i + 1) + '->' + str(i + 2)
            #     first_column = first_column + [insert_str]
            # else:
            #     insert_str = str(i + 1) + '->1' 
            #     first_column = first_column + [insert_str]

        motion_table.insert(loc=0, column='Step-stare order', value=first_column)
        return motion_table

    @classmethod
    def calculate_max_move(cls, point1, point2, lidars):
        azimuth_max = max(map(lambda x: abs(cls.rollover(point1, point2, x)[0]), lidars))
        elevation_max = max(map(lambda x: abs(cls.rollover(point1, point2, x)[1]), lidars))

        return max(azimuth_max,elevation_max)

    @classmethod
    def rollover(cls, point1, point2, windscanner):
        angles_1 = cls.generate_beam_coords(windscanner, point1, 1)[0]
        angles_2 = cls.generate_beam_coords(windscanner, point2, 1)[0]

        if abs(angles_1[0] - angles_2[0]) > 180:
            if abs(360 - angles_1[0] + angles_2[0]) < 180:
                azimuth_displacement = 360 - angles_1[0] + angles_2[0]
            else:
                azimuth_displacement = 360 + angles_1[0] - angles_2[0]
        else:
            azimuth_displacement =  angles_1[0] - angles_2[0]

        if abs(angles_1[1] - angles_2[1]) > 180:
            if abs(360 - angles_1[1] + angles_2[1]) < 180:
                elevation_displacement = 360 - angles_1[1] + angles_2[1]
            else:
                elevation_displacement = 360 + angles_1[1] - angles_2[1]
        else:
            elevation_displacement =  angles_1[1] - angles_2[1]


        return np.array([azimuth_displacement,elevation_displacement])

    @classmethod
    def trajectory2displacement(cls, lidar_pos, trajectory, rollover = True):
        angles_start = cls.generate_beam_coords(lidar_pos, 
                                                np.roll(trajectory, 1, axis = 0), opt = 0)[:, (0,1)]
        # angles_start = cls.generate_beam_coords(lidar_pos, trajectory, opt = 0)[:, (0,1)]
        # -1 performs shift-left for one element (originally used!)
        # 1 performs shift-right for one element
        angles_stop = cls.generate_beam_coords(lidar_pos, 
                                               np.roll(trajectory, 0, axis = 0), opt = 0)[:, (0,1)]
        angular_displacement = abs(angles_start - angles_stop)


        ind_1 = np.where((angular_displacement[:, 0] > 180) & 
                         (abs(360 - angular_displacement[:, 0]) <= 180))

        ind_2 = np.where((angular_displacement[:, 0] > 180) & 
                         (abs(360 - angular_displacement[:, 0]) > 180))
        angular_displacement[:, 0][ind_1] = 360 - angular_displacement[:, 0][ind_1]
        angular_displacement[:, 0][ind_2] = 360 + angular_displacement[:, 0][ind_2]


        ind_1 = np.where((angular_displacement[:, 1] > 180) & 
                         (abs(360 - angular_displacement[:, 1]) <= 180))
        ind_2 = np.where((angular_displacement[:, 1] > 180) & 
                         (abs(360 - angular_displacement[:, 1]) > 180))
        angular_displacement[:, 1][ind_1] = 360 - angular_displacement[:, 1][ind_1]
        angular_displacement[:, 1][ind_2] = 360 + angular_displacement[:, 1][ind_2]
        return np.round(angles_start, cls.NO_DIGITS), np.round(angles_stop,cls.NO_DIGITS), np.abs(angular_displacement)

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
        gis_layer_id : str, optional
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
        'gis_layer_id' points to either 'combined' or 'second_lidar' layer. If
        'gis_layer_id' is not provided the method will use 'combined' layer.

        If 'use_optimized_trajectory' is set to True, it is required that the 
        method self.optimize_trajectory was run prior the current method, 
        otherwise the current method will update the lidar instance considering
        the order of measurement points as is.

        Currently the method only support step-stare trajectory, so the argument
        wheter on not set 'motion_type' it will not impact the trajectory calculation.


        Examples
        --------

        """

        if ('points_type' in kwargs and 
            kwargs['points_type'] in self.POINTS_TYPE and 
            kwargs['points_type'] in self.measurements_dictionary
            ):
            kwargs.update({'points_type' : kwargs['points_type']})
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
        points_type : str, optional
            Indicates which points to be used to update the lidar instace.
        use_reachable_points : boolean, optional
            Indicates whether to update the lidar instance
            only considering the reachable points.
        gis_layer_id : str, optional
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
        'gis_layer_id' points to either 'combined' or 'second_lidar' layer. If
        'gis_layer_id' is not provided the method will use 'combined' layer.

        If 'use_optimized_trajectory' is set to True, it is required that the 
        method self.optimize_trajectory was run prior the current method, 
        otherwise the current method will update the lidar instance considering
        the order of measurement points as is.

        Currently the method only support step-stare trajectory, so the argument
        wheter on not set 'motion_type' it will not impact the trajectory calculation.


        Examples
        --------

        """
        if ('points_type' in kwargs and 
            kwargs['points_type'] in self.POINTS_TYPE and 
            kwargs['points_type'] in self.measurements_dictionary
            ):
            measurement_pts = self.measurement_type_selector(kwargs['points_type'])
            self.measurements_selector = kwargs['points_type']

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
                                'gis_layer_id' in kwargs and
                                (kwargs['gis_layer_id'] == 'combined' or 
                                kwargs['gis_layer_id'] == 'second_lidar_placement') and
                                self.layer_selector(kwargs['gis_layer_id']) is not None
                            ):
                            layer = self.layer_selector(kwargs['gis_layer_id'])
                            i, j = self.find_mesh_point_index(self.lidar_dictionary[kwargs['lidar_id']]['position'])
                            self.lidar_dictionary[kwargs['lidar_id']]['reachable_points'] = layer[i,j,:]                        
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
        elif len(measurement_pts) > 0:
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

            nrows, ncols = self.x.shape

            self.mesh_utm = np.array([self.x.T, self.y.T, self.z.T]).T.reshape(-1, 3)
            self.mesh_geo = self.utm2geo(self.mesh_utm, self.long_zone, self.hemisphere)
            self.mesh_indexes = np.array(range(0,len(self.mesh_utm),1)).reshape(nrows,ncols)         
            self.flags['mesh_generated'] = True


    def find_mesh_point_index(self, point):
        """
        Finds index of the closest point in a set
        to the test point
        """
        dist_2D = np.sum((self.mesh_utm[:,(0,1)] - point[:2])**2, axis=1)
        index = np.argmin(dist_2D)

        i, j = np.array(np.where(self.mesh_indexes == index)).flatten()
        
        return i, j

    def generate_campaign_layout(self):
        if self.flags['lidar_pos_2'] and self.flags['second_lidar_layer']:
            i, j = self.find_mesh_point_index(self.lidar_pos_2)
            self.reachable_points = self.second_lidar_layer[i,j,:]
            measurement_pts = self.measurement_type_selector(self.measurements_selector)
            self.measurements_reachable = measurement_pts[np.where(self.reachable_points>0)]
            self.flags['measurements_reachable'] = True

            # call for trajectory optimization
            # call for trajectory generation
        else:
            print('Previous steps are not completed!!!')




    def generate_second_lidar_layer(self, **kwargs):
        self.measurements_selector = self.combined_layer_pts_type
        measurement_pts = self.measurement_type_selector(self.measurements_selector)
        if len(measurement_pts) > 0:
            if 'lidar_id' in kwargs:
                if kwargs['lidar_id'] in self.lidar_dictionary:
                    # storing id of the lidar used to generate second lidar placement layer
                    self.first_lidar_id = kwargs['lidar_id']
                    lidar_position = self.lidar_dictionary[kwargs['lidar_id']]['position']
                    self.generate_intersecting_angle_layer(lidar_position, measurement_pts)
                    self.flags['intersecting_angle_layer_generated'] = True
                    self.update_lidar_instance(lidar_id = kwargs['lidar_id'], points_type = self.measurements_selector)
                    self.second_lidar_layer = self.combined_layer * self.intersecting_angle_layer
                    # reachable_points = self.lidar_dictionary[kwargs['lidar_id']]['reachable_points']
                    # self.second_lidar_layer = self.combined_layer * self.intersecting_angle_layer * reachable_points
                    self.flags['second_lidar_layer'] = True
                else:
                    print('Lidar does not exist in self.lidar dict, halting operation!')
            else:
                print('Lidar id not provided as the keyword argument, halting operation!')
        else:
            print('No measurement points provided, halting operation!')


    
    def generate_intersecting_angle_layer(self, lidar_position, measurement_pts):

        nrows, ncols = self.x.shape
        no_pts = len(measurement_pts)
        azimuths_1 = self.generate_beam_coords(lidar_position,measurement_pts,0)[:,0]
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

        if ('points_type' in kwargs and 
            kwargs['points_type'] in self.POINTS_TYPE and 
            kwargs['points_type'] in self.measurements_dictionary
            ):
            self.measurements_selector = kwargs['points_type']

            if len(self.measurement_type_selector(self.measurements_selector)) > 0:
                print('Generating combined layer for ' + self.measurements_selector + ' measurement points!')
                self.generate_mesh()
                self.generate_topographic_layer()
                self.generate_beam_coords_mesh(**kwargs)
                self.generate_range_restriction_layer()
                self.generate_elevation_restriction_layer()
                self.generate_los_blck_layer(**kwargs)

                nrows, ncols = self.x.shape
                if (self.flags['los_blck_layer_generated'] and 
                    self.flags['topography_layer_generated']
                    ):

                    self.combined_layer = self.elevation_angle_layer * self.range_layer * self.los_blck_layer

                    if self.flags['landcover_layer_generated']:
                        self.combined_layer = self.combined_layer * self.restriction_zones_layer.reshape((nrows,ncols,1))
                        self.flags['combined_layer_generated'] = True
                        self.combined_layer_pts_type = kwargs['points_type']
                    else:
                        print('Combined layer generated without landcover data!')
                        self.flags['combined_layer_generated'] = True
                        self.combined_layer_pts_type = kwargs['points_type']
                else:
                    print('Either topography or los blockage layer are missing!')
                    print('Aborting the combined layer generation!')
            else:
                print('Instance in self.measurements_dictionary for type'+ self.measurements_selector + ' is empty!')
                print('Aborting the combined layer generation!')
        else:
            print('Either points_type was not provided or for the provided points_type there is no instance in self.measurements_dictionary!')
            print('Aborting the combined layer generation!')


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
        if self.flags['topography_layer_generated']:
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
                    print('For points type \''+ self.measurements_selector + '\' there are no measurement points in the measurements dictionary!')
                    print('Aborting the los blockage layer generation!')
            else:                
                print('Points type \''+ self.measurements_selector + '\' does not exist in the measurements dictionary!')
                print('Aborting the los blockage layer generation!')
        else:
            print('Topography layer not generates!')
            print('Aborting the los blockage layer generation!')


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
                file_name_str = "los_blockage_" + str(i+1) + ".asc"
                file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str) 
                los_blck_tmp  = np.loadtxt(file_path.absolute().as_posix(), skiprows=6)
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
        if (self.flags['topography_exported'] and 
            self.flags['measurements_exported'] and 
            self.measurement_type_selector(self.measurements_selector) is not None

           ):
            measurement_pts = self.measurement_type_selector(self.measurements_selector)
            terrain_height = self.get_elevation(self.long_zone + self.lat_zone, measurement_pts)
            measurement_height = measurement_pts[:,2]
            height_diff = measurement_height - terrain_height

            for i in range(0,len(measurement_pts)):
                wbt = whitebox.WhiteboxTools()
                wbt.set_working_dir(self.OUTPUT_DATA_PATH.absolute().as_posix())
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
            storing_file_path = self.OUTPUT_DATA_PATH.joinpath('topography.asc') 

            f = open(storing_file_path, 'w')
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

        if self.flags['output_path_set'] and self.flags['measurements_added']: 
            pts = self.measurement_type_selector(self.measurements_selector)
            

            pts_dict=[]
            for i,pt in enumerate(pts)  :
                pts_dict.append({'Name': "MP_" + str(i), 'E': pt[0], 'N': pt[1]})
                pts_df = pd.DataFrame(pts_dict)
                pts_df['geometry'] = pts_df.apply(lambda x: Point((float(x.E), float(x.N))), axis=1)
                pts_df = geopandas.GeoDataFrame(pts_df, geometry='geometry')
                pts_df.crs= "+init=epsg:" + self.epsg_code

                file_name_str = 'measurement_pt_' + str(i + 1) + '.shp'
                file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str) 
                pts_df.to_file(file_path.absolute().as_posix(), driver='ESRI Shapefile')

                pts_dict=[]
            self.flags['measurements_exported'] = True


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
            self.elevation_angle_layer = np.copy(self.elevation_angle_array)
            self.elevation_angle_layer[np.where((self.elevation_angle_layer <= self.MAX_ELEVATION_ANGLE))] = 1
            self.elevation_angle_layer[np.where((self.elevation_angle_layer > self.MAX_ELEVATION_ANGLE))] = 0
        else:
            print('No beams coordinated generated, run self.gerate_beam_coords_mesh(str) first!')    

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

        Notes
        --------
        The measurement points must exists and mesh generated 
        before calling this method.

        """
        # measurement point selector:
        
        if ('points_type' in kwargs and 
            kwargs['points_type'] in self.POINTS_TYPE and 
            kwargs['points_type'] in self.measurements_dictionary
            ):

            measurement_pts = self.measurement_type_selector(kwargs['points_type'])
            self.measurements_selector = kwargs['points_type']

            if measurement_pts is not None:
                try:

                    array_shape = (measurement_pts.shape[0], )  + self.mesh_utm.shape
                    self.beam_coords = np.empty(array_shape, dtype=float)

                    for i, pts in enumerate(measurement_pts):
                        self.beam_coords[i] = self.generate_beam_coords(self.mesh_utm, pts)
                    self.flags['beam_coords_generated'] = True

                    # splitting beam coords to three arrays 
                    nrows, ncols = self.x.shape
                    self.azimuth_angle_array = self.beam_coords[:,:,0].T.reshape(nrows,ncols,len(measurement_pts))   
                    self.elevation_angle_array = self.beam_coords[:,:,1].T.reshape(nrows,ncols,len(measurement_pts))
                    self.range_array = self.beam_coords[:,:,2].T.reshape(nrows,ncols,len(measurement_pts))                
                except:
                    print('Something went wrong! Check measurement points')
            else:
                print('Instance in self.measurements_dictionary for type'+ self.measurements_selector + ' is empty!')
                print('Aborting the beam steering coordinates generation for the mesh points!')
        else:
            print('Either points_type was not provided or for the provided points_type there is no instance in self.measurements_dictionary!')
            print('Aborting the beam steering coordinates generation for the mesh points!')



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
            return np.asarray(self.measurements_dictionary['initial'].values[:, 1:].tolist())
        elif points_type == 'optimized':
            return np.asarray(self.measurements_dictionary['optimized'].values[:, 1:].tolist())
        elif points_type == 'reachable':
            return np.asarray(self.measurements_dictionary['reachable'].values[:, 1:].tolist())
        elif points_type == 'identified':
            return np.asarray(self.measurements_dictionary['identified'].values[:, 1:].tolist())
        elif points_type == 'misc':
            return np.asarray(self.measurements_dictionary['misc'].values[:, 1:].tolist())
        else:
            return None

    def layer_selector(self, layer_type):
        """
        Selects GIS layer according to the provided type.

        Parameters
        ----------
        layer_type : str
            A string indicating which layer to be returned

        Returns
        -------
        layer : ndarray
            Depending on the input type this method returns one
            of the following GIS layers:
            Orography
            Landcover
            Canopy height
            Topography
            Restriction zones
            Elevation angle constrained
            Range restriction
            LOS blockage 
            Combined 
            Intersecting angle constrained
            Second lidar placement
            Aerial image
            Misc layer
        Notes
        -----
        This method is used during the generation of the beam steering coordinates.
        """        

        if layer_type == 'orography':
            return self.orography_layer
        elif layer_type == 'landcover':
            return self.landcover_layer
        elif layer_type == 'canopy_height':
            return self.canopy_height_layer
        elif layer_type == 'topography':
            return self.topography_layer
        elif layer_type == 'restriction_zones':
            return self.restriction_zones_layer
        elif layer_type == 'elevation_angle_contrained':
            return self.elevation_angle_layer
        elif layer_type == 'range_contrained':
            return self.range_layer
        elif layer_type == 'los_blockage':
            return self.los_blck_layer
        elif layer_type == 'combined':
            return self.combined_layer
        elif layer_type == 'intersecting_angle_contrained':
            return self.intersecting_angle_layer
        elif layer_type == 'second_lidar_placement':
            return self.second_lidar_layer        
        elif layer_type == 'misc':
            return self.misc_layer
        else:
            return None            



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
            if self.flags['orography_layer_generated']:
                if self.flags['landcover_layers_generated']:
                    self.topography_layer = self.canopy_height_layer + self.orography_layer
                    self.flags['topography_layer_generated'] = True
                else:
                    self.topography_layer = self.orography_layer
                    self.flags['topography_layer_generated'] = True
                    print('Canopy height missing!')
                    print('Topography layer generated using only orography layer!')
            else:
                print('Cannot generate topography layer following layers are missing:')
                print('Orography height')
                self.flags['topography_layer_generated'] = False


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
            self.orography_layer = self.mesh_utm[:,2].reshape(nrows, ncols)
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
        if self.flags['landcover_path_set']:
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
            file_path = self.OUTPUT_DATA_PATH.joinpath('landcover_cropped_utm.tif') 
            im = gdal.Open(file_path.absolute().as_posix())
            band = im.GetRasterBand(1)
            land_cover_array = np.flip(band.ReadAsArray(),axis=0)
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
            if self.flags['landcover_path_set']:
                if self.flags['output_path_set']:               
                    input_image = gdal.Open(self.LANDCOVER_DATA_PATH.absolute().as_posix(), gdal.GA_ReadOnly)
                    storing_file_path = self.OUTPUT_DATA_PATH.joinpath('landcover_cropped_utm.tif') 

                    gdal.Warp(storing_file_path.absolute().as_posix(), 
                                input_image,format = 'GTiff',
                                outputBounds=[self.mesh_corners_utm[0,0], self.mesh_corners_utm[0,1],
                                            self.mesh_corners_utm[1,0], self.mesh_corners_utm[1,1]],
                                dstSRS='EPSG:'+self.epsg_code, 
                                width=int(1 + 2 * self.MESH_EXTENT / self.MESH_RES), 
                                height=int(1 + 2 * self.MESH_EXTENT / self.MESH_RES))

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
                    print('Wrong dimensions!\nLidar position is described by 3 parameters:\n(1)Easting\n(2)Northing\n(3)Height!')
                    print('Lidar position was not added!')
                    return False
        else:
            print('Input is not numpy array!')
            print('Lidar position was not added!')
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
    def generate_beam_coords(lidar_pos, meas_pt_pos, opt=1):
        """
        Generates beam steering coordinates in spherical coordinate system from multiple lidar positions to a single measurement point and vice verse.

        Parameters
        ----------
        opt : int
            opt = 0 -> from single lidar pos to multi points
            opt = 1 -> from multiple lidar pos to single point
        lidar_pos : ndarray
            nD array containing data with `float` or `int` type
            corresponding to x, y and z coordinates of multiple lidar positions.
            nD array data are expressed in meters.
        meas_pt_pos : ndarray
            3D array containing data with `float` or `int` type
            corresponding to x, y and z coordinates of a measurement point.
            3D array data are expressed in meters.
        """
        if opt == 1:
            # testing if  lidar pos has single or multiple positions
            if len(lidar_pos.shape) == 2:
                x_array = lidar_pos[:, 0]
                y_array = lidar_pos[:, 1]
                z_array = lidar_pos[:, 2]
            else:
                x_array = np.array([lidar_pos[0]])
                y_array = np.array([lidar_pos[1]])
                z_array = np.array([lidar_pos[2]])


            # calculating difference between lidar_pos and meas_pt_pos coordiantes
            dif_xyz = np.array([x_array - meas_pt_pos[0], y_array - meas_pt_pos[1], z_array - meas_pt_pos[2]])    

            # distance between lidar and measurement point in space
            distance_3D = np.sum(dif_xyz**2,axis=0)**(1./2)

            # distance between lidar and measurement point in a horizontal plane
            distance_2D = np.sum(np.abs([dif_xyz[0],dif_xyz[1]])**2,axis=0)**(1./2)

            # in radians
            azimuth = np.arctan2(meas_pt_pos[0] - x_array, meas_pt_pos[1] - y_array)
            # conversion to metrological convention
            azimuth = (360 + azimuth * (180 / np.pi)) % 360

            # in radians
            elevation = np.arccos(distance_2D / distance_3D)
            # conversion to metrological convention
            elevation = np.sign(meas_pt_pos[2] - z_array) * (elevation * (180 / np.pi))

            return np.transpose(np.array([azimuth, elevation, distance_3D]))
        else:
            if len(meas_pt_pos.shape) == 2:
                x_array = meas_pt_pos[:, 0]
                y_array = meas_pt_pos[:, 1]
                z_array = meas_pt_pos[:, 2]
            else:
                x_array = np.array([meas_pt_pos[0]])
                y_array = np.array([meas_pt_pos[1]])
                z_array = np.array([meas_pt_pos[2]])

            # calculating difference between lidar_pos and meas_pt_pos coordiantes
            dif_xyz = np.array([lidar_pos[0] - x_array, lidar_pos[1] - y_array, lidar_pos[2] - z_array])    

            # distance between lidar and measurement point in space
            distance_3D = np.sum(dif_xyz**2,axis=0)**(1./2)

            # distance between lidar and measurement point in a horizontal plane
            distance_2D = np.sum(np.abs([dif_xyz[0],dif_xyz[1]])**2,axis=0)**(1./2)

            # in radians
            azimuth = np.arctan2(x_array-lidar_pos[0], y_array-lidar_pos[1])
            # conversion to metrological convention
            azimuth = (360 + azimuth * (180 / np.pi)) % 360

            # in radians
            elevation = np.arccos(distance_2D / distance_3D)
            # conversion to metrological convention
            elevation = np.sign(z_array - lidar_pos[2]) * (elevation * (180 / np.pi))

            return np.transpose(np.array([azimuth, elevation, distance_3D]))        

    def export_yaml(self, **kwargs):
        lidar_dict_sub = dict((k, self.lidar_dictionary[k]) for k in kwargs['lidar_ids'] if k in self.lidar_dictionary)
        # building header of YAML
        yaml_file = self.__yaml_template['skeleton']
        yaml_file = yaml_file.replace('insertCOORDSYS', 'UTM')
        yaml_file = yaml_file.replace('insertUTMzone',  self.long_zone + self.lat_zone)
        yaml_file = yaml_file.replace('insertEPSGcode',  self.epsg_code)
        yaml_file = yaml_file.replace('insertNOlidars',  str(len(lidar_dict_sub)))
        yaml_file = yaml_file.replace('insertMOscenarios',  '1')
    
    
        # building lidar part of YAML
        template_str = self.__yaml_template['lidar']
        lidar_long_str = ""
        lidar_ids = list(lidar_dict_sub.keys())
        for lidar in lidar_ids:
            lidar_str = template_str.replace('insertLIDARid', lidar)
            lidar_pos = lidar_dict_sub[lidar]['position']
            lidar_str = lidar_str.replace('insertLIDAReasting', str(lidar_pos[0]) )
            lidar_str = lidar_str.replace('insertLIDARnorthing', str(lidar_pos[1]) )  
            lidar_str = lidar_str.replace('insertLIDARheight', str(lidar_pos[2]) )
            lidar_long_str = lidar_long_str + lidar_str
        
        yaml_file = yaml_file.replace('insertLIDARdetails', lidar_long_str)
        
        # building scenario part of YAML
        scenario_yaml = self.__yaml_template['scenario']
        scenario_yaml = scenario_yaml.replace('insertScenarioID', 'step-stare scenario')
        scenario_yaml = scenario_yaml.replace('insertNOtransects', '1')
        scenario_yaml = scenario_yaml.replace('insertLIDARids', str(list(lidar_dict_sub.keys())))
        scenario_yaml = scenario_yaml.replace('insertSYNC', '1')
        scenario_yaml = scenario_yaml.replace('insertSCANtype', 'step-stare')
        scenario_yaml = scenario_yaml.replace('insertFFTsize', str(self.FFT_SIZE))
        scenario_yaml = scenario_yaml.replace('insertPULSElenght', str(self.PULSE_LENGTH))
        scenario_yaml = scenario_yaml.replace('insertACCtime', str(self.ACCUMULATION_TIME))    
        scenario_yaml = scenario_yaml.replace('insertMINrange', str(self.MIN_RANGE))
        scenario_yaml = scenario_yaml.replace('insertMAXrange', str(self.MAX_RANGE))   
        scenario_yaml = scenario_yaml.replace('max_no_of_gates', str(self.MAX_NO_OF_GATES))
        yaml_file = yaml_file.replace('insertMEASUREMENTscenarios', scenario_yaml)
        
        # building transect part of YAML
        transect_yaml = self.__yaml_template['transect']
        points_str = ""
        
        points = lidar_dict_sub[lidar_ids[0]]['trajectory'].values
        timing = lidar_dict_sub[lidar_ids[0]]['motion_config'].values[:,-1]
        
        for i, point in enumerate(points):
            points_yaml = self.__yaml_template['points']
            points_yaml = points_yaml.replace('insertPTid', str(int(point[0])))
            points_yaml = points_yaml.replace('insertPTeasting', str(point[1]))
            points_yaml = points_yaml.replace('insertPTnorthing', str(point[2]))
            points_yaml = points_yaml.replace('insertPTheight', str(point[3]))
            points_yaml = points_yaml.replace('insertPTtiming', str(timing[i]))
            points_str = points_str + points_yaml
        transect_yaml = transect_yaml.replace('insertTransectPoints', points_str)
        
        yaml_file = yaml_file.replace('insertTransects', transect_yaml)
        
        file_name_str = "measurement_scenario.yaml"
        file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str) 
        output_file = open(file_path,"w+")
        output_file.write(yaml_file)
        output_file.close()


        xml_file =  self.yaml2xml(file_path)

        file_name_str = "measurement_scenario.xml"
        file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str) 
        output_file = open(file_path,"w+")
        output_file.write(xml_file)
        output_file.close()

    def export_measurement_scenario(self, **kwargs):
        if ('lidar_ids' in kwargs and set(kwargs['lidar_ids']).issubset(self.lidar_dictionary)):

            for lidar in kwargs['lidar_ids']:
                self.export_measurement_scenario_single(lidar_id = lidar)
        
            self.export_yaml(**kwargs)
        else:
            print('One or more lidar ids don\'t exist in the lidar dictionary')
            print('Available lidar ids: ' + str(list(self.lidar_dictionary.keys())))

    def export_measurement_scenario_single(self, **kwargs):
        if ('lidar_id' in kwargs):
            if (kwargs['lidar_id'] in self.lidar_dictionary):
                self.export_motion_config(**kwargs)
                self.export_range_gate(**kwargs)
            else:
                print('Lidar instance \'' + kwargs['lidar_id'] + '\' does not exist in the lidar dictionary!')
                print('Aborting the operation!')
        else:
            print('lidar_id not provided as a keyword argument')
            print('Aborting the operation!')

    def export_motion_config(self, **kwargs):
        # needs to check if output data folder exists!!!!
        if ('lidar_id' in kwargs and kwargs['lidar_id'] in self.lidar_dictionary):
            if self.lidar_dictionary[kwargs['lidar_id']]['motion_config'] is not None:
                motion_config = self.lidar_dictionary[kwargs['lidar_id']]['motion_config']
                motion_program = self.__pmc_template['skeleton']
                in_loop_str = ""

                for i,row in enumerate(motion_config.values):
                    new_pt = self.__pmc_template['motion'].replace("insertMotionTime", str(row[-1]))                    
                    new_pt = new_pt.replace("insertHalfMotionTime", str(row[-1]/2))
                    new_pt = new_pt.replace("insertAzimuth", str(row[1]))
                    new_pt = new_pt.replace("insertElevation", str(row[2])) 
                    
                    in_loop_str = in_loop_str + new_pt
                
                    if i == 0:
                        motion_program = motion_program.replace("1st_azimuth", str(row[1]))
                        motion_program = motion_program.replace("1st_elevation", str(row[2]))
    
                motion_program = motion_program.replace("insertMeasurements", in_loop_str)
                if self.ACCUMULATION_TIME % 100 == 0:
                    if (self.PULSE_LENGTH in [100, 200, 400]):
                        if self.PULSE_LENGTH == 400:
                            PRF = 10 # in kHz
                        elif self.PULSE_LENGTH == 200:
                            PRF = 20
                        elif self.PULSE_LENGTH == 100:
                            PRF = 40
        
                        no_pulses = PRF * self.ACCUMULATION_TIME
                        motion_program = motion_program.replace("insertAccTime", 
                                                                str(self.ACCUMULATION_TIME))
                        motion_program = motion_program.replace("insertTriggers", str(no_pulses))
                        motion_program = motion_program.replace("insertPRF", str(PRF))

                        file_name_str = kwargs['lidar_id'] + "_motion.pmc"
                        file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str) 
                        output_file = open(file_path,"w+")
                        output_file.write(motion_program)
                        output_file.close()
                    else:
                        print('Not allowed pulse lenght!')
                        print('Aborting the operation!')
                else:
                    print('Not allowed accomulation time! It must be a multiple of 100 ms')
                    print('Aborting the operation!')

            else:
                print('something')
                print('Aborting the operation!')
        else:
            print('something')
            print('Aborting the operation!')

    def export_range_gate(self, **kwargs):
        if ('lidar_id' in kwargs and kwargs['lidar_id'] in self.lidar_dictionary):
            if self.lidar_dictionary[kwargs['lidar_id']]['motion_config'] is not None:
                if len(self.lidar_dictionary[kwargs['lidar_id']]['motion_config']) == len(self.lidar_dictionary[kwargs['lidar_id']]['probing_coordinates']):
                    if self.ACCUMULATION_TIME % 100 == 0: 
                        if (self.PULSE_LENGTH in [100, 200, 400]):
                            if self.PULSE_LENGTH == 400:
                                lidar_mode = 'Long'
                            elif self.PULSE_LENGTH == 200:
                                lidar_mode = 'Middle'
                            elif self.PULSE_LENGTH == 100:
                                lidar_mode = 'Short'
                            
                            # selecting range gates from the probing coordinates key 
                            # which are stored in last column and converting them to int
                            range_gates = self.lidar_dictionary[kwargs['lidar_id']]['probing_coordinates'].values[:,3].astype(int)

                            range_gates.sort()
                            range_gates = range_gates.tolist()
                            no_los = len(range_gates)

                            no_used_ranges = len(range_gates)
                            no_remain_ranges = self.MAX_NO_OF_GATES - no_used_ranges
                            prequal_range_gates = np.linspace(self.MIN_RANGE, min(range_gates) , int(no_remain_ranges/2)).astype(int).tolist()
                            sequal_range_gates = np.linspace(max(range_gates) + self.MIN_RANGE, self.MAX_RANGE, int(no_remain_ranges/2)).astype(int).tolist()
                            range_gates = prequal_range_gates + range_gates + sequal_range_gates

                            range_gate_file =  self.generate_range_gate_file(self.__rg_template, no_los, range_gates, lidar_mode, self.FFT_SIZE, self.ACCUMULATION_TIME)

                            file_name_str = kwargs['lidar_id'] + "_range_gates.txt"
                            file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str) 
                            output_file = open(file_path,"w+")
                            output_file.write(range_gate_file)
                            output_file.close()
                        else:
                            print('Not allowed pulse lenght!')
                            print('Aborting the operation!')
                    else:
                        print('Not allowed accumulation time! It must be a multiple of 100 ms')
                        print('Aborting the operation!')
                else:
                    print('Probing coordinates and motion config are misaligned !')
                    print('Aborting the operation!')
            else:
                print('something')
                print('Aborting the operation!')
        else:
            print('something')
            print('Aborting the operation!')

    def export_kml(self, **kwargs):

        # first check if lidar_ids exist in kwargs
        # and in lidar_dictionary
        if('lidar_ids' in kwargs and 
            set(kwargs['lidar_ids']).issubset(self.lidar_dictionary)
            ):

            kml = simplekml.Kml()

            lidar_pos_utm = [self.lidar_dictionary[lidar]['position'] for lidar in kwargs['lidar_ids']]
            lidar_pos_utm = np.asarray(lidar_pos_utm)
            lidar_pos_geo = self.utm2geo(lidar_pos_utm, self.long_zone, self.hemisphere)

            for i,lidar in enumerate(kwargs['lidar_ids']):
                kml.newpoint(name = lidar, 
                            coords=[(lidar_pos_geo[i][1], 
                                    lidar_pos_geo[i][0], 
                                    lidar_pos_geo[i][2])],
                            altitudemode = simplekml.AltitudeMode.absolute)

            trajectories = [self.lidar_dictionary[lidar]['trajectory'].values 
                            for lidar in kwargs['lidar_ids']]
            trajectories = np.asarray(trajectories)
            trajectories_lengths = [len(single) for single in trajectories]

            if trajectories_lengths[1:] == trajectories_lengths[:-1]:
                if np.all(np.all(trajectories == trajectories[0,:], axis = 0)):
                    trajectory_geo = self.utm2geo(trajectories[0][:,1:], self.long_zone, self.hemisphere)
                    
                    trajectory_geo[:, 0], trajectory_geo[:, 1] = trajectory_geo[:, 1], trajectory_geo[:, 0].copy()
                    
                    trajectory_geo_tuple = [tuple(l) for l in trajectory_geo]
                    
                    ls = kml.newlinestring(name="Trajectory")
                    ls.coords = trajectory_geo_tuple
                    ls.altitudemode = simplekml.AltitudeMode.absolute
                    ls.style.linestyle.width = 4
                    ls.style.linestyle.color = simplekml.Color.green
                    
                    for i, pt_coords in enumerate(trajectory_geo_tuple):
                        pt = kml.newpoint(name = 'pt_' + str(i + 1))
                        pt.coords = [pt_coords]
                        pt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
                        pt.altitudemode = simplekml.AltitudeMode.absolute
                        
                        
                    for layer in kwargs['layers']:                    
                        self.export_layer(layer_type = layer)
                        file_name_str = layer + '.tif'
    #                    file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str)
                        
                        map_center = np.mean(self.mesh_corners_geo, axis = 0)
                        ground = kml.newgroundoverlay(name = layer)

                        ground.icon.href = file_name_str
                        ground.latlonbox.north = self.mesh_corners_geo[1,0]
                        ground.latlonbox.south = self.mesh_corners_geo[0,0]
                        ground.latlonbox.east = self.mesh_corners_geo[1,1]
                        ground.latlonbox.west = self.mesh_corners_geo[0,1]
                        ground.color="7Dffffff"

                        ground.lookat.latitude = map_center[0]
                        ground.lookat.longitude = map_center[1]
                        ground.lookat.range = 200
                        ground.lookat.heading = 0
                        ground.lookat.tilt = 0                    
                    
                    file_name_str = "campaign_design.kml"
                    file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str)
                    kml.save(file_path.absolute().as_posix())
                else:
                    print('Trajectories are not the same')
            else:
                print('Trajectories are not the same') 

    def export_layer(self, **kwargs):
        # need to check if folder exists
        if ('layer_type' in kwargs and 
            kwargs['layer_type'] in self.LAYER_TYPE and 
            self.layer_selector(kwargs['layer_type']) is not None
            ):
            layer = self.layer_selector(kwargs['layer_type'])
            
            if len(layer.shape) > 2:
                layer = np.sum(layer, axis = 2)
    
            array_rescaled = (255.0 / layer.max() * (layer - layer.min())).astype(np.uint8)
            array_rescaled = np.flip(array_rescaled, axis = 0)
            image = Image.fromarray(np.uint8(plt.cm.RdBu_r(array_rescaled)*255))
    
            multi_band_array = np.array(image)
            
            rows = multi_band_array.shape[0]
            cols = multi_band_array.shape[1]
            bands = multi_band_array.shape[2]
            file_name_str = kwargs['layer_type'] + '.tif'
            file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str)
            dst_filename = file_path.absolute().as_posix()
            
            x_pixels = rows  # number of pixels in x
            y_pixels = cols  # number of pixels in y
            driver = gdal.GetDriverByName('GTiff')
            options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
            dataset = driver.Create(dst_filename,x_pixels, y_pixels, bands,gdal.GDT_Byte,options = options)
            
            origin_x = self.mesh_corners_utm[0][0]
            origin_y = self.mesh_corners_utm[1][1]
            pixel_width = self.MESH_RES
            geotrans = (origin_x, pixel_width, 0, origin_y, 0, -pixel_width)
    
            proj = osr.SpatialReference()
            proj.ImportFromEPSG(int(self.epsg_code))
            proj = proj.ExportToWkt()
    
            for band in range(bands):
                dataset.GetRasterBand(band + 1).WriteArray(multi_band_array[:,:,band])
    
    
            dataset.SetGeoTransform(geotrans)
            dataset.SetProjection(proj)
            dataset.FlushCache()
            dataset=None
            
            self.resize_tiff(dst_filename, self.ZOOM)
            del_folder_content(self.OUTPUT_DATA_PATH, self.FILE_EXTENSIONS)
    
                
    @staticmethod        
    def resize_tiff(file_path, resize_value):
        original_file = gdal.Open(file_path)
        single_band = original_file.GetRasterBand(1)
        
        y_pixels = single_band.YSize * resize_value
        x_pixels = single_band.XSize * resize_value
        bands = 4
        
        driver = gdal.GetDriverByName('GTiff')
        options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
        modified_file = driver.Create(file_path,x_pixels, y_pixels,bands,gdal.GDT_Byte,options = options)
        modified_file.SetProjection(original_file.GetProjection())
        geotransform = list(original_file.GetGeoTransform())
        
        geotransform[1] /= resize_value
        geotransform[5] /= resize_value
        modified_file.SetGeoTransform(geotransform)
        
        for i in range(1,bands + 1):
            single_band = original_file.GetRasterBand(i)
            
            data = single_band.ReadAsArray(buf_xsize=x_pixels, buf_ysize=y_pixels)  # Specify a larger buffer size when reading data
            out_band = modified_file.GetRasterBand(i)
            out_band.WriteArray(data)
        
        out_band.FlushCache()
        out_band.ComputeStatistics(False)
    
    @staticmethod
    def yaml2xml(yaml_file_path):
        with open(yaml_file_path, 'r') as stream:
            yaml_file = yaml.safe_load(stream)
        xml_file = parseString(dicttoxml.dicttoxml(yaml_file,attr_type=False))
        xml_file = xml_file.toprettyxml()        
        return xml_file