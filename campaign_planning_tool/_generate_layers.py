import numpy as np
import pandas as pd
import geopandas
from shapely.geometry import Point
import whitebox

import srtm
from pyproj import Proj

# Instead of GDAL
import rasterio
import rasterio.warp
from rasterio.transform import from_origin
from fiona.crs import from_epsg

# for working with files and folders
from pathlib import Path
import tempfile
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

class LayersGIS():
    """
    Class containing methods for performing geometrical calculationas 
    and generation of various GIS layer.

    Methods
    ------
    generate_mesh(points_id)
        Generates a rectangular horizontal mesh containing equally spaced points.
    find_mesh_points_index(point)
        For a given input point returns indexes of the closest point
        in the generated mesh.
    generate_beam_coords(lidar_pos, meas_pt_pos, opt):
        Generates beam steering coordinates in spherical coordinate system from 
        multiple lidar positions to a single measurement point and vice verse.
    generate_first_lidar_placement_layer(points_id)
        Generates the first lidar placement layer which is used 
        for the positioning of the first lidars.
    generate_additional_lidar_placement_layer(lidar_id)
        Generates the additional lidar placement layer which is used
        for the positioning of the consecutive lidar(s).
    lidar_position_suggestion(layer_id, treshold)
        Returns ndarray of lidar positions for which number of reachable points 
        is equal or bigger than the given treshold value.    
    get_elevation(utm_zone, points_utm)
        Fetch elevation from the SRTM database for 
        a number of points described by in the UTM coordinates.
    layer_selector(layer_id)
        Selects GIS layer according to the provided type.
    set_utm_zone(utm_zone)
        Sets EPSG code, latitudinal and longitudinal zones to the CPT instance. 
    utm2geo(points_utm, long_zone, hemisphere)
        Converts an array of points in the UTM coord system to
        an array of point in the GEO coord system.
    """

    @staticmethod
    def generate_beam_coords(lidar_pos, meas_pt_pos, opt=1):
        """
        Generates beam steering coordinates in spherical coordinate system from 
        multiple lidar positions to a single measurement point and vice verse.

        Parameters
        ----------
        opt : int
            opt = 0 -> from single lidar pos to many measurement points
            opt = 1 -> from many lidars to single measurement point
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
            dif_xyz = np.array([x_array - meas_pt_pos[0], 
                                y_array - meas_pt_pos[1], 
                                z_array - meas_pt_pos[2]])    

            # distance between lidar and measurement point in space
            distance_3D = np.sum(dif_xyz**2,axis=0)**(1./2)

            # distance between lidar and measurement point in a horizontal plane
            distance_2D = np.sum(np.abs([dif_xyz[0],dif_xyz[1]])**2,axis=0)**(1./2)

            # in radians
            azimuth = np.arctan2(meas_pt_pos[0] - x_array, 
                                 meas_pt_pos[1] - y_array)
            # conversion to metrological convention
            azimuth = (360 + azimuth * (180 / np.pi)) % 360

            # in radians
            elevation = np.arccos(distance_2D / distance_3D)
            # conversion to metrological convention
            elevation = (np.sign(meas_pt_pos[2] - z_array) 
                         * (elevation * (180 / np.pi)))

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
            dif_xyz = np.array([lidar_pos[0] - x_array, 
                                lidar_pos[1] - y_array, 
                                lidar_pos[2] - z_array])    

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
            elevation = (np.sign(z_array - lidar_pos[2]) 
                         * (elevation * (180 / np.pi)))

            return np.transpose(np.array([azimuth, elevation, distance_3D]))   

    @staticmethod  
    def __check_utm_zone(utm_zone):
        """
        Checks whether UTM grid zone is valid or not.

        Parameters
        ----------
        utm_zone : str
            A string representing an UTM grid zone, containing digits (1 to 60) 
            indicating the longitudinal zone followed by a character (from 'C' 
            to 'X' excluding 'O') corresponding to the latitudinal zone.
        
        Returns
        -------
        out : bool
            A boolean indicating True or False.        
        """ 
        flag = False
        lat_zones = ['C','D','E','F','G','H',
                     'J','K','L','M','N','P',
                     'Q','R','S','T','U','V','W','X']
        try:
            lat_zone = utm_zone[-1].upper() # in case users put lower case 
            long_zone = int(utm_zone[:-1])
            if lat_zone in lat_zones:
                print('Correct latitudinal zone!')
                flag = True
            else:
                print('Incorrect latitudinal zone!\
                       \nEnter a correct latitudinal zone!')
                flag = False
            
            if long_zone >= 1 and long_zone <= 60:
                print('Correct longitudinal zone!')
                flag = True and flag
            else:
                print('Incorrect longitudinal zone!\
                      \nEnter a correct longitudinal zone!')
                flag = False
        except:
            flag = False
            print('Wrong input!\nHint: there should not be spaces' 
                  + 'between longitudinal and latitudinal zones'
                  + 'when expressing the UTM zone!')
        return flag

    @staticmethod
    def __which_hemisphere(utm_zone):
        """
        Returns whether UTM grid zone belongs to the Northern 
        or Southern hemisphere. 

        Parameters
        ----------
        utm_zone : str
            A string representing an UTM grid zone, containing digits (1 to 60) 
            indicating the longitudinal zone followed by a character (from 'C' 
            to 'X' excluding 'O') corresponding to the latitudinal zone.
        
        Returns
        -------
        out : str
            A string indicating North or South hemisphere.

        """
 
        lat_zones = ['C','D','E','F','G','H',
                     'J','K','L','M','N','P',
                     'Q','R','S','T','U','V','W','X']

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
    def __utm2epsg(utm_zone):
        """
        Converts UTM grid zone to EPSG code.

        Parameters
        ----------
        utm_zone : str
            A string representing an UTM grid zone, containing digits (1 to 60) 
            indicating the longitudinal zone followed by a character (from 'C' 
            to 'X' excluding 'O') corresponding to the latitudinal zone.
        Returns
        -------
        out : str
            A string containing EPSG code.
        
        Examples
        --------
        If UTM grid zone exists:
        >>> __utm2epsg('31V') 
        '32631'

        If UTM grid zone doesn't exist:
        >>> __utm2epsg('61Z')
        None
        """
        lat_zones = ['C','D','E','F','G','H',
                     'J','K','L','M','N','P',
                     'Q','R','S','T','U','V','W','X']
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

    def set_utm_zone(self, utm_zone):
        """
        Sets EPSG code, latitudinal and longitudinal zones to the CPT instance. 
        
        Parameters
        ----------
        utm_zone : str
            A string representing an UTM grid zone, containing digits (1 to 60) 
            indicating the longitudinal zone followed by a character (from 'C' 
            to 'X' excluding 'O') corresponding to the latitudinal zone.
        """
        if self.__check_utm_zone(utm_zone):
            self.long_zone = utm_zone[:-1]
            self.lat_zone = utm_zone[-1].upper() 
            self.epsg_code = self.__utm2epsg(utm_zone)
            self.hemisphere = self.__which_hemisphere(utm_zone) 
            self.flags['utm_set'] = True
            return print('UTM zone set')
        else:
            return print('UTM zone not set')

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
        """
        geo_projection = Proj("+proj=utm +zone=" 
                              + long_zone 
                              + " +" + hemisphere 
                              + " +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

        points_geo = reversed(geo_projection(points_utm[:,0], 
                                             points_utm[:,1],inverse=True))
        points_geo = np.array(list(points_geo))
        points_geo = np.append(points_geo, 
                               np.array([points_utm[:,2]]),
                               axis = 0).transpose()
        return points_geo

    @classmethod
    def get_elevation(cls, utm_zone, points_utm):
        """
        Fetch elevation from the SRTM database for 
        a number of points described by in the UTM coordinates.

        Parameters
        ----------
        utm_zone : str
            A string representing an UTM grid zone, containing digits (1 to 60) 
            indicating the longitudinal zone followed by a character (from 'C' 
            to 'X' excluding 'O') corresponding to the latitudinal zone.
        points_utm : ndarray
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
        information are obtained from the SRTM DEM database [1].

        References
        ----------
        .. [1] https://www2.jpl.nasa.gov/srtm/cbanddataproducts.html

        See also
        --------
        self.utm2geo(points_utm, long_zone, hemisphere) : converts utm to geo
        """             
        if cls.__check_utm_zone(utm_zone):
            hemisphere = cls.__which_hemisphere(utm_zone)
            long_zone = utm_zone[:-1]
            points_geo = cls.utm2geo(points_utm, long_zone, hemisphere)
            elevation_data = srtm.get_data()

            # this works if multiple points are provided but fails if single one is given
            elevation = np.asarray([elevation_data.get_elevation(pt[0],pt[1]) 
            if (elevation_data.get_elevation(pt[0],pt[1]) != None) 
                and (elevation_data.get_elevation(pt[0],pt[1]) != np.nan)
                else 0 for pt in points_geo])
            elevation[np.isnan(elevation)] = cls.NO_DATA_VALUE
            return elevation
        else:
           return None

    def layer_selector(self, layer_id):
        """
        Selects GIS layer according to the provided type.

        Parameters
        ----------
        layer_id : str
            A string indicating which layer to be returned.

        Returns
        -------
        layer : ndarray
            Depending on the input type this method returns GIS layers.

        See als
        -------
        self.LAYER_ID
        """        

        if layer_id == 'orography':
            self.legend_label = 'Height asl [m]'
            return self.orography_layer
        elif layer_id == 'landcover':
            self.legend_label = 'CLC code'
            return self.landcover_layer
        elif layer_id == 'canopy_height':
            return self.canopy_height_layer
            self.legend_label = 'Height agl [m]'
        elif layer_id == 'topography':
            self.legend_label = 'Height asl [m]'
            return self.topography_layer
        elif layer_id == 'restriction_zones':
            self.legend_label = '0 - forbiden, 1 - allowed'
            return self.restriction_zones_layer
        elif layer_id == 'elevation_angle_contrained':
            self.legend_label = 'Number of reseachable points []'
            return self.elevation_angle_layer
        elif layer_id == 'range_contrained':
            self.legend_label = 'Number of reseachable points []'
            return self.range_layer
        elif layer_id == 'los_blockage':
            self.legend_label = 'Number of reseachable points []'
            return self.los_blck_layer
        elif layer_id == 'first_lidar_placement':
            self.legend_label = 'Number of reseachable points []'
            return self.first_lidar_layer
        elif layer_id == 'intersecting_angle_contrained':
            self.legend_label = 'Number of reseachable points []'
            return self.intersecting_angle_layer
        elif layer_id == 'additional_lidar_placement':
            self.legend_label = 'Number of reseachable points []'
            return self.additional_lidar_layer        
        elif layer_id == 'misc':
            self.legend_label = '???'
            return self.misc_layer
        else:
            return None            

    def __update_layer_dict(self, layer_id, 
                                  points_id = None,
                                  lidars_id = None, 
                                  message = '' ):
        """
        Updates layer dictionary with information about layer creation.
        
        Parameters
        ----------
            layer_id : str
                String indicating which layer info is being updated.
            points_id : str
                String indicating points used during the layer generation.
            lidars_id : str
                String indicating lidar(s) used during the layer generation.
            message : str
                String storing message related to the layer generation.        
        """        
        dict_input = {layer_id:{
                                'points_id': points_id,     
                                'lidars_id': lidars_id,
                                'message' : message
                                }}
        self.layers_info.update(dict_input)

    def generate_mesh(self, points_id):
        """
        Generates a rectangular horizontal mesh containing equally spaced points.
        
        Parameters
        ----------
        points_id : str
            A string indicating which measurement points to be
            used as the input for the mesh generation.
        
        Returns
        -------
        self.mesh_corners : ndarray
            ndarray containing lower left and upper right corner of the mesh
        self.mesh : ndarray
            ndarray of mesh points

        Notes
        --------
        In case mesh center is not provided, but initial measurement points are
        this method will find the barycenter of measurement points and consider 
        it as the mesh center. If mesh extent is not provided, a default value 
        of 5000 meters will be considered.      
        """

        rules = [
                 points_id in self.POINTS_ID,
                 points_id in self.measurements_dictionary
                 ]
        
        if all(rules):
            measurement_pts = self.points_selector(points_id)

            if len(measurement_pts) > 0:
                self.mesh_center = np.int_(np.mean(measurement_pts,axis = 0))
                self.flags['mesh_center_added'] = True
            else:
                print('Mesh center missing!')
                print('Mesh cannot be generated!')
                self.flags['mesh_center_added'] = False

        if self.flags['mesh_center_added']:
            # securing that the input parameters are int 
            self.mesh_center = np.int_(self.mesh_center)
            self.MESH_EXTENT = int(
                                   int(self.MESH_EXTENT / self.MESH_RES) 
                                   * self.MESH_RES)

            self.mesh_corners_utm = np.array([self.mesh_center - self.MESH_EXTENT, 
                                            self.mesh_center + self.MESH_EXTENT])

            self.__corners = np.asarray([
                self.mesh_center + [ self.MESH_EXTENT, self.MESH_EXTENT,0],
                self.mesh_center + [ self.MESH_EXTENT,-self.MESH_EXTENT,0],
                self.mesh_center + [-self.MESH_EXTENT, self.MESH_EXTENT,0],
                self.mesh_center + [-self.MESH_EXTENT,-self.MESH_EXTENT,0]])

            self.mesh_corners_geo = self.utm2geo(self.__corners, 
                                                 self.long_zone, 
                                                 self.hemisphere)                                 

            # self.mesh_corners_geo = self.utm2geo(self.mesh_corners_utm, 
            #                                      self.long_zone, 
            #                                      self.hemisphere)                                 


            self.x, self.y = np.meshgrid(
                    np.arange(self.mesh_corners_utm[0][0], 
                              self.mesh_corners_utm[1][0] + self.MESH_RES, 
                              self.MESH_RES),
                    np.arange(self.mesh_corners_utm[0][1], 
                              self.mesh_corners_utm[1][1] + self.MESH_RES, 
                              self.MESH_RES)
                                        )
            self.z = np.full(self.x.shape, self.mesh_center[2])		

            nrows, ncols = self.x.shape

            self.mesh_utm = np.array([self.x.T, 
                                      self.y.T, 
                                      self.z.T]).T.reshape(-1, 3)
            self.mesh_geo = self.utm2geo(self.mesh_utm, 
                                         self.long_zone, 
                                         self.hemisphere)
            self.mesh_indexes = np.array(
                            range(0,len(self.mesh_utm),1)).reshape(nrows,ncols)
            self.mesh_coordinates = self.mesh_utm.reshape(nrows,ncols,3)
            self.flags['mesh_generated'] = True


    def find_mesh_point_index(self, point):
        """
        For a given input point returns indexes of the closest point
        in the generated mesh.

        Parameters
        ----------
        point : ndarray
            nD array containing coordinates of the test point

        Returns
        -------
        i : int
            Index i of the closest matching point in the mesh.
            Index i correspond to the x axis.
        j : int 
            Index j of the closest matching point in the mesh.
            Index j correspond to the y axis.
        """
        dist_2D = np.sum((self.mesh_utm[:,(0,1)] - point[:2])**2, axis=1)
        index = np.argmin(dist_2D)

        i, j = np.array(np.where(self.mesh_indexes == index)).flatten()
        
        return i, j

    def __generate_beam_coords_mesh(self, points_id):
        """
        Generates beam steering coordinates from every mesh point 
        to every measurement point.

        Parameters
        ----------
        points_id : str
            A string indicating which measurement points to be
            used for the beam steering coordinates calculation.

        Notes
        --------
        The measurement points must exists and mesh generated 
        before calling this method.

        """
        rules = [
                 points_id in self.POINTS_ID,
                 points_id in self.measurements_dictionary
                 ]
        
        if all(rules):
            measurement_pts = self.points_selector(points_id)
            self.points_id = points_id
            array_shape = (measurement_pts.shape[0], ) + self.mesh_utm.shape
            self.beam_coords = np.empty(array_shape, dtype=float)

            for i, pts in enumerate(measurement_pts):
                self.beam_coords[i] = self.generate_beam_coords(self.mesh_utm, 
                                                                pts)
            self.flags['beam_coords_generated'] = True

            # splitting beam coords to three arrays 
            nrows, ncols = self.x.shape
            self.azimuth_angle_array = self.beam_coords[:,:,0].T.reshape(nrows,
                                                    ncols,len(measurement_pts))   
            self.elevation_angle_array = self.beam_coords[:,:,1].T.reshape(nrows,
                                                    ncols,len(measurement_pts))
            self.range_array = self.beam_coords[:,:,2].T.reshape(nrows,ncols,
                                                          len(measurement_pts))
        else:
            print('One of the following conditions was not satisfied:')
            print('\tpoints_id not in self.POINTS_ID')
            print('\tempty key for given points_id in measurement dictionary')
      

    def __generate_elevation_restriction_layer(self, points_id):
        """
        Generates elevation restricted GIS layer.

        Paremeters
        ----------
        points_id : str
            Id indicating which points are used to 
            generate range restriction layer.        

        Notes
        -----
        The beams coordinates for every mesh point must be
        generated (self.__generate_beam_coords_mesh()) before
        calling this method.
        """        
        if self.flags['beam_coords_generated'] == True:
            self.elevation_angle_layer = np.copy(self.elevation_angle_array)
            good_indexes = np.where((self.elevation_angle_layer 
                                        <= self.MAX_ELEVATION_ANGLE))
            bad_indexes = np.where((self.elevation_angle_layer 
                                        > self.MAX_ELEVATION_ANGLE))                                        
            self.elevation_angle_layer[good_indexes] = 1
            self.elevation_angle_layer[bad_indexes] = 0
            self.__update_layer_dict('elevation_angle_contrained',
                                      points_id)
        else:
            print('No beams coordinated generated!'
                  + '\nRun self.gerate_beam_coords_mesh(str) first!'
                  + 'Aborting the operation!')    

    def __generate_range_restriction_layer(self, points_id):
        """
        Generates range restricted GIS layer.

        Paremeters
        ----------
        points_id : str
            Id indicating which points are used to 
            generate range restriction layer.

        Notes
        -----
        The beams coordinates for every mesh point must be
        generated (self.__generate_beam_coords_mesh()) before
        calling this method.
        """
        if self.flags['beam_coords_generated'] == True:
            self.range_layer = np.copy(self.range_array)
            good_indexes = np.where((self.range_layer <= self.AVERAGE_RANGE))
            bad_indexes = np.where((self.range_layer > self.AVERAGE_RANGE))
            self.range_layer[good_indexes] = 1
            self.range_layer[bad_indexes] = 0
            self.__update_layer_dict('range_contrained',points_id)                      
        else:
            print('No beams coordinated generated!\n Run self.gerate_beam_coords_mesh() first!')

    def __generate_orography_layer(self):
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
        .. [1] https://github.com/tkrajina/srtm.py

        See also
        --------
        self.generate_mesh(points_id) : mesh generation
        """        
                
        if self.flags['mesh_generated']:        
            nrows, ncols = self.x.shape
            elevation_data = srtm.get_data()

            self.mesh_utm[:,2] = np.asarray([
                elevation_data.get_elevation(x[0],x[1]) 
                if (elevation_data.get_elevation(x[0],x[1]) != None) 
                and (elevation_data.get_elevation(x[0],x[1]) != np.nan) else 0 
                for x in self.mesh_geo])

            self.mesh_geo[:,2] = self.mesh_utm[:,2]
            self.orography_layer = self.mesh_utm[:,2].reshape(nrows, ncols)
            self.flags['orography_layer_generated'] = True
            self.__update_layer_dict('orography')                        
        else:
            print('Mesh not generated -> orography layer cannot be generated ')

    def __import_landcover_data(self):
        """
        Generates landcover layer based on the CORINE landcover data.

        Notes
        --------
        It is necessary that the CORINE landcover data are cropped
        to the area correspoding to the previously generated mesh.

        Currently the method only works with the CORINE data [1].

        References
        ----------
        .. [1] https://land.copernicus.eu/pan-european/corine-land-cover

        See also
        --------
        self.__crop_landcover_data() : cropping landcover data
        self.generate_mesh() : mesh generation
        """         
        if self.flags['landcover_path_set']:
            data = rasterio.open(self.LANDCOVER_DATA_PATH.absolute().as_posix())
            coords_transformed = rasterio.warp.transform(from_epsg(self.epsg_code),
                                                         data.crs.data,
                                                         self.mesh_utm[:,0],
                                                         self.mesh_utm[:,1])
            nrows, ncols = self.x.shape
            coords_transformed = np.asarray(coords_transformed).T


            clc_data = []

            with data as src:
                for val in src.sample(coords_transformed):
                    clc_data = clc_data + list(val)

            self.landcover_layer = np.array(clc_data).reshape(nrows,ncols)

            self.flags['landcover_layer_generated'] = True
        else:
            print('No landcover data provided!')

    def __generate_restriction_zones(self):
        """
        Converts specific CORINE landcover CLC codes
        to the restriction zones layer.

        Notes
        --------
        It is necessary that the base landcover layer is generated.

        Currently the method only works with the CORINE data [1].

        The method converts specific CLC codes[1], corresponding to specific 
        landcover types such as for example water body, to the zones which
        are restricted for the lidar installation. 

        References
        ----------
        .. [1] https://land.copernicus.eu/pan-european/corine-land-cover


        See also
        --------
        self.__crop_landcover_data() : cropping landcover data
        self.__import_landcover_data() : importing cropped landcover data
        """           
        if self.flags['landcover_layer_generated']:        
            self.restriction_zones_layer = np.copy(self.landcover_layer)

            self.restriction_zones_layer[np.where(
                                       self.restriction_zones_layer < 23)] = 1
            self.restriction_zones_layer[np.where(
                                     (self.restriction_zones_layer > 25) 
                                   & (self.restriction_zones_layer < 35))] = 1
            self.restriction_zones_layer[np.where(
                                      self.restriction_zones_layer > 44)] = 1

            self.restriction_zones_layer[np.where(
                                     (self.restriction_zones_layer >= 23) 
                                   & (self.restriction_zones_layer <= 25))] = 0

            self.restriction_zones_layer[np.where(
                                     (self.restriction_zones_layer >= 35) 
                                   & (self.restriction_zones_layer <= 44))] = 0
            self.flags['restriction_zones_generated'] = True
            self.__update_layer_dict('restriction_zones')
        else:
            print('No landcover layer generated!')
            print('Aborting the operation!')

    def __generate_canopy_height(self):
        """
        Converts specific CORINE landcover CLC codes
        to the canopy height layer.

        Notes
        --------
        It is necessary that the base landcover layer is generated.
        Currently the method only works with the CORINE data.

        The method converts specific CLC codes, corresponding to forest, 
        to the canopy height. 
        
        It simply adds 20 m for CLC codes correspoding to forest.

        See also
        --------
        self.__crop_landcover_data() : cropping landcover data
        self.__import_landcover_data() : importing cropped landcover data
        """              

        if self.flags['landcover_layer_generated']:
            self.canopy_height_layer = np.copy(self.landcover_layer)
            self.canopy_height_layer[np.where(
                                            self.canopy_height_layer < 23)] = 0
            self.canopy_height_layer[np.where(
                                          self.canopy_height_layer == 23)] = 20
            self.canopy_height_layer[np.where(
                                          self.canopy_height_layer == 24)] = 20
            self.canopy_height_layer[np.where(
                                          self.canopy_height_layer == 25)] = 20
            self.canopy_height_layer[np.where(
                                           self.canopy_height_layer >  25)] = 0
            self.flags['canopy_height_generated'] = True
            self.__update_layer_dict('canopy_height')
        else:
            print('No landcover layer generated -> canopy height layer not generated!')

    def __generate_landcover_layer(self):
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
        self.__crop_landcover_data() : cropping landcover data
        self.__import_landcover_data() : importing cropped landcover data
        self.__generate_canopy_height() : canopy height generation
        self.__generate_restriction_zones() : restriction zone generation
        """   
        if self.flags['landcover_path_set']:
            try:
                self.__import_landcover_data()
                self.__generate_canopy_height()
                self.__generate_restriction_zones()
                self.flags['landcover_layers_generated'] = True
                self.__update_layer_dict('landcover')
            except:
                print('The landcover data does not exist!')
                print('Aborting the operation!')
                self.flags['landcover_layers_generated'] = False
        else:
            print('Path to landcover data not provided!')
            self.flags['landcover_layers_generated'] = False

    def __generate_topographic_layer(self):
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
        self.__generate_orography_layer() : orography layer generation
        self.__generate_landcover_layer() : landcover layer generation
        """        

        if self.flags['mesh_generated']:
            self.__generate_orography_layer()
            self.__generate_landcover_layer()
            if self.flags['orography_layer_generated']:
                if self.flags['landcover_layers_generated']:
                    self.topography_layer = (self.canopy_height_layer 
                                             + self.orography_layer)
                    self.flags['topography_layer_generated'] = True
                    self.__update_layer_dict('topography',
                                              None,
                                              None,
                                             'orography + canopy')
                else:
                    self.topography_layer = self.orography_layer
                    self.flags['topography_layer_generated'] = True
                    self.__update_layer_dict('topography',
                                              None,
                                              None,
                                             'orography height only')       
                    print('Canopy height missing!')
                    print('Topography layer generated using only orography layer!')
            else:
                print('Cannot generate topography layer following layers are missing:')
                print('Orography height')
                self.flags['topography_layer_generated'] = False

    def __export_topography(self):
        """
        Exports the topography layer as a ASCI shapefile.
        
        Notes
        --------
        The topography layer must be generated and output data
        folder set before calling this method.
        
        The method writes out a ASCI shape file. The shapefile is 
        used in the site viewshed analysis.

        See also
        --------
        add_measurements() : adding measurement points to the CPT class instance 
        __viewshed_analysis() : the site viewshed analysis
        """        
        if (self.__tempfolder.is_dir()
            and self.flags['topography_layer_generated']):
            topography_array = np.flip(self.topography_layer,axis=0)
            storing_file_path = self.__tempfolder.joinpath('topography.asc') 

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

    def __export_measurements(self):
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
        __viewshed_analysis() : the site viewshed analysis
        """

        if self.__tempfolder.is_dir() and self.flags['measurements_added']: 
            pts = self.points_selector(self.points_id)
            

            pts_dict=[]
            for i,pt in enumerate(pts)  :
                pts_dict.append({'Name': "MP_" + str(i), 'E': pt[0], 'N': pt[1]})
                pts_df = pd.DataFrame(pts_dict)
                pts_df['geometry'] = pts_df.apply(lambda x: Point((float(x.E), float(x.N))), axis=1)
                pts_df = geopandas.GeoDataFrame(pts_df, geometry='geometry')
                pts_df.crs = {'init': 'epsg:' + self.epsg_code}
#                pts_df.crs= "+init=epsg:" + self.epsg_code
                

                file_name_str = 'measurement_pt_' + str(i + 1) + '.shp'
                file_path = self.__tempfolder.joinpath(file_name_str) 
                pts_df.to_file(file_path.absolute().as_posix(), 
                               driver='ESRI Shapefile')

                pts_dict=[]
            self.flags['measurements_exported'] = True

    def __viewshed_analysis(self):
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
        __export_topography() : topography layer shapefile exporter
        __export_measurements() : measurement points shapefile exporter
        """
        if (self.flags['topography_exported'] and 
            self.flags['measurements_exported'] and 
            self.points_selector(self.points_id) is not None

           ):
            measurement_pts = self.points_selector(self.points_id)
            terrain_height = self.get_elevation(self.long_zone 
                                              + self.lat_zone, measurement_pts)
            measurement_height = measurement_pts[:,2]
            height_diff = measurement_height - terrain_height

            for i in range(0,len(measurement_pts)):
                wbt = whitebox.WhiteboxTools()
                wbt.set_working_dir(self.__tempfolder.as_posix())
                wbt.verbose = False
                wbt.viewshed("topography.asc",
                             "measurement_pt_" +str(i+1)+".shp",
                             "los_blockage_" +str(i+1)+".asc",
                             height_diff[i])
            self.flags['viewshed_analyzed'] = True

    def __viewshed_processing(self):
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
        __viewshed_analysis() : viewshed analysis of the site
        """        

        if self.flags['viewshed_analyzed']:
            measurement_pts = self.points_selector(self.points_id)
            nrows, ncols = self.x.shape
            no_pts = len(measurement_pts)

            self.los_blck_layer = np.empty((nrows, ncols, no_pts), dtype=float)


            for i in range(0,len(measurement_pts)):
                file_name_str = "los_blockage_" + str(i+1) + ".asc"
                file_path = self.__tempfolder.joinpath(file_name_str) 
                los_blck_tmp  = np.loadtxt(file_path.absolute().as_posix(),
                                           skiprows=6)
                los_blck_tmp  = np.flip(los_blck_tmp, axis = 0)
                self.los_blck_layer[:,:,i] = los_blck_tmp
            
            self.flags['viewshed_performed'] = True

    def __generate_los_blck_layer(self, points_id):
        """
        Generates the los blockage layer by performing 
        viewshed analysis for the selected site.

        Parameters
        ----------
        points_id : str
            A string indicating which measurement points to be
            used as the input for the method.        
        
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
            if points_id in self.POINTS_ID:
                self.points_id = points_id

                if self.points_selector(self.points_id) is not None:
                    tempfolder = tempfile.TemporaryDirectory()
                    self.__tempfolder = Path(tempfolder.name).absolute()
                    self.__export_measurements()
                    self.__export_topography()
                    self.__viewshed_analysis()
                    self.__viewshed_processing()
                    self.flags['los_blck_layer_generated'] = True
                    self.__update_layer_dict('los_blockage', points_id)
#                    tempfolder.cleanup()                             
                else:
                    print('For points type \''
                           + points_id 
                           + '\' there are no measurement points ' 
                           + 'in the measurements dictionary!')
                    print('Aborting the los blockage layer generation!')
            else:                
                print('Points type \''
                      + points_id
                      + '\' does not exist in the measurements dictionary!')
                print('Aborting the los blockage layer generation!')
        else:
            print('Topography layer not generates!')
            print('Aborting the los blockage layer generation!')

    def generate_first_lidar_placement_layer(self, points_id):
        """
        Generates the combined layer which is used
        for the positioning of the first lidars.

        Parameters
        ----------
        points_id : str
            A string indicating which measurement points to be
            used as the input for the method.        
        
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
        rules = [points_id in self.POINTS_ID,
                 points_id in self.measurements_dictionary
                 ]
        
        if all(rules):
            measurement_pts = self.points_selector(points_id)
            self.points_id = points_id
            print('Generating combined layer for ' 
                    + points_id 
                    + ' measurement points!')

            self.generate_mesh(points_id)
            self.__generate_topographic_layer()
            self.__generate_beam_coords_mesh(points_id)
            self.__generate_range_restriction_layer(points_id)
            self.__generate_elevation_restriction_layer(points_id)
            self.__generate_los_blck_layer(points_id)
            self.__update_layer_dict('first_lidar_placement', points_id)

            nrows, ncols = self.x.shape
            if (self.flags['los_blck_layer_generated'] and 
                self.flags['topography_layer_generated']
                ):

                self.first_lidar_layer = (self.elevation_angle_layer 
                                        * self.range_layer 
                                        * self.los_blck_layer)

                if self.flags['landcover_layer_generated']:
                    self.first_lidar_layer = (self.first_lidar_layer 
                    * self.restriction_zones_layer.reshape((nrows,
                                                            ncols,1)))
                    self.flags['first_lidar_layer_generated'] = True
                    self.first_lidar_layer_pts_type = points_id
                    print('First lidar placement layer generated with landcover data!')
                else:
                    self.flags['first_lidar_layer_generated'] = True
                    self.first_lidar_layer_pts_type = points_id
                    print('First lidar placement layer generated without landcover data!')
            else:
                print('Either topography or los blockage layer are missing!')
                print('Aborting the first lidar placement layer generation!')   
        else:
            print('Either points_id was not provided' 
                  +' or for the provided points_id there is ' 
                  + 'no instance in self.measurements_dictionary!')
            print('Aborting the first lidar placement layer generation!')

    def __generate_intersecting_angle_layer(self, 
                                            lidar_id, 
                                            points_id):
        """
        Generates intersecting angle layer.
        
        Parameters
        ----------
        lidar_pos : ndarray
            nD array containing data with `float` or `int` type
            corresponding to x, y and z coordinates of a lidar position.
            nD array data are expressed in meters.
        measurement_pts : ndarray
            nD array containing data with `float` or `int` type
            corresponding to x, y and z coordinates of measurement points. 
            nD array data are expressed in meters.

        """
        lidar_pos = self.lidar_dictionary[lidar_id]['position']
        measurement_pts = self.points_selector(self.points_id)
        nrows, ncols = self.x.shape
        no_pts = len(measurement_pts)
        azimuths_1 = self.generate_beam_coords(lidar_pos,measurement_pts,0)[:,0]
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
        self.__update_layer_dict('intersecting_angle_contrained', 
                                 self.first_lidar_layer_pts_type,
                                 lidar_id
                                 )

    def generate_additional_lidar_placement_layer(self, lidar_id):
        """
        Generates the combined layer which is used
        for the positioning of the first lidars.

        Parameters
        ----------
        lidar_id : str
            A string representing id of the first placed lidar.

        Notes
        --------
        Before calling this method, combined layer must be previously generated.
        For the provided id an instance in the lidar dictionary must exist.
        """
        
        self.points_id = self.first_lidar_layer_pts_type
        measurement_pts = self.points_selector(self.points_id)
        if measurement_pts is not None:
            if lidar_id in self.lidar_dictionary:
                # storing id of the lidar used to generate second lidar placement layer
                self.first_lidar_id = lidar_id
                self.__generate_intersecting_angle_layer(lidar_id,
                                                            self.points_id)
                self.flags['intersecting_angle_layer_generated'] = True
                self.update_lidar_instance(lidar_id = lidar_id, 
                                            layer_id = 'first_lidar_placement')
                self.additional_lidar_layer = (self.first_lidar_layer 
                                            * self.intersecting_angle_layer)
                # reachable_points = self.lidar_dictionary[kwargs['lidar_id']]['reachable_points']
                # self.second_lidar_layer = self.combined_layer * self.intersecting_angle_layer * reachable_points
                self.flags['additional_lidar_layer'] = True
                self.__update_layer_dict('additional_lidar_placement', 
                                        self.first_lidar_layer_pts_type,
                                        lidar_id
                                        )                    
            else:
                print('Lidar does not exist in self.lidar dict, halting operation!')
        else:
            print('No measurement points provided, halting operation!')

    def lidar_position_suggestion(self, layer_id, treshold):
        """
        Returns ndarray of lidar positions for which number of reachable points 
        is equal or bigger than the given treshold value.

        Parameters
        ----------
        layer_id : str
            A string indicating which layer to be used to find lidar positions.
            Typically this is either equal to 'firs_lidar_placement' or
            'additional_lidar_placement'.
        treshold : int
            Treshold indicating a minimum number reachable points that a 
            potential lidar position should provide.
        """
        layer = self.layer_selector(layer_id)
        layer = np.sum(layer, axis = 2)
        indexes = np.where(layer >= treshold)
        lidar_positions = self.mesh_coordinates[indexes]
        return lidar_positions