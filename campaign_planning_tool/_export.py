import numpy as np

# Instead of GDAL
import rasterio
import rasterio.warp
from rasterio.transform import from_origin
from fiona.crs import from_epsg

import yaml # installation via conda
from xml.dom.minidom import parseString
import dicttoxml
import simplekml
import matplotlib.pyplot as plt
from PIL import Image

from pathlib import Path
import tempfile
import os, shutil

class Export():
    """
    A class containing methods for exporting CPT results.

    Methods
    ------
    export_kml(file_name, **kwargs)
        Exports GIS layers, lidar positions and trajectory in a KML file.
    export_measurement_scenario(lidar_ids)
        Exports measurement scenarios for given lidars.
    """

    __yaml_template = {'skeleton': 
"""# This is the config file for the trajectory generation
# for long-range WindScanners written in YAML
#
#

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
      author: 'CPT'
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
              move_time: insertPTtiming # in ms
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

    def __export_motion_config(self, lidar_id):
        """
        Exports motion config as motion program for PMAC.

        Attributes
        ---------
        lidar_id : str
            A string corresponding to the key (name) in the lidar dictionary.

        Returns
        -------
        True or False depending on the success.

        Notes
        -----
        This method will create motion program to drive scannner head(s) 
        only if previously the corresponding key in the lidar dictionary 
        was updated, therefore containing generated trajectory.

        Currently this method only works for long-range WindScanners!
        """
        motion_config = self.lidar_dictionary[lidar_id]['motion_config']
        if motion_config is not None:
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
                    if self.PULSE_LENGTH == 400: # in ns
                        PRF = 10 # in kHz
                    elif self.PULSE_LENGTH == 200: # in ns
                        PRF = 20
                    elif self.PULSE_LENGTH == 100: # in ns
                        PRF = 40

                    no_pulses = PRF * self.ACCUMULATION_TIME
                    motion_program = motion_program.replace("insertAccTime", 
                                                            str(self.ACCUMULATION_TIME))
                    motion_program = motion_program.replace("insertTriggers", str(no_pulses))
                    motion_program = motion_program.replace("insertPRF", str(PRF))

                    file_name_str = lidar_id + "_motion.pmc"
                    file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str) 
                    output_file = open(file_path,"w+")
                    output_file.write(motion_program)
                    output_file.close()
                    return True
                else:
                    print('Not allowed pulse lenght!')
                    print('Allowed pulse lengths are: 100, 200 and 400 ns!')
                    print('Check self.PULSE_LENGTH constant and set it accordingly!')
                    print('Aborting the operation!')
                    return False
            else:
                print('Not allowed accomulation time! It must be a multiple of 100 ms')
                print('Check self.ACCUMULATION_TIME contstant and set it accordingly!')
                print('Aborting the operation!')
                return False
        else:
            print('Lidar instance \''+ lidar_id +'\' is not updated!')
            print('Call self.update_lidar_instance(**kwargs) before executing this method!')
            print('Aborting the operation!')
            return False

    def __export_range_gate(self, lidar_id):
        """
        Exports laser and FPGA config as a range gate file 
        for WindScanner Client Software (WCS).

        Attributes
        ---------
        lidar_id : str
            A string corresponding to the key (name) in the lidar dictionary.

        Returns
        -------
        True or False depending on the success.

        Notes
        -----
        This method will creates range gate file which is used to 
        set FPGA board, AOM and EDFA of the long-range WindScanner.

        Currently this method only works for long-range WindScanners!
        """
        if self.lidar_dictionary[lidar_id]['motion_config'] is not None:
            if len(self.lidar_dictionary[lidar_id]['motion_config']) == len(self.lidar_dictionary[lidar_id]['probing_coordinates']):
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
                        range_gates = self.lidar_dictionary[lidar_id]['probing_coordinates'].values[:,3].astype(int)

                        range_gates.sort()
                        range_gates = range_gates.tolist()
                        no_los = len(range_gates)

                        no_used_ranges = len(range_gates)
                        no_remain_ranges = self.MAX_NO_OF_GATES - no_used_ranges
                        prequal_range_gates = np.linspace(self.MIN_RANGE, min(range_gates) , int(no_remain_ranges/2)).astype(int).tolist()
                        sequal_range_gates = np.linspace(max(range_gates) + self.MIN_RANGE, self.MAX_RANGE, int(no_remain_ranges/2)).astype(int).tolist()
                        range_gates = prequal_range_gates + range_gates + sequal_range_gates

                        range_gate_file =  self.__generate_range_gate_file(self.__rg_template, no_los, range_gates, lidar_mode, self.FFT_SIZE, self.ACCUMULATION_TIME)

                        file_name_str = lidar_id + "_range_gates.txt"
                        file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str) 
                        output_file = open(file_path,"w+")
                        output_file.write(range_gate_file)
                        output_file.close()
                        return True
                    else:
                        print('Not allowed pulse length!')
                        print('Allowed pulse lengths are: 100, 200 and 400 ns!')
                        print('Check self.PULSE_LENGTH constant and set it accordingly!')
                        print('Aborting the operation!')
                        return False
                else:
                    print('Not allowed accumulation time! It must be a multiple of 100 ms')
                    print('Check self.ACCUMULATION_TIME constant and set it accordingly!')
                    print('Aborting the operation!')
                    return False
            else:
                print('Probing coordinates and motion config are misaligned!')
                print('Aborting the operation!')
                return False
        else:
            print('Lidar instance \''+ lidar_id +'\' is not updated!')
            print('Call self.update_lidar_instance(**kwargs) before executing this method!')
            print('Aborting the operation!')
            return False

    def __export_yaml_xml(self, lidar_ids):
        """
        Exports derived CPT results as YAML and XML files.

        Attributes
        ---------
        lidar_ids : list of strings
            A list containing lidar ids as strings corresponding
            to keys in the lidar dictionary       
        """

        lidar_dict_sub = dict((k, self.lidar_dictionary[k]) for k in lidar_ids if k in self.lidar_dictionary)

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
        
        # Building transect part of YAML
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
        
        # Exporting to yaml file
        file_name_str = "measurement_scenario.yaml"
        file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str) 
        output_file = open(file_path,"w+")
        output_file.write(yaml_file)
        output_file.close()

        # Reading the exported yaml file and converting it to an xml file
        xml_file =  self.__yaml2xml(file_path)
        file_name_str = "measurement_scenario.xml"
        file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str) 
        output_file = open(file_path,"w+")
        output_file.write(xml_file)
        output_file.close()

    def export_measurement_scenario(self, lidar_ids):
        """
        Exports measurement scenario as config files for scanning lidar(s).

        Parameters
        ---------
        lidar_ids : list of strings
            A list containing lidar ids as strings corresponding
            to the keys in the lidar dictionary.

        Notes
        -----
        This method will create following files:
            (1) Motion program to drive scanner head(s)
            (2) Range gate file to configure laser and FPGA
            (3) YAML and XML files containing info from (1) and (2)
        """
        if self.OUTPUT_DATA_PATH.exists():

            if set(lidar_ids).issubset(self.lidar_dictionary):
                flag = True
                for lidar in lidar_ids:
                    flag_mc = self.__export_motion_config(lidar)
                    flag_rg = self.__export_range_gate(lidar)
                    flag = flag * flag_mc * flag_rg
                if flag:         
                    self.__export_yaml_xml(lidar_ids)
                    print('Measurement scenario export successful!')
                else:
                    print('Measurement scenario export unsuccessful!')
            else:
                print('One or more lidar ids don\'t exist in the lidar dictionary')
                print('Available lidar ids: ' + str(list(self.lidar_dictionary.keys())))
                print('Aborting the current action!')
        else:
            print('Output folder path is not valid!')
            print('Set a proper output folder path!')
            print('Call method set_path(path_str, **kwargs)!')
            print('Aborting the operation!')
    @staticmethod
    def __get_corners_geo(map_center, extent, half_res, epsg_code):

        corner_1 = map_center + [-extent,  extent] + [-half_res, half_res]
        corner_2 = map_center + [ extent,  extent] + [ half_res, half_res]
        corner_3 = map_center + [ extent, -extent] + [ half_res,-half_res]
        corner_4 = map_center + [-extent, -extent] + [-half_res,-half_res]

        corners = np.array([corner_4, 
                            corner_3, 
                            corner_2, 
                            corner_1])

        corners_geo = rasterio.warp.transform(from_epsg(epsg_code),
                                              from_epsg(4326),
                                              corners[:,0],
                                              corners[:,1])

        corners_geo = np.asarray(corners_geo).T
        return  list(map(tuple, corners_geo))  


    def export_kml(self, file_name, **kwargs):
        """
        Depending on the provided **kwargs the method export GIS layers, 
        lidar positions and/or trajectory to a KML file.

        Parameters
        ----------
        file_name : str
            A string indicating the KML file name

        Keyword Arguments
        -----------------
        lidar_ids : list of strings
            A list containing lidar ids as strings corresponding
            to keys in the lidar dictionary
        layer_ids : list of string
            A list of strings corresponding to the GIS layers
        """        
        # missing checking on layer type!
        anything_in = False
        kml = simplekml.Kml()
        if ('layer_ids' in kwargs):
            if set(kwargs['layer_ids']).issubset(self.LAYER_ID):
                for layer in kwargs['layer_ids']:                    
                    self.__export_layer(layer_id = layer)
                    file_name_str = layer + '.tif'

                    map_center = np.mean(self.mesh_corners_geo, axis = 0)
                    corners_geo = self.__get_corners_geo(self.mesh_center[:2], 
                                                         self.MESH_EXTENT, 
                                                         self.MESH_RES/2, 
                                                         self.epsg_code)

                    ground = kml.newgroundoverlay(name = layer)

                    ground.gxlatlonquad.coords = corners_geo
                    ground.icon.href = file_name_str
                    ground.color="7Dffffff"
                    ground.lookat.latitude = map_center[0]
                    ground.lookat.longitude = map_center[1]
                    ground.lookat.range = 25000
                    ground.lookat.heading = 0
                    ground.lookat.tilt = 0
                    anything_in = True           


        if('lidar_ids' in kwargs):
            if set(kwargs['lidar_ids']).issubset(self.lidar_dictionary):
                lidar_pos_utm = [self.lidar_dictionary[lidar]['position'] 
                                 for lidar in kwargs['lidar_ids']]
                lidar_pos_utm = np.asarray(lidar_pos_utm)
                lidar_pos_geo = self.utm2geo(lidar_pos_utm, self.long_zone, self.hemisphere)
                flag_trajectory = True
                for i,lidar in enumerate(kwargs['lidar_ids']):
                    kml.newpoint(name = lidar, 
                                coords=[(lidar_pos_geo[i][1], 
                                        lidar_pos_geo[i][0], 
                                        lidar_pos_geo[i][2])],
                                altitudemode = simplekml.AltitudeMode.absolute)
                    if self.lidar_dictionary[lidar]['trajectory'] is not None:
                        flag_trajectory = flag_trajectory * True
                    else:
                        flag_trajectory = flag_trajectory * False

                anything_in = True
                if flag_trajectory:
                    trajectories = [self.lidar_dictionary[lidar]['trajectory'].values 
                                    for lidar in kwargs['lidar_ids']]
                    trajectories = np.asarray(trajectories)
                    trajectories_lengths = [len(single) for single in trajectories]

                    if (trajectories_lengths[1:] == trajectories_lengths[:-1] 
                        and trajectories_lengths[0] is not None):
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
                        else:
                            print('Trajectories are not the same for the provided lidar_ids!')
                            print('Trajectories will not be saved in KML!')                    
                else:
                    print('Trajectories are not the same for the provided lidar_ids!')
                    print('Trajectories will not be saved in KML!')
            else:
                print('One or more lidar ids don\'t exist in the lidar dictionary')
                print('Available lidar ids: ' + str(list(self.lidar_dictionary.keys())))
                print('Lidar positions will not be saved in KML!')                                    
                                    
              
        if anything_in:                       
            file_path = self.OUTPUT_DATA_PATH.joinpath(file_name + '.kml')
            kml.save(file_path.absolute().as_posix())
            print('KML exported successful!')

    def __export_layer(self, layer_id):
        """
        Exports GIS layers as GeoTIFF.

        Attributes
        ---------
        layer_id : list of string
            A list of strings corresponding to the GIS layers

        """
        if layer_id in self.LAYER_ID:
            if self.layer_selector(layer_id) is not None:
                file_name_str = layer_id + '.tif'
                file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str)                
                                
                layer = np.flip(self.layer_selector(layer_id), axis = 0)                
                if len(layer.shape) > 2:
                    layer = np.sum(layer, axis = 2)

                # Resizing array to produce high resolution GeoTIFF
                layer = layer.repeat(self.ZOOM, axis=0).repeat(self.ZOOM, axis=1)
        
                # Creating RGB np array 
                array_rescaled = (255.0 / layer.max() * (layer - layer.min())).astype(np.uint8)
                image = Image.fromarray(np.uint8(plt.cm.RdBu_r(array_rescaled)*255))    
                multi_band_array = np.array(image)
                
                rows = multi_band_array.shape[0]
                cols = multi_band_array.shape[1]
                bands = multi_band_array.shape[2]

                # Removing and adding 1/2 resolution from x and y respectevely
                # in order to properly geolocate GeoTIFF image
                top_x = self.mesh_corners_utm[:,0].min() - self.MESH_RES / 2
                top_y = self.mesh_corners_utm[:,1].max() + self.MESH_RES / 2 

                dst_filename = file_path.absolute().as_posix()
               
                dataset = rasterio.open(
                            dst_filename,
                            'w',
                            driver = 'GTiff',
                            height = rows,
                            width = cols,
                            count = bands, 
                            dtype = multi_band_array.dtype,
                            crs = from_epsg(self.epsg_code),
                            transform = from_origin(top_x, 
                                                    top_y, 
                                                    self.MESH_RES / self.ZOOM, 
                                                    self.MESH_RES / self.ZOOM)
                                        )    
                

                dataset.write(array_rescaled, 1)

                for band in range(bands):
                    dataset.write(multi_band_array[:,:,band], band + 1)
                
                dataset.close()      

            else:
                print('Requested layer is empty!')
                print('Aborting the the operation!')
        else:
            print('Requested layer does not exist!')
            print('Aborting the the operation!')
    
    @staticmethod
    def __yaml2xml(yaml_file_path):
        """
        Converter from YAML to XML format.

        Attributes
        ---------
        yaml_file_path : str
            A path to the YAML file which
            undergoes the conversion to XML.

        """        
        with open(yaml_file_path, 'r') as stream:
            yaml_file = yaml.safe_load(stream)
        xml_file = parseString(dicttoxml.dicttoxml(yaml_file,attr_type=False))
        xml_file = xml_file.toprettyxml()        
        return xml_file    

    @staticmethod
    def __generate_range_gate_file(template_str, no_los, range_gates, lidar_mode, fft_size, accumulation_time):
        """
        Range gate file generator.

        Attributes
        ---------
        template_str : str
            A string containing the base range gate file template.
        no_los : int
            Number of line of sight.
        range_gates : ndarray
            nD array of type int containing range gate center positions.
        lidar_mode : str
            A string indicating the operational mode a scanning lidar.
        fft_size : int
            An integer indicating number of fft points used during
            the spectral analysis of the backscattered signal.
        accumulation_time : int
            An integer indicating the accumulation time of the 
            Doppler spectra.

        Notes
        -----
        This method is only applicabe for  long-range WindScanners!
        """        
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
    