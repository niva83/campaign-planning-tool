class Exporters:
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
            
            self.resize_tiff(dst_filename, self.__ZOOM)
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