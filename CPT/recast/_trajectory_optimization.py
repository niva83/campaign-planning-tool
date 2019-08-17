import numpy as np
import pandas as pd
from random import shuffle

class OptimizeTrajectory():

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
    def calculate_max_move(cls, point1, point2, lidars):
        azimuth_max = max(map(lambda x: abs(cls.rollover(point1, point2, x)[0]), lidars))
        elevation_max = max(map(lambda x: abs(cls.rollover(point1, point2, x)[1]), lidars))

        return max(azimuth_max,elevation_max)

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
        points_id : str
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
        if ('points_id' in kwargs and
             kwargs['points_id'] in self.POINTS_TYPE and
             kwargs['points_id'] in self.measurements_dictionary and
             len(self.measurements_dictionary[kwargs['points_id']]) > 0
            ):
            if ('lidar_ids' in kwargs and
                 set(kwargs['lidar_ids']).issubset(self.lidar_dictionary)
                ):
                points = self.measurements_dictionary[kwargs['points_id']].values[:, 1:].tolist()

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

    def optimize_trajectory(self, **kwargs):
        """
        Finding a shortest trajectory through the set of measurement points.
        
        Parameters
        ----------
        see kwargs

        
        Keyword paramaters
        ------------------
        points_id : str, required
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
        if ('points_id' in kwargs and
             kwargs['points_id'] in self.POINTS_TYPE and
             kwargs['points_id'] in self.measurements_dictionary and
             len(self.measurements_dictionary[kwargs['points_id']]) > 0
            ):
            if ('lidar_ids' in kwargs and
                 set(kwargs['lidar_ids']).issubset(self.lidar_dictionary)
                ):

                measurement_pts = self.measurements_dictionary[kwargs['points_id']].values[:, 1:].tolist()
                self.measurements_selector = kwargs['points_id']
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
                                            points_id = kwargs['points_id'])

                if 'sync' in kwargs and kwargs['sync']:
                    self.sync_trajectory(**kwargs)                            


            else: 
                print('One or more lidar ids don\'t exist in the lidar dictionary')
                print('Available lidar ids: ' + str(list(self.lidar_dictionary.keys())))

        else:
            print('Either point type id does not exist or for the corresponding measurement dictionary instance there are no points!')

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