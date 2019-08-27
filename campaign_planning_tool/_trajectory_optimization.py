import numpy as np
import pandas as pd
from random import shuffle

def displacement2time(displacement, Amax, Vmax):
    """
    Calculates minimum move time to perform predefined angular motions.
    
    Parameters
    ----------
    displacement : ndarray
        nD array containing angular displacements 
        that a motion system needs to perform.
        The displacements unit depends on the units of Amax and Vmax.
        Typically the unit is in degrees.
    Amax : float
        Maximum permited acceleration of the motion system.
        Typically the unit for Amax is degrees/s^2 .
    Vmax : float
        Maximum permited velocity of the motion system (aka rated speed).
        Typically the unit for Vmas is degree/s .
    
    Returns
    -------
    move_time : ndarray
        nD array containing minimum move time to perform the diplacements
        The unit of move_time elements depends on the input parameters.
        Typically the unit is s (seconds).

    Notes
    -----
    Equations used to calculate move time are based on [1], 
    assuming an infinite jerk.
    
    References
    ----------
    .. [1] Peters R. D.: Ideal Lift Kinematics: Complete Equations for 
        Plotting Optimum Motion  Elevator Technology 6, 
        Proceedings of ELEVCON’95, 1995.
    """

    move_time = np.empty((len(displacement),), dtype=float)

    # find indexes for which the scanner head 
    # will reach maximum velocity (i.e. rated speed)
    index_a = np.where(displacement > (Vmax**2) / Amax)

    # find indexes for which the scanner head 
    # will not reach maximum velocity (i.e. rated speed)
    index_b = np.where(displacement <= (Vmax**2) / Amax)

    move_time[index_a] = displacement[index_a] / Vmax + Vmax / Amax
    move_time[index_b] = 2 * np.sqrt(displacement[index_b] / Amax)


    return move_time

class OptimizeTrajectory():
    """
    A class containing methods for optimizing and generating lidar trajectories.

    Methods
    ------
    optimize_trajectory(self, lidar_ids, **kwargs)
        Finding a shortest trajectory through the set of measurement points.
    generate_trajectory(lidar_pos, trajectory)
        Generates step-stare trajectory based on the lidar position and 
        trajectory points.
    """


    def __sync_trajectory(self, lidar_ids):
        """
        Syncs trajectories of selected lidars.
        
        Parameters
        ----------
        lidar_ids : list of strings
            A list containing lidar ids as strings corresponding
            to the keys in the lidar dictionary.

        
        """
        if set(lidar_ids).issubset(self.lidar_dictionary):

            print('Synchronizing trajectories for lidar instances:' 
                    + str(lidar_ids))                                                 
            sync_time = []
            try:
                for lidar in lidar_ids:
                    motion_table = self.lidar_dictionary[lidar]['motion_config']
                    timing = motion_table.loc[:, 'Move time [ms]'].values
                    sync_time = sync_time + [timing]

                sync_time = np.max(np.asarray(sync_time).T, axis = 1)

                
                for lidar in lidar_ids:
                    self.lidar_dictionary[lidar]['motion_config']['Move time [ms]'] = sync_time
            except:
                print('Number of trajectory points for lidar instances don\'t match!')
                print('Aborting syncing of the trajectories!')

        else: 
            print('One or more lidar ids don\'t exist in the lidar dictionary')
            print('Available lidar ids: ' + str(list(self.lidar_dictionary.keys())))
            print('Aborting syncing of the trajectories!')


                
    @classmethod
    def trajectory2displacement(cls, lidar_pos, trajectory, __rollover = True):
        """
        Converts trajectory described in a Cartesian coordinate system to 
        angular positions of the lidar scanner heads. 
        
        Parameters
        ----------
        lidar_pos : ndarray
            nD array containing the lidar position in a Cartesian 
            coordinate system.
        trajectory : ndarray
            nD array containing trajectory points in a Cartesian 
            coordinateys system.
        __rollover : boolean
            Indicates whether the lidar motion controller has 
            a __rollover capability.
            The default value is set to True.
        
        Returns
        -------
        angles_start : ndarray
            nD array containing the starting position of the scanner head.
        angles_stop : ndarray
            nD array containing the ending position of the scanner head.
        angular_displacement : ndarray
            nD array containing angular displacements that the motion system
            needs to perform when moving from one to another trajectory point.
        
        Notes
        --------
        PMAC contains many registers, particularly position registers, that 
        have the potential to roll over, changing from their maximum positive 
        value to their maximum negative value, or vice versa, in a change of 
        one increment. 
        """
        angles_start = cls.generate_beam_coords(lidar_pos, 
                                              np.roll(trajectory, 1, axis = 0), 
                                              opt = 0)[:, (0,1)]
        angles_stop = cls.generate_beam_coords(lidar_pos, 
                                              np.roll(trajectory, 0, axis = 0), 
                                              opt = 0)[:, (0,1)]
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
        return np.round(angles_start, cls.NO_DIGITS), \
               np.round(angles_stop, cls.NO_DIGITS), \
               np.abs(angular_displacement)


    @classmethod
    def __rollover(cls, point_1, point_2, lidar_position):
        """
        Calculates a minimum angular move between two trajectory points.
        
        Parameters
        ----------
        point_1 : ndarray
            nD array containing the first trajectory point coordinates 
            in the Cartesian coordinate system.
        point_2 : ndarray
            nD array containing the second trajectory point coordinates 
            in the Cartesian coordinate system.
        lidar_position : ndarray
            nD array containing the lidar position in the Cartesian 
            coordinate system.      
        Returns
        -------
        A minimum angular motion around the azimuth and elevation axes.

        Notes
        --------
        This method considers the rollover to be feasible for the motion system.
        """

        angles_1 = cls.generate_beam_coords(lidar_position, point_1, 1)[0]
        angles_2 = cls.generate_beam_coords(lidar_position, point_2, 1)[0]

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
    def __calculate_max_move(cls, point_1, point_2, lidars):
        """
        Considering all provided lidar positions it calculates maximum
        angular displacement they need to perform from one to another point.
        
        Parameters
        ----------
        point_1 : ndarray
            nD array containing the first trajectory point coordinates 
            in the Cartesian coordinate system.
        point_2 : ndarray
            nD array containing the second trajectory point coordinates 
            in the Cartesian coordinate system.
        lidars : ndarray
            nD array containing lidars positions in the Cartesian 
            coordinate system.      

        Returns
        -------
        Maximum angular displacement lidars need to perform.

        """
        
        azimuth_max = max(map(lambda x: abs(cls.__rollover(point_1, 
                                                           point_2, 
                                                           x)[0]), lidars))
        elevation_max = max(map(lambda x: abs(cls.__rollover(point_1, 
                                                             point_2, 
                                                             x)[1]), lidars))

        return max(azimuth_max,elevation_max)

    def __tsp(self,  lidar_ids, points, start=None):
        """
        Solving a travelening salesman problem for a set of points and 
        two lidar positions.
        
        Parameters
        ----------
        start : int
            Presetting the trajectory starting point.
            A default value is set to None.
        lidar_ids : list of str
            A list of strings containing lidar ids.
        points : list
            A list of coordinates of measurement points.
        
        Returns
        -------
        trajectory : ndarray
            An ordered nD array containing optimized trajectory points.
        
        See also
        --------
        self.generate_trajectory : generation of trajectories

        Notes
        --------
        The optimization of the trajectory is performed through the adaptation
        of the Nearest Neighbor Heuristics solution for the traveling salesman
        problem [1,2]. 

        References
        ----------
        .. [1] Nikola Vasiljevic, Andrea Vignaroli, Andreas Bechmann and 
            Rozenn Wagner: Digitalization of scanning lidar measurement
            campaign planning, WES, 2019.
        .. [2] Reinelt, G.: The Traveling Salesman: Computational Solutions 
            for TSP Applications, Springer-Verlag, Berlin, Heidelberg, 1994.

        Examples
        --------
        """
        if (len(points) > 0 and len(lidar_ids) > 0):
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
            for lidar in lidar_ids:
                lidars = lidars + [self.lidar_dictionary[lidar]['position']]
            while unvisited_points:
                last_point = trajectory[-1]
                max_angular_displacements = []

                # calculates maximum angular move from the last
                # trajectory point to any other point which is not
                # a part of the trajectory            
                for next_point in unvisited_points:
                    max_displacement = self.__calculate_max_move(
                                                                last_point, 
                                                                next_point, 
                                                                lidars
                                                                )
                    max_angular_displacements = (max_angular_displacements 
                                                    + [max_displacement])

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

    def generate_trajectory(self, lidar_pos, trajectory):
        """
        Generates step-stare trajectory based on the lidar position and 
        trajectory points.
        
        Parameters
        ----------
        lidar_pos : ndarray
            nD array containing the lidar position in a Cartesian 
            coordinate system.
        trajectory : ndarray
            nD array containing trajectory points in a Cartesian 
            coordinate system.
        
        Returns
        -------
        motion_table : pd dataframe
            Pandas dataframe containing beam steering angles and motion time.
        """

        _, angles_stop, angular_displacement =  self.trajectory2displacement(
                                                                    lidar_pos, 
                                                                    trajectory)


        move_time = displacement2time(np.max(angular_displacement, axis = 1),
                                      self.MAX_ACCELERATION, 
                                      self.MAX_VELOCITY)


        timing = np.ceil(move_time * 1000)

        matrix = np.array([angles_stop[:,0],
                           angles_stop[:,1],
                           timing]).T

        motion_table = pd.DataFrame(matrix, columns = ["Azimuth [deg]", 
                                                       "Elevation [deg]", 
                                                       "Move time [ms]"])
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

    def optimize_trajectory(self, lidar_ids, **kwargs):
        """
        Finding a shortest trajectory through the set of measurement points.
        
        Parameters
        ----------
        lidar_ids : list of str
            A list of strings containing lidar ids.
        
        Keyword Arguments
        ------------------
        sync : bool, optional
            Indicates whether to sync trajectories or not.
        only_common_points : bool, optional
            Indicates whether to make trajectories only through common
            reachable points.
        
        Returns
        -------
        self.trajectory : ndarray
            An ordered nD array containing trajectory points.
        
        See also
        --------
        self.generate_trajectory : generation of synchronized trajectories

        Notes
        --------
        The optimization of the trajectory is performed by applying the adapted 
        traveling salesman problem to the measurement point set while varing the
        starting point of the trajecRtory. This secures the shortest trajectory. 

        References
        ----------
        .. [1] Nikola Vasiljevic, Andrea Vignaroli, Andreas Bechmann and 
            Rozenn Wagner: Digitalization of scanning lidar measurement
            campaign planning, WES, 2019.

        """        
        # selecting points which will be used for optimization

        if (set(lidar_ids).issubset(self.lidar_dictionary) and
            all([self.lidar_dictionary[lidar]['points_id'] 
                for lidar in lidar_ids])
                ):
            points_id = self.lidar_dictionary[lidar_ids[0]]['points_id']
            measurement_pts = self.points_selector(points_id)

            if 'only_common_points' in kwargs and kwargs['only_common_points']:                     
                common_points = np.prod(np.array([
                self.lidar_dictionary[lidar]['reachable_points'] 
                                for lidar in lidar_ids]        
                                        ), axis=0)
                common_points_indexes = np.where(common_points > 0)
                measurement_pts = measurement_pts[common_points_indexes]

            if len(measurement_pts) > 0:
                self.points_id = points_id
                sync_time_list = []
                for i in range(0,len(measurement_pts)):
                    trajectory = self.__tsp(lidar_ids,
                                            measurement_pts.tolist(), 
                                            i)
                    # needs to record each lidar timing for each move
                    # and then if we want to keep them in syn
                    sync_time = []
                    for lidar in lidar_ids:

                        motion_table = self.generate_trajectory(
                                self.lidar_dictionary[lidar]['position'], 
                                trajectory)

                        timing = motion_table.loc[:, 'Move time [ms]'].values
                        sync_time = sync_time + [timing]
                    sync_time = np.sum(
                                    np.max(
                                        np.asarray(sync_time).T, 
                                        axis = 1)
                                        )
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
                min_traj_ind = np.where(
                        sync_time_list == np.min(sync_time_list))[0][0]
                        
                trajectory = self.__tsp(lidar_ids,
                                        measurement_pts.tolist(),
                                        min_traj_ind)

                trajectory = pd.DataFrame(trajectory, columns = [
                                                        "Easting [m]", 
                                                        "Northing [m]", 
                                                        "Height asl [m]"
                                                                ]
                                            )

                trajectory.insert(loc=0, 
                        column='Point no.', 
                        value=np.array(range(1,len(trajectory) + 1))
                                    )
                self.trajectory = trajectory
                self.flags['trajectory_optimized'] = True   
            
                print('Lidar instances:' 
                    + str(lidar_ids) 
                    + ' will be updated with the optimized trajectory')
                for lidar in lidar_ids:
                    self.update_lidar_instance(lidar_id = lidar, 
                                                trajectory = trajectory)

                if 'sync' in kwargs and kwargs['sync']:
                    self.__sync_trajectory(lidar_ids)      
            else:
                print('No measurement points!')
                print('Aborting the trajectory optimization!')                      
        else: 
            print('One or more lidar ids don\'t \
                    exist in the lidar dictionary')
            print('Available lidar ids: ' 
                    + str(list(self.lidar_dictionary.keys())))
            print('Aborting the trajectory optimization!')