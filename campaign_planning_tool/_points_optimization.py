import numpy as np
from itertools import combinations, product

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

    # This last bit is optional if you're okay with "C"
    # being a structured array...
    # C = C.view(A.dtype).reshape(-1, ncols)
    if len(D) == 0:
        return np.array([])

    D = D.view(A.dtype).reshape(-1, ncols)
    return D


class OptimizeMeasurements():
    """
    A class used for optimizing measurement positions.
    
    Methods
    -------
    optimize_measurements(points_id)
        Optimizes measurement positions by solving disc covering problem.    
    """
    def __generate_disc_matrix(self, points_id):
        """
        Generates mid points between any combination of 
        two measurement points which act as disc centers. 

        Parameters
        ----------
        points_id : str
            A string indicating which measurement points to be
            used as the input for the optimization.

        Returns
        -------
        discs : ndarray
            An array of mid points between all combinations of two 
            measurement points.
        matrix : ndarray
            A binary matrix indicating which measurement points are 
            covered by each disc.
            The matrix shape is (len(discs), len(measurements_initial)).
            The value 1 indicates that a point is covered a disc.
            The value 0 indicates that a point is not covered by a disc.
            The matrix is sorted in decending order, having a row with a maximum 
            number of 1 (i.e. covered discs) positioned at the top.

        See also
        --------
        optimize_measurements(points_id) : implementation of disc covering problem

        Notes
        --------
        generate_disc_matrix() method is used to generate necessary inputs for 
        the greedy implementation of the disc covering problem which optimizes 
        the measurement points for the field campaign. It is required that 
        measurement points are added to the measurement dictionary before 
        calling this method.

        References
        ----------
        .. [1] Ahmad Biniaz, Paul Liu, Anil Maheshwari and Michiel Smid, 
           Approximation algorithms for the unit disk cover problem in 2D and 3D,
           Computational Geometry, Volume 60, Pages 8-18, 2017,
           https://doi.org/10.1016/j.comgeo.2016.04.002.

        """
        if points_id in self.POINTS_ID:
            measurement_pts = self.points_selector(points_id)
            self.points_id = points_id
    
            if measurement_pts is not None:
                # find discs in 2D (horizontal plane!)
                points_list = list(measurement_pts[:,(0,1)])
                points_combination_list = list(combinations(points_list, 2))
                points_combination_array = np.asarray(points_combination_list)
                 
                discs = (points_combination_array[:,0] 
                         + points_combination_array[:,1]) / 2

                discs_pts_list = list(product(list(discs), points_list))
                discs_pts_array = np.asarray(discs_pts_list)
                # calculate distances between points and disc centers
                distances =  np.linalg.norm(discs_pts_array[:,0] 
                                            - discs_pts_array[:,1], 
                                            axis = 1)
                # converts calculated distances to 0 or 1 depending on REP_RAD
                distances = np.where(distances <= self.REP_RADIUS, 1, 0)
                # converts a flat array to matrix 
                matrix = np.asarray(np.split(distances,len(discs)))

                # removing discs which cover same points
                unique_discs = np.unique(matrix, return_index= True, axis = 0)
                matrix = unique_discs[0]
                discs = discs[unique_discs[1]]

                # remove discs which cover only one point
                # because it is better to use the existing point
                # instead of a disc which center probably does 
                # not coincides with the point location
                ind = np.where(np.sum(matrix,axis = 1) > 1)
                matrix = matrix[ind]
                discs = discs[ind]

                total_covered_points = np.sum(matrix,axis = 1)

                # sorts matrix in the descending order
                # discs that cover max no of points located on top
                matrix = matrix[(-1*total_covered_points).argsort()]
                discs = discs[(-1*total_covered_points).argsort()]

                # adding 0 m for elevation of each disc so 
                # disc positions match shape of measurement points
                discs = np.append(discs.T, 
                                  np.array([np.zeros(len(discs))]),axis=0).T
                return discs, matrix
            else:
                print("No measurement points -> nothing to optimize!")
                return None, None
        else:
            print("There is no instance in the measurement point \
                   dictionary for the given points_id!")
            return None, None

    @staticmethod
    def __find_unique_indexes(matrix):
        """
        Finds which discs cover at unique points and which don't.
        
        Parameters
        ----------
        matrix : ndarray
            A binary matrix indicating which measurement points are 
            covered by each disc.
            The matrix shape is (len(discs), len(measurements_initial)).
            The value 1 indicates that a point is covered a disc.
            The value 0 indicates that a point is not covered by a disc.
            The matrix is sorted in decending order, having a row with a maximum 
            number of 1 (i.e. covered discs) positioned at the top.
        
        Returns
        -------
        unique_indexes : list 
            A list of integers containing indexes of discs 
        none_unique_indexes : list 
        """

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
    def __minimize_discs(cls, matrix,disc):
        """
        Minimizes number of discs which cover set of points
        
        Parameters
        ----------
        discs : ndarray
            An array of mid points between all combinations of two 
            measurement points.
        matrix : ndarray
            A binary matrix indicating which measurement points are 
            covered by each disc.
            The matrix shape is (len(discs), len(measurements_initial)).
            The value 1 indicates that a point is covered a disc.
            The value 0 indicates that a point is not covered by a disc.
            The matrix is sorted in decending order, having a row with a maximum 
            number of 1 (i.e. covered discs) positioned at the top.
        
        Returns
        -------
        A minimum subset of discs which cover points.
        
        Notes
        --------
        Discs which cover only one point are removed.
        """
        unique_indexes, none_unique_indexes = cls.__find_unique_indexes(matrix)
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

    def optimize_measurements(self, points_id):
        """
        Optimizes measurement positions by solving disc covering problem.
        
        Parameters
        ----------
        points_id : str
            A string indicating which measurement points to be
            used as the input for the optimization.

        Returns
        -------
        Optimized positions will be stored in the measurement dictionary
        as 'optimized' instance.
        
        See also
        --------
        generate_disc_matrix() : method which calculates this method inputs

        Notes
        --------
        A greedy implementation of the disc covering problem for 
        a set of measurement points.

        References
        ----------
        .. [1] Ahmad Biniaz, Paul Liu, Anil Maheshwari and Michiel Smid, 
           Approximation algorithms for the unit disk cover problem in 2D and 3D,
           Computational Geometry, Volume 60, Pages 8-18, 2017,
           https://doi.org/10.1016/j.comgeo.2016.04.002.
        """
        if points_id in self.POINTS_ID:
            measurement_pts = self.points_selector(points_id)
            self.points_id = points_id

            if measurement_pts is not None:
                height_total = measurement_pts[:,2]
                height_terrain = self.get_elevation(self.long_zone 
                                                    + self.lat_zone, 
                                                    measurement_pts)
                measurement_height = abs(height_total - height_terrain)
                measurement_height_mean = np.average(measurement_height)


                print('Optimizing ' 
                        + self.points_id 
                        + ' measurement points!')
                discs, matrix = self.__generate_disc_matrix(points_id)

                points_uncovered = measurement_pts
                points_covered_total = np.zeros((0,3), measurement_pts.dtype)

                i = 0
                j = len(points_uncovered)
                disc_indexes = []
                while i <= (len(discs) - 1) and j > 0 :
                    indexes = np.where(matrix[i] == 1 )
                    # matrix = matrix * (1 - matrix[i])
                    points_covered = measurement_pts[indexes]
                    points_new = array_difference(points_covered, 
                                                    points_covered_total)

                    if len(points_new) > 0:
                        points_covered_total = np.append(
                                                    points_covered_total, 
                                                    points_new,
                                                    axis=0)

                        disc_indexes = disc_indexes + [i]
                    points_uncovered = array_difference(
                                                        points_uncovered, 
                                                        points_covered)
                    i += 1
                    j = len(points_uncovered)

                # makes subset of discs and matrix
                discs_selected = discs[disc_indexes]
                matrix_selected = matrix[disc_indexes]

                # minimize number of discs
                if len(discs_selected) > 1:
                    discs_selected = self.__minimize_discs(
                                                        matrix_selected,
                                                        discs_selected)
                self.disc_temp = discs_selected
                # if we don't cover all the points remaining 
                # uncovered points must be added to the array
                self.uncovered = points_uncovered
                self.covered = points_covered_total
                if len(points_uncovered) > 0:
                    measurements_optimized = np.append(discs_selected, 
                                                        points_uncovered, 
                                                        axis = 0)
                    terrain_height = self.get_elevation(self.long_zone 
                                                        + self.lat_zone, 
                                                measurements_optimized)
                    measurements_optimized[:, 2] = (terrain_height 
                                                + measurement_height_mean)
                    self.add_measurement_instances('optimized',
                                                    measurements_optimized)

                # if we cover all the points then
                # the optimized measurements are
                # are equal to the disc centers
                else:
                    measurements_optimized = discs_selected
                    terrain_height = self.get_elevation(self.long_zone
                                                        + self.lat_zone, 
                                                measurements_optimized)
                    measurements_optimized[:, 2] = (terrain_height 
                                                + measurement_height_mean)
                    self.add_measurement_instances('optimized',
                                                    measurements_optimized)

                # in case when none of the measurement
                # points are covered by this method than
                # the optimized points should be equal to
                # the original measurements points
                # if len(measurements_optimized) == len(measurement_pts):
                #     self.add_measurement_instances(points = measurement_pts, points_id = 'optimized')
                    
            else:
                print("No measurement positions added to \
                        measurenet dictionary")
                print('Aborting the operation!')
        else:
            print(points_id + "does not exist in the measurement \
                    point dictionary!")
            print('Aborting the operation!')
