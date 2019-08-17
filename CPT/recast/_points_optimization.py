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

    # This last bit is optional if you're okay with "C" being a structured array...
    # C = C.view(A.dtype).reshape(-1, ncols)
    if len(D) == 0:
        return np.array([])

    D = D.view(A.dtype).reshape(-1, ncols)
    return D


class OptimizeMeasurements():
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
        points_id : str
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

        """
        if 'points_id' in kwargs and kwargs['points_id'] in self.POINTS_TYPE:
            measurement_pts = self.measurement_type_selector(kwargs['points_id'])
            self.measurements_selector = kwargs['points_id']
    
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
                return discs, matrix
            else:
                print("No measurement points -> nothing to optimize!")
        else:
            print("There is no instance in the measurement point dictionary for the given points_id!")

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
        if 'points_id' in kwargs and kwargs['points_id'] in self.POINTS_TYPE:
            measurement_pts = self.measurement_type_selector(kwargs['points_id'])
            self.measurements_selector = kwargs['points_id']
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
                self.add_measurement_instances(points = measurements_optimized, points_id = 'optimized')

            # if we cover all the points then
            # the optimized measurements are
            # are equal to the disc centers
            else:
                measurements_optimized = discs_selected
                terrain_height = self.get_elevation(self.long_zone + self.lat_zone, measurements_optimized)
                measurements_optimized[:, 2] = terrain_height + np.average(measure_pt_height)
                self.add_measurement_instances(points = measurements_optimized, points_id = 'optimized')

            # in case when none of the measurement
            # points are covered by this method than
            # the optimized points should be equal to
            # the original measurements points
            # if len(measurements_optimized) == len(measurement_pts):
            #     self.add_measurement_instances(points = measurement_pts, points_id = 'optimized')
                
        else:
            print("No measurement positions added, nothing to optimize!")    