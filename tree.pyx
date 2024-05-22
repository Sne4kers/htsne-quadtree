# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as cnp
import cython
from libcpp.vector cimport vector

ctypedef cnp.npy_float64 DTYPE_t          # Type of X
ctypedef cnp.npy_intp SIZE_t              # Type for indices and counters
ctypedef cnp.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef cnp.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cdef extern from "point.hpp":
    cdef struct Point:
        DTYPE_t x
        DTYPE_t y

cdef extern from "cell.hpp":
    cdef struct Cell:
        INT32_t parent_idx

        vector[INT32_t] children_idx
        bint is_leaf

        Point barycenter
        Point center

        Point min_bounds
        Point max_bounds

        SIZE_t cumulative_size
        SIZE_t depth
        DTYPE_t lorentz_factor

cdef extern from "infinity_quad_tree.hpp":
    cdef cppclass InfinityQuadTree:
        InfinityQuadTree() except +
        InfinityQuadTree(vector[Point] points) except +
        vector[Cell] get_nodes()
        size_t approximate_centers_of_mass(double x, double y, double theta_sq, double* combined_results)

cdef extern from "infinity_quad_tree_no_shortcut.hpp":
    cdef cppclass InfinityQuadTreeNoShortcut:
        InfinityQuadTreeNoShortcut() except +
        InfinityQuadTreeNoShortcut(vector[Point] points) except +
        vector[Cell] get_nodes()
        size_t approximate_centers_of_mass(double x, double y, double theta_sq, double* combined_results)

cdef class PyInfinityQuadTree:
    cdef InfinityQuadTree py_tree

    def __cinit__(self, points):
        cdef vector[Point] points_vector
        for i in range(len(points)):
            points_vector.push_back(Point(points[i][0], points[i][1]))
        self.py_tree = InfinityQuadTree(points_vector)

        # print("BUILT!")

        # print(self.py_tree.get_nodes())

    cpdef get_nodes(self):
        return self.py_tree.get_nodes()

    cpdef summarize(self, DTYPE_t[:] query_pt, DTYPE_t[:, :] X, float theta):
        # Used for testing summarize
        cdef:
            DTYPE_t[:] summary
            int n_samples, n_dimensions

        n_samples = X.shape[0]
        n_dimensions = X.shape[1]
        summary = np.zeros(4 * n_samples, dtype=np.float64)

        idx = self.py_tree.approximate_centers_of_mass(query_pt[0], query_pt[1], theta * theta, &summary[0])
        return idx, summary

cdef class PyInfinityQuadTreeNoShortcut:
    cdef InfinityQuadTreeNoShortcut py_tree

    def __cinit__(self, points):
        cdef vector[Point] points_vector
        for i in range(len(points)):
            points_vector.push_back(Point(points[i][0], points[i][1]))
        self.py_tree = InfinityQuadTreeNoShortcut(points_vector)

        # print("BUILT!")

        # print(self.py_tree.get_nodes())

    cpdef get_nodes(self):
        return self.py_tree.get_nodes()

    cpdef summarize(self, DTYPE_t[:] query_pt, DTYPE_t[:, :] X, float theta):
        # Used for testing summarize
        cdef:
            DTYPE_t[:] summary
            int n_samples, n_dimensions

        n_samples = X.shape[0]
        n_dimensions = X.shape[1]
        summary = np.zeros(4 * n_samples, dtype=np.float64)

        idx = self.py_tree.approximate_centers_of_mass(query_pt[0], query_pt[1], theta * theta, &summary[0])
        return idx, summary

