# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as cnp
import cython
from libcpp.vector cimport vector
from libcpp.algorithm cimport partition

ctypedef cnp.npy_float64 DTYPE_t          # Type of X
ctypedef cnp.npy_intp SIZE_t              # Type for indices and counters
ctypedef cnp.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef cnp.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cdef extern from "point.hpp":
    cdef struct Point:
        float x
        float y

cdef extern from "cell.hpp":
    cdef struct Cell:
        size_t parent_idx
        # size_t child_bottom_left
        # size_t child_bottom_right
        # size_t child_top_left
        # size_t child_top_right
        vector[size_t] children_idx
        bint is_leaf
        # double min_bounds_x
        # double min_bounds_y
        # double max_bounds_x
        vector[double] min_bounds
        vector[double] max_bounds

cdef extern from "infinity_quad_tree.hpp":
    cdef cppclass InfinityQuadTree:
        InfinityQuadTree() except +
        InfinityQuadTree(SIZE_t n) except +
        vector[Cell] nodes

        size_t rec_build_tree(vector[Point].iterator begin_points, vector[Point].iterator end_points, float min_bounds_x, float min_bounds_y, float max_bounds_x, float max_bounds_y, SIZE_t parent)

cdef class PyInfinityQuadTree:
    cdef InfinityQuadTree py_tree

    def __cinit__(self, n=1000):
        self.py_tree = InfinityQuadTree(n)

    cpdef get_nodes(self):
        return self.py_tree.nodes

    cpdef build_tree(self, points, min_bounds_x, min_bounds_y, max_bounds_x, max_bounds_y):
        cdef vector[Point] points_vector
        points_vector.reserve(len(points))
        for i in range(len(points)):
            print("converted some")
            points_vector.emplace_back(Point(points[i][0], points[i][1]))
        print("converted nodes ", points_vector.size())

        self.py_tree.rec_build_tree(points_vector.begin(), points_vector.end(), min_bounds_x, min_bounds_y, max_bounds_x, max_bounds_y, 0)

        print(self.py_tree.nodes.size())



