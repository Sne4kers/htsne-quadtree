#pragma once
#include "point.hpp"
#include <stddef.h>
#include <vector>

struct Cell{
    size_t parent_idx;
    std::vector<size_t> children_idx;

    bool is_leaf;
    size_t cumulative_size;
    size_t depth;

    Point center;
    Point min_bounds;
    Point max_bounds;

    Point barycenter;
    double lorentz_factor;

    Cell(
        size_t parent_idx_, size_t depth_, const Point& min_bounds_, const Point& max_bounds_) : 
        parent_idx(parent_idx_), 
        children_idx{{0, 0, 0, 0}}, 
        is_leaf(false), 
        cumulative_size(0), 
        depth(depth_),
        center{(min_bounds_ + max_bounds_) / 2},
        min_bounds{min_bounds_}, 
        max_bounds{max_bounds_},
        barycenter{(min_bounds_ + max_bounds_) / 2},
        lorentz_factor(1)
    {};
};