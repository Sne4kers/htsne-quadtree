#pragma once
#include <stddef.h>
#include <vector>

struct Cell{
    size_t parent_idx;
    std::vector<size_t> children_idx;
    // size_t child_bottom_right;
    // size_t child_top_left;
    // size_t child_top_right;

    bool is_leaf;

    // double min_bounds_x;
    // double min_bounds_y;
    // double max_bounds_x;
    // double max_bounds_y;
    std::vector<double> min_bounds;
    std::vector<double> max_bounds;

    // Cell(size_t parent_idx_, 
    // size_t child_top_left_, size_t child_top_right_ , 
    // size_t child_bottom_left_, size_t child_bottom_right_, 
    // bool is_leaf_, 
    // double min_bounds_x_, double min_bounds_y_, 
    // double max_bounds_x_, double max_bounds_y_) 
    // : parent_idx(parent_idx_), 
    // child_bottom_left(child_bottom_left_), child_bottom_right(child_bottom_right_), 
    // child_top_left(child_top_left_), child_top_right(child_top_right_), 
    // is_leaf(is_leaf_), 
    // min_bounds_x(min_bounds_x_), min_bounds_y(min_bounds_y_), 
    // max_bounds_x(max_bounds_x_), max_bounds_y(max_bounds_y_)
    // {};

    Cell(size_t parent_idx_, std::vector<size_t> children_idx_, bool is_leaf_, std::vector<double> min_bounds_, std::vector<double> max_bounds_
    ) : parent_idx(parent_idx_), 
    children_idx{children_idx_}, is_leaf(is_leaf_), min_bounds{min_bounds_}, max_bounds{max_bounds_}
    {};
};