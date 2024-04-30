#pragma once
#include "cell.hpp"
#include "point.hpp"
#include <stddef.h>
#include <vector>
#include <algorithm>

class InfinityQuadTree {
public:
    std::vector<Cell> nodes;
    InfinityQuadTree(int n) {
        nodes.reserve(n);
    }
    InfinityQuadTree() {}
    ~InfinityQuadTree(){}

    size_t rec_build_tree(std::vector<Point>::iterator begin_points, std::vector<Point>::iterator end_points, float min_bounds_x, float min_bounds_y, float max_bounds_x, float max_bounds_y, size_t parent){
        float bb_center_x = (max_bounds_x - min_bounds_x) / 2 + min_bounds_x;
        float bb_center_y = (max_bounds_y - min_bounds_y) / 2 + min_bounds_y;

        if (begin_points == end_points)
            return 0;

        size_t result_idx = nodes.size();
        nodes.emplace_back(Cell(parent, {0, 0, 0, 0}, false, {min_bounds_x, min_bounds_y}, {max_bounds_x, max_bounds_y}));

        if (begin_points + 1 == end_points) {
            if (isBoxWithinUnitCircle(min_bounds_x, min_bounds_y, max_bounds_x, max_bounds_y))
                return result_idx;
        }
            
        auto split_y = std::partition(begin_points, end_points, [bb_center_y](Point a){ return a.y < bb_center_y; });
        auto split_x_lower = partition(begin_points, split_y, [bb_center_x](Point a){ return a.x < bb_center_x; });
        auto split_x_upper = partition(split_y, end_points, [bb_center_x](Point a){ return a.x < bb_center_x; });

        nodes[result_idx].children_idx[0] = rec_build_tree(split_y, split_x_upper, min_bounds_x, bb_center_y, bb_center_x, max_bounds_y, result_idx);
        nodes[result_idx].children_idx[1] = rec_build_tree(split_x_upper, end_points, bb_center_x, bb_center_y, max_bounds_x, max_bounds_y, result_idx);
        nodes[result_idx].children_idx[2] = rec_build_tree(begin_points, split_x_lower, min_bounds_x, min_bounds_y, bb_center_x, bb_center_y, result_idx);
        nodes[result_idx].children_idx[3] = rec_build_tree(split_x_lower, split_y, bb_center_x, min_bounds_y, max_bounds_x, bb_center_y, result_idx);

        if (nodes[result_idx].children_idx[0] == 0 || nodes[result_idx].children_idx[1] == 0 || nodes[result_idx].children_idx[2] == 0 || nodes[result_idx].children_idx[3])
            nodes[result_idx].is_leaf = true;

        return result_idx;
    }
private:
    static bool isBoxWithinUnitCircle(float min_bounds_x, float min_bounds_y, float max_bounds_x, float max_bounds_y) {
        float d1 = min_bounds_x*min_bounds_x + min_bounds_y*min_bounds_y;
        float d2 = max_bounds_x*max_bounds_x + min_bounds_y*min_bounds_y;
        float d3 = max_bounds_x*max_bounds_x + max_bounds_y*max_bounds_y;
        float d4 = min_bounds_x*min_bounds_x + max_bounds_y*max_bounds_y;
        return d1 < 1.0 && d2 < 1.0 && d3 < 1.0 && d4 < 1.0;
    }
};