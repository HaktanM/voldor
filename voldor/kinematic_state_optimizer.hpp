#pragma once
// Include standard libraries for std::cout manipulation
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip> // For std::fixed and std::setprecision

// include Ceres optimization library
#include "ceres/ceres.h"

// include Eigen
#include <Eigen/Eigen>
#include <Eigen/Core>

// include OpenCV for Visualization
#include <opencv2/opencv.hpp>


class KINEMATIC_STATE {
public:
    KINEMATIC_STATE(const float* flows_pt,
                    float* depth_pt,
                    const float* kinematic_state_pt,
                    double* velocity_innovation,
                    int width,
                    int height);

    void optimize();

    void visualize(const float* img_frame_pt);

private:
    Eigen::MatrixXd curr_flow;
	Eigen::MatrixXd curr_depth_map;
    Eigen::MatrixXd kinematic_state;

    int _width{0};
    int _height{0};
}; // class KINEMATIC_STATE