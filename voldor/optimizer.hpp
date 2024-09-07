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

// Optimization Library
#include "ceres/ceres.h"

// To measure the elapsed time
#include <chrono>


namespace VU{ // VU Stands For Visualization Tools
    void makeColorWheel(std::vector<cv::Vec3b> &colorwheel);
    cv::Mat flowToColor(const cv::Mat &flow);
    cv::Mat warpFlow(const cv::Mat& img, const cv::Mat& flow);
    cv::Mat EigenVectorFlow2cvMat(Eigen::MatrixXd vectorized_flow, int width, int height);
} // namespace VU


namespace LU{ // LU stands for Lie Algebra Utils
    static double _tolerance{1e-10};
    Eigen::Matrix3d Skew(Eigen::Vector3d vec);
    Eigen::Matrix3d exp_SO3(Eigen::Vector3d psi);
} // namespace LU

namespace SOLVER{
    // Camera Intrinsic and Extrinsic Calibration Parameters are stored.
    class CamModel{
    public:
        CamModel();
        // Intrinsic Parameters
        const int _width  = 640;
        const int _height = 512;
        double fx{320.0}, fy{320.0}, cx{320.0}, cy{256.0};
        Eigen::Matrix3d K, K_inv;

        // Extrinsic Parameters
        Eigen::Matrix3d R_cam_gimbal, R_gimbal_body, R_c_b, R_b_c;
        Eigen::Vector3d t_c_b, t_b_c;
        Eigen::Matrix4d T_c_b = Eigen::MatrixXd::Identity(4,4);
        Eigen::Matrix4d T_b_c = Eigen::MatrixXd::Identity(4,4);

        // Get Estimated Optical Flow From Estimated State and IMU measurements
        Eigen::MatrixXd getEstimatedOF(const Eigen::RowVectorXd vectorized_depth_map, const Eigen::Matrix3d Delta_R, const Eigen::Vector3d Delta_t, const Eigen::Matrix3d R_b_g, const Eigen::Vector3d v_gb_b, const double dT) const;
        
        Eigen::Vector3d getGravity() const{
            return _gravity_direction * _gravity_magnitude;
        }
    private:
        // Usefull Parameters
        Eigen::MatrixXd _vectorized_pixels   = Eigen::MatrixXd::Ones(3, _width*_height);
        Eigen::MatrixXd _vectorized_pixels2  = Eigen::MatrixXd::Zero(2, _width*_height);
        Eigen::MatrixXd _vectorized_bearings = Eigen::MatrixXd::Zero(3, _width*_height);

        void getVectorizedPixels();

        // default gravity magnitude
        double _gravity_magnitude{9.80665016174316};
        Eigen::Vector3d _gravity_direction{0.0, 0.0, 1.0};
    }; // cam_model


    // Define Optical Flow Constraint
    struct OpticalFlowConstraint{
    public:
        OpticalFlowConstraint(  
            const Eigen::MatrixXd& vectorized_flow,
            const Eigen::RowVectorXd& vectorized_depth_map,
            const Eigen::MatrixXd& Delta_R,
            const Eigen::VectorXd& Delta_t,
            const Eigen::MatrixXd& R_g_b,
            const Eigen::VectorXd& v_gb_b,
            const double &dT
        )
        : _vectorized_flow(vectorized_flow), _vectorized_depth_map(vectorized_depth_map), _Delta_R(Delta_R), _Delta_t(Delta_t), _R_g_b(R_g_b), _v_gb_b(v_gb_b), _dT(dT) 
    {};

        // Operator to compute the Mahalanobis distance
        bool operator()(const double* const x, double* residual) const {
            Eigen::Vector3d innovation_vector(x[0], x[1], x[2]);
            Eigen::Vector3d v_innovated = _v_gb_b + innovation_vector;
            Eigen::MatrixXd vectorized_estimated_flow = _cam_model.getEstimatedOF(_vectorized_depth_map, _Delta_R, _Delta_t, _R_g_b, v_innovated, _dT);
            Eigen::MatrixXd flow_res = vectorized_estimated_flow - _vectorized_flow;

            // residual[0] = (flow_res.row(0).array() / _vectorized_depth_map.array()).array().abs().sum() ;
            // residual[1] = (flow_res.row(1).array() / _vectorized_depth_map.array()).array().abs().sum() ;
            residual[0] = flow_res.array().abs().sum();
            return true;
        }

    private:
        SOLVER::CamModel _cam_model;
        Eigen::MatrixXd _vectorized_flow;     // Observed Optical Flow
        Eigen::RowVectorXd _vectorized_depth_map;         // Estimated or measured depth map
        Eigen::MatrixXd _Delta_R;           // Preintegrated IMU measurement for rotation
        Eigen::VectorXd _Delta_t;           // Preintegrated IMU measurement for translation
        Eigen::MatrixXd _R_g_b;             // Initial Camera Orientation w.r.t. global frame
        Eigen::VectorXd _v_gb_b;            // Inverse Camera Velocity w.r.t. global frame
        double _dT;                         // Elapsed Time Between Two Frames           
    }; // struct OpticalFlowConstraint


    // Define Optical Flow Constraint
    struct InnovationDecomposer{
    public:
        InnovationDecomposer(  
            const Eigen::Vector3d& innovation,
            const Eigen::MatrixXd& R_g_b,
            const double &dT
        )
        : _innovation(innovation), _R_g_b(R_g_b), _dT(dT) 
        {};

        // Operator to compute the Mahalanobis distance
        bool operator()(const double* const x, double* residual) const {

            Eigen::Vector3d rot_inn(x[0], x[1], 0.0);       // Innovation Terms
            Eigen::Vector3d vel_inn(x[2], x[3], x[4]);       // Innovation Terms
            Eigen::Matrix3d R_inn = LU::exp_SO3(rot_inn);
            Eigen::Vector3d expected_inn = vel_inn + 0.5 * _dT * _R_g_b * (R_inn - Eigen::MatrixXd::Identity(3,3)) * _cam_model.getGravity();
            Eigen::Vector3d inn_res = expected_inn - _innovation;
            
            // Innovation Constraint
            residual[0] = inn_res[0];
            residual[1] = inn_res[1];
            residual[2] = inn_res[2];

            // Cost for rotation innovation
            residual[3] = x[0] * _rot_weight;
            residual[4] = x[1] * _rot_weight;

            // Cost for velocity innovation
            residual[5] = x[2] * _vel_weight;
            residual[6] = x[3] * _vel_weight;
            residual[7] = x[4] * _vel_weight;

            return true;
        }

    private:
        SOLVER::CamModel _cam_model;
        Eigen::Vector3d _innovation;
        Eigen::Matrix3d _R_g_b;                             // Initial Camera Orientation w.r.t. global frame
        double _dT;                         // Elapsed Time Between Two Frames           
        double _rot_weight{10.0};
        double _vel_weight{5.0};
    }; // struct OpticalFlowConstraint

    class Manager{
    public:
        Manager(const float* flows_pt,
                float* depth_pt,
                const float* kinematic_params_pt,
                int width,
                int height);
    
        void visualize(const float* img_frame_pt);
        void computeInnovation(float* state_innovation_pt);

        SOLVER::CamModel _cam_model;

        // CERES Solver Options
        ceres::Solver::Options _options;
        ceres::Solver::Summary _summary;
    private:       
        float* _depth_pt;
        const float* _flows_pt;
        int _width{0}, _height{0};
        double _dT{0.05};   // Time interval between two consequtive optical flow measurements (second)

        Eigen::Matrix3d _R_g_b;
        Eigen::Vector3d _v_gb_b;
        Eigen::Matrix3d _Delta_R;
        Eigen::Vector3d _Delta_t;
    }; // Manager
} // namespace Optimizer


