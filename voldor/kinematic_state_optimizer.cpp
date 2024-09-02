#include "kinematic_state_optimizer.hpp"

struct OpticalFlowConstraint{
    OpticalFlowConstraint(  const Eigen::MatrixXd& observed_flow, 
                            const Eigen::MatrixXd& depth_map,
                            const Eigen::Matrix3d& delta_R,
                            const Eigen::Vector3d& delta_t,
                            const Eigen::Matrix3d& R_c_g,
                            const Eigen::Vector3d& v_gc_g,
                            const double& dT
    )
    : _observed_flow(observed_flow), _depth_map(depth_map), _delta_R(delta_R), _delta_t(delta_t), 
    _R_c_g(R_c_g), _v_gc_g(v_gc_g), _dT(dT) {}

    // Operator to compute the Mahalanobis distance
    bool operator()(const double* const x, double* residual) const {
        Eigen::Vector3d diff;
        diff[0] = x[0];
        diff[1] = x[1];
        diff[2] = x[2];
        
        residual[0] = diff.norm();
        return true;
    }
    
private:
    Eigen::MatrixXd _observed_flow;     // Observed Optical Flow
    Eigen::MatrixXd _depth_map;         // Estimated or measured depth map
    Eigen::Matrix3d _delta_R;           // Preintegrated IMU measurement for rotation
    Eigen::Vector3d _delta_t;           // Preintegrated IMU measurement for translation
    Eigen::Matrix3d _R_c_g;             // Initial Camera Orientation w.r.t. global frame
    Eigen::Vector3d _v_gc_g;            // Inverse Camera Velocity w.r.t. global frame
    double _dT;                         // Elapsed Time Between Two Frames           
    double _gravity_magnitude = 9.80665016174316;
};

struct InnovationCost {
    InnovationCost(const Eigen::Matrix3d& covariance)
        : inv_covariance_(covariance) {
            std::cout << "Inside the cost : " << std::endl << inv_covariance_ << std::endl;
        }

    // Operator to compute the Mahalanobis distance
    bool operator()(const double* const x, double* residual) const {
        Eigen::Vector3d diff;
        diff.x() = x[0] - 1.0;
        diff.y() = x[1] - 2.0;
        diff.z() = x[2] - 1.0;
        
        residual[0] = std::sqrt(diff.transpose() * inv_covariance_ * diff);

        std::cout << "diff : " << std::endl << diff << std::endl;
        std::cout << "residual[0] : " << std::endl << residual[0] << std::endl;

        return true;
    }
    
private:
    Eigen::Matrix3d inv_covariance_;    // Inverse covariance as Eigen matrix
};


KINEMATIC_STATE::KINEMATIC_STATE(   const float* flows_pt,
                                    float* depth_pt,
                                    const float* kinematic_state_pt,
                                    double* velocity_innovation,
                                    int w,
                                    int h)
{
    // Convert data into Eigen Matrices
	Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> curr_flow_float(flows_pt, 2, _width*_height);
	Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> curr_depth_float(depth_pt, 1, _height*_width);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> kinematic_state_float(kinematic_state_pt, 3, 12);

    // For optimization framework we need that datatype is double
	curr_flow = curr_flow_float.cast<double>();
	curr_depth_map = curr_depth_float.cast<double>();
    kinematic_state = kinematic_state_float.cast<double>();

    // Decompose the kinematic state into individual parameters
    Eigen::Matrix3d R_b_g = kinematic_state.block(0,0,3,3);
    Eigen::Vector3d v_b_g = kinematic_state.block(0,3,3,1);
    Eigen::Matrix3d R_c_b = kinematic_state.block(0,4,3,3);
    Eigen::Vector3d t_c_b = kinematic_state.block(0,7,3,1);
    Eigen::Matrix3d Delta_R = kinematic_state.block(0,8,3,3);
    Eigen::Vector3d Delta_t = kinematic_state.block(0,11,3,1);

    double dT{0.05};   // Time interval between two consequtive optical flow measurements (second)

    _width = w;
    _height = h;
} // void KINEMATIC_STATE::KINEMATIC_STATE

void KINEMATIC_STATE::optimize() { } // void VOLDOR::init


void KINEMATIC_STATE::visualize(const float* img_frame_pt)
{
    // Convert the Image OpenCV Matrix
    cv::Mat img_frame = cv::Mat(cv::Size(_width, _height), CV_32F, (void*)(img_frame_pt));

     // Convert to unsigned 8-bit (CV_8U) with scaling
    cv::Mat uchar_img;
    img_frame.convertTo(uchar_img, CV_8UC1); // Multiply by 255 to scale the values to [0, 255]


    cv::imshow("Window", uchar_img);
    cv::waitKey();
} //void KINEMATIC_STATE::visualize