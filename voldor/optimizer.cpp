#include "optimizer.hpp"


// Generate the color wheel based on the algorithm by Baker et al.
void VU::makeColorWheel(std::vector<cv::Vec3b> &colorwheel) {
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;

    int i;

    for (i = 0; i < RY; i++) colorwheel.push_back(cv::Vec3b(255, 255 * i / RY, 0));
    for (i = 0; i < YG; i++) colorwheel.push_back(cv::Vec3b(255 - 255 * i / YG, 255, 0));
    for (i = 0; i < GC; i++) colorwheel.push_back(cv::Vec3b(0, 255, 255 * i / GC));
    for (i = 0; i < CB; i++) colorwheel.push_back(cv::Vec3b(0, 255 - 255 * i / CB, 255));
    for (i = 0; i < BM; i++) colorwheel.push_back(cv::Vec3b(255 * i / BM, 0, 255));
    for (i = 0; i < MR; i++) colorwheel.push_back(cv::Vec3b(255, 0, 255 - 255 * i / MR));
}

// Convert optical flow to color based on the color wheel
cv::Mat VU::flowToColor(const cv::Mat &flow_in) {
    cv::Mat flow;

    // Flow is assumed to be float
    flow_in.convertTo(flow, CV_32F);

    cv::Mat color(flow.size(), CV_8UC3);
    std::vector<cv::Vec3b> colorwheel;
    makeColorWheel(colorwheel);
    int ncols = colorwheel.size();

    for (int y = 0; y < flow.rows; y++) {
        for (int x = 0; x < flow.cols; x++) {
            
            cv::Vec2f flow_at_xy = flow.at<cv::Vec2f>(y, x);
            float fx = flow_at_xy[0];
            float fy = flow_at_xy[1];
            float rad = sqrt(fx * fx + fy * fy);
            float angle = atan2(-fy, -fx) / CV_PI;
            float fk = (angle + 1.0) / 2.0 * (ncols - 1);
            int k0 = (int)fk;
            int k1 = (k0 + 1) % ncols;
            float f = fk - k0;

            cv::Vec3b col0 = colorwheel[k0];
            cv::Vec3b col1 = colorwheel[k1];
            cv::Vec3b col;

            for (int b = 0; b < 3; b++)
                col[b] = (uchar)((1 - f) * col0[b] + f * col1[b]);

            if (rad <= 1.0)
                col *= rad;
            else
                col *= 1.0;
            color.at<cv::Vec3b>(y, x) = col;
            
        }
    }
    return color;
}


cv::Mat VU::warpFlow(const cv::Mat& img, const cv::Mat& flow) {
    // Get the height and width from the flow
    int h = flow.rows;
    int w = flow.cols;

    // Create a new flow matrix
    cv::Mat flowRemap = flow.clone();

    // Datatype should be CV_32F
    flowRemap.convertTo(flowRemap, CV_32F);

    // Negate the flow
    flowRemap *= -1;

    // Adjust flow for remapping
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            flowRemap.at<cv::Vec2f>(y, x)[0] += (float)x; // Add x coordinate
            flowRemap.at<cv::Vec2f>(y, x)[1] += (float)y; // Add y coordinate
        }
    }

    // Remap the image using the flow
    cv::Mat warpedImg;
    cv::remap(img, warpedImg, flowRemap, cv::Mat(), cv::INTER_LINEAR);

    return warpedImg;
}

// Eigen::MatrixXd cvFlow2Eigen(cv::Mat flow){
//      // Vector to hold the two channels
//     std::vector<cv::Mat> flow_channels(2);

//     // flow datatype should be double
//     flow.convertTo(flow, CV_64F);

//     // Split the flow into its channels
//     cv::split(flow, flow_channels);

//     // cv::transpose(flow_channels[0], flow_channels[0]);
//     // cv::transpose(flow_channels[1], flow_channels[1]);

//     // Vectorize observed flow
//     Eigen::MatrixXd vectorized_flow_x = Eigen::Map<Eigen::RowVectorXd>(flow_channels[0].ptr<double>(), flow_channels[0].rows*flow_channels[0].cols);
//     Eigen::MatrixXd vectorized_flow_y = Eigen::Map<Eigen::RowVectorXd>(flow_channels[1].ptr<double>(), flow_channels[1].rows*flow_channels[1].cols);

//     Eigen::MatrixXd vectorized_flow(2, vectorized_flow_x.cols());
//     vectorized_flow << vectorized_flow_x, vectorized_flow_y;
//     return vectorized_flow;
// }

cv::Mat VU::EigenVectorFlow2cvMat(Eigen::MatrixXd vectorized_flow, int width, int height){
    Eigen::MatrixXd vectorized_flow_x = vectorized_flow.row(0);
    Eigen::MatrixXd vectorized_flow_y = vectorized_flow.row(1);

    // Convert Flow into OpenCV cv::Mat object
    cv::Mat flow_x, flow_y;
    flow_x = cv::Mat(height, width, CV_64F, vectorized_flow_x.data()).clone(); 
    flow_y = cv::Mat(height, width, CV_64F, vectorized_flow_y.data()).clone(); 

    // Create a vector to hold the channels
    std::vector<cv::Mat> channels = { flow_x, flow_y };

    // Merge channels into a single multi-channel Mat
    cv::Mat flow;
    cv::merge(channels, flow);
    return flow;
}

Eigen::Matrix3d LU::Skew(Eigen::Vector3d vec){
    Eigen::Matrix3d S;
    S <<    0.0, -vec[2], vec[1],
            vec[2], 0.0, -vec[0],
            -vec[1], vec[0], 0.0;
    return S;
}

Eigen::Matrix3d LU::exp_SO3(Eigen::Vector3d psi){
    double angle = psi.norm();

    // If psi is too small, return Identity
    if(angle<LU::_tolerance){
        return Eigen::MatrixXd::Identity(3,3);
    }

    Eigen::Vector3d unit_psi        = psi / angle;
    Eigen::Matrix3d unit_psi_skew   = LU::Skew(unit_psi);

    Eigen::Matrix3d R = Eigen::MatrixXd::Identity(3,3) + std::sin(angle) * unit_psi_skew + (1-std::cos(angle)) * unit_psi_skew * unit_psi_skew;

    return R;
}




SOLVER::CamModel::CamModel(){

    // Initialize Intrinsic Parameters
    K << fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0;
    K_inv = K.inverse();

    // std::cout << K_inv << std::endl;

    // Initialize Extrinsic Parameters
    R_cam_gimbal << 0.0, 0.0, 1.0,
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0;

    R_gimbal_body << 0.0, 0.0, -1.0,
                        0.0, 1.0, 0.0,
                        1.0, 0.0, 0.0;

    // Camera Frame Expressed in Body Frame
    R_c_b = R_gimbal_body * R_cam_gimbal;       // The operator (*) is Matrix Multiplication in Eigen Library
    t_c_b << 0.0, 0.5, 0.0;
    T_c_b.block(0,0,3,3) = R_c_b;
    T_c_b.block(0,3,3,1) = t_c_b;

    
    // Body Frame Expressed in Camera Frame
    R_b_c = R_c_b.transpose();
    t_b_c = - R_b_c * t_c_b;                    // The operator (*) is Matrix Multiplication in Eigen Library
    T_b_c.block(0,0,3,3) = R_b_c;
    T_b_c.block(0,3,3,1) = t_b_c;

    // Compute Useful Parameters to Be Used Later
    getVectorizedPixels();
    _vectorized_bearings = K_inv * _vectorized_pixels;  // The operator (*) is Matrix Multiplication in Eigen Library
    _vectorized_pixels2  = _vectorized_pixels.topRows(2);
} // SOLVER::CamModel::getVectorizedPixels()


void SOLVER::CamModel::getVectorizedPixels(){
    for(int row=0; row<_height; ++row){
        for(int col=0; col<_width; ++col){
            _vectorized_pixels(0, row*_width + col) = (double)col + 0.5;  // The Pixel Coordinate is assigned to be middle of the pixel
            _vectorized_pixels(1, row*_width + col) = (double)row + 0.5;  // Hence we add 0.5
        }
    }

} // SOLVER::CamModel::getPixelMat()




Eigen::MatrixXd SOLVER::CamModel::getEstimatedOF(const Eigen::RowVectorXd vectorized_depth_map, const Eigen::Matrix3d Delta_R, const Eigen::Vector3d Delta_t, const Eigen::Matrix3d R_g_b, const Eigen::Vector3d v_gb_b, const double dT) const{   

    Eigen::Vector3d gravity = _gravity_direction * _gravity_magnitude;

    ////////// Get The Incremental Pose //////////
    // Orientation
    Eigen::Matrix3d R_bn_bc, R_bc_bn;
    R_bn_bc = Delta_R;   // R_body(next)_body(current) ->  Axes of the next body frame resolved in current body frame
    R_bc_bn = R_bn_bc.transpose();
    
    // Translarion
    Eigen::Vector3d t_bn_bc; 
    t_bn_bc = Delta_t + v_gb_b * dT + (0.5*dT*dT) * R_g_b * gravity; // t_body(next)_body(current) ->  Translation vector from current body frame to next body frame expressed in current body frame

    ////////// Express the Iterative Pose In Camera Frame //////////
    // Orientation
    Eigen::Matrix3d R_cc_cn = R_b_c * R_bc_bn * R_c_b; // R_cam(current)_cam(next) ->  Axes of the current cam frame resolved in next cam frame
    // Translation
    Eigen::Vector3d t_cn_cc, t_cc_cn; 
    t_cn_cc = R_b_c * (t_bn_bc + (R_bn_bc - Eigen::Matrix3d::Identity(3,3))*t_c_b); // t_cam(next)_cam(current) ->  Translation vector from current cam frame to next cam frame expressed in current cam frame
    t_cc_cn = - R_cc_cn * t_cn_cc; // t_cam(current)_cam(next) ->  Translation vector from next cam frame to current cam frame expressed in next cam frame

    
    ////////// Compute The Estimated Optical Flow //////////
    // get omega
    Eigen::MatrixXd KR = K * R_cc_cn;
    Eigen::MatrixXd    w_XY   =   KR.topRows(2) * _vectorized_bearings;
    Eigen::RowVectorXd w_Z    =     KR.row(2)   * _vectorized_bearings;;

    // get b
    Eigen::VectorXd b = K * t_cc_cn;
    Eigen::VectorXd b_XY = b.topRows(2);
    
    // Get Propagated Pixels
    Eigen::MatrixXd _propagated_pixels = ((w_XY.array().rowwise() * vectorized_depth_map.array()).colwise() + b_XY.array()).array().rowwise() / ((w_Z.array() * vectorized_depth_map.array()) + b(2,0)).array();

    // Get Estimated Flow
    Eigen::MatrixXd vectorized_estimated_flow = _propagated_pixels - _vectorized_pixels2;

    return vectorized_estimated_flow;
} // Eigen::MatrixXd SOLVER::CamModel::getEstimatedOF


SOLVER::Manager::Manager(const float* flows_pt,
                         float* depth_pt,
                         const float* kinematic_params_pt,
                         int width,
                         int height)
    : _flows_pt(flows_pt), _depth_pt(depth_pt), _height(height), _width(width)
{   
    // Set up the optimizer parameters
    _options.linear_solver_type = ceres::DENSE_QR;
    _options.minimizer_progress_to_stdout = false;
    _options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    _options.max_num_iterations = 500;
    _options.num_threads = 8;
    _options.min_trust_region_radius = 1e-12;

    // Convert data into Eigen Matrices
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> kinematic_params_float(kinematic_params_pt, 3, 12);
    Eigen::MatrixXd kinematic_params = kinematic_params_float.cast<double>(); // Convert datatype to double

    // Decompose the kinematic state into individual parameters
    _R_g_b = kinematic_params.block(0,0,3,3);
    _v_gb_b = kinematic_params.block(0,3,3,1);
    _Delta_R = kinematic_params.block(0,8,3,3);
    _Delta_t = kinematic_params.block(0,11,3,1);
} // SOLVER::Manager::Manager


void SOLVER::Manager::computeInnovation(float* state_innovation_pt){
    // Get The Observed Flow
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> flow_float(_flows_pt, 2, _width*_height);
    Eigen::MatrixXd observed_vectorized_flow = flow_float.cast<double>(); // For optimization framework we need that datatype is double

    // Get the Estimated Depth Map
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> depth_float(_depth_pt, 1, _width*_height);
    Eigen::MatrixXd vectorized_depth_map = depth_float.cast<double>(); // For optimization framework we need that datatype is double

    // // Mini Visualization To Check if the Data Is Interpreted Correctly
    // Eigen::MatrixXd estimated_vectorized_flow = _cam_model.getEstimatedOF(vectorized_depth_map, _Delta_R, _Delta_t, _R_g_b, _v_gb_b, _dT);
    // cv::Mat observed_flow_vis = VU::EigenVectorFlow2cvMat(observed_vectorized_flow, _width, _height);
    // observed_flow_vis = VU::flowToColor(observed_flow_vis);
    // cv::imshow("observed_flow_vis", observed_flow_vis);

    // cv::Mat estimated_flow_vis = VU::EigenVectorFlow2cvMat(estimated_vectorized_flow, _width, _height);
    // estimated_flow_vis = VU::flowToColor(estimated_flow_vis);
    // cv::imshow("estimated_flow_vis", estimated_flow_vis);

    // cv::waitKey();

    // Initial guess for the optimization
    double x[3] = {0.0, 0.0, 0.0};

    // Optical Flow Cost
    ceres::CostFunction* optical_flow_constraint = new ceres::NumericDiffCostFunction<SOLVER::OpticalFlowConstraint, ceres::CENTRAL, 1, 3>(
        new SOLVER::OpticalFlowConstraint(observed_vectorized_flow, vectorized_depth_map, _Delta_R, _Delta_t, _R_g_b, _v_gb_b, _dT));

    // Compute the Innovation Term
    ceres::Problem problem;
    problem.AddResidualBlock(optical_flow_constraint, new ceres::HuberLoss(0.1), x);
    ceres::Solve(_options, &problem, &_summary);

    /////////// Decompose Innivation into Velocity and Rotation ///////////
    Eigen::Vector3d innovation(x[0], x[1], x[2]);
    ceres::CostFunction* decomposition_cost = new ceres::NumericDiffCostFunction<SOLVER::InnovationDecomposer, ceres::CENTRAL, 8, 5>(
        new SOLVER::InnovationDecomposer(innovation, _R_g_b, _dT));
    
    double state_inn[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    ceres::Problem decomposer;
    decomposer.AddResidualBlock(decomposition_cost, new ceres::HuberLoss(0.1), state_inn);
    ceres::Solve(_options, &decomposer, &_summary);

    // Convert the Output Into float
    for (int i = 0; i < 5; ++i) {
        state_innovation_pt[i] = static_cast<float>(state_inn[i]);
    }
}


void SOLVER::Manager::visualize(const float* img_frame_pt)
{
    // Visualize Image Frames
    cv::Mat curr_img_frame = cv::Mat(cv::Size(_width, _height), CV_32F, (void*)(img_frame_pt));
    curr_img_frame.convertTo(curr_img_frame, CV_8U);
    cv::imshow("curr_img_frame", curr_img_frame);

    cv::Mat next_img_frame = cv::Mat(cv::Size(_width, _height), CV_32F, (void*)(img_frame_pt + _width*_height));
    next_img_frame.convertTo(next_img_frame, CV_8U);
    cv::imshow("next_img_frame", next_img_frame);

    // Visualize Depth Map
    cv::Mat depthMap = cv::Mat(cv::Size(_width, _height), CV_32F, (void*)(_depth_pt));
    depthMap.convertTo(depthMap, CV_64F);
    cv::Mat depthMap_vis = depthMap / 100.0 * 255.0;
    depthMap_vis.convertTo(depthMap_vis, CV_8U);
    cv::imshow("depthMap_vis", depthMap_vis);

    // Visualize Observed Optical Flow
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> flow_float(_flows_pt, 2, _width*_height);
    Eigen::MatrixXd observed_vectorized_flow = flow_float.cast<double>(); // Convert dataypye to double
    cv::Mat observed_flow = VU::EigenVectorFlow2cvMat(observed_vectorized_flow, _width, _height);
    // cv::Mat observed_flow = cv::Mat(cv::Size(_width, _height), CV_32FC2, (void*)(_flows_pt));
    cv::Mat observed_flow_vis = VU::flowToColor(observed_flow);
    cv::imshow("observed_flow_vis", observed_flow_vis);
    
    // Visualize Warping
    curr_img_frame.convertTo(curr_img_frame, CV_64F);
    next_img_frame.convertTo(next_img_frame, CV_64F);

    cv::Mat warped_frame = VU::warpFlow(curr_img_frame, observed_flow);
    cv::Mat warping_error = next_img_frame - warped_frame;

    warping_error = cv::abs(warping_error);
    warping_error.convertTo(warping_error, CV_8U);
    cv::imshow("warping_error", warping_error);
    

    // Visualize Estimated Flow
    // Eigen::MatrixXd vectorized_depth_map = Eigen::Map<Eigen::RowVectorXd>(depthMap.ptr<double>(), depthMap.rows * depthMap.cols);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> depth_float(_depth_pt, 1, _width*_height);
    Eigen::MatrixXd vectorized_depth_map = depth_float.cast<double>(); // Convert dataypye to double
    
    auto start = std::chrono::steady_clock::now();
    Eigen::MatrixXd estimated_vectorized_flow = _cam_model.getEstimatedOF(vectorized_depth_map, _Delta_R, _Delta_t, _R_g_b, _v_gb_b, _dT);
    auto end = std::chrono::steady_clock::now();
    
    // Calculate the elapsed time in milliseconds
    auto elapsed_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time: " << elapsed_milliseconds.count() << " ms\n";

    cv::Mat estimated_flow = VU::EigenVectorFlow2cvMat(estimated_vectorized_flow, _width, _height);
    cv::Mat estimated_flow_vis = VU::flowToColor(estimated_flow);
    cv::imshow("estimated_flow_vis", estimated_flow_vis);

    // Verify that the Estimated Flow is Working Fine
    cv::Mat estimated_warped_frame = VU::warpFlow(curr_img_frame, estimated_flow);
    cv::Mat estimated_warping_error = next_img_frame - estimated_warped_frame;

    estimated_warping_error = cv::abs(estimated_warping_error);
    estimated_warping_error.convertTo(estimated_warping_error, CV_8U);
    cv::imshow("estimated_warping_error", estimated_warping_error);

    cv::waitKey();
} // void Manager::visualize