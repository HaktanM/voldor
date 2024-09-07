#include "py_export.h"
#include "voldor.h"
#include "optimizer.hpp"
#include <iterator>

int py_voldor_wrapper(
	// inputs
	const float* flows_pt,
	const float* img_frame_pt,
	const float* kinematic_state_pt,
	const float* disparity_pt, const float* disparity_pconf_pt,
	const float* depth_priors_pt, const float* depth_prior_poses_pt, const float* depth_prior_pconfs_pt,
	const float fx, const float fy, const float cx, const float cy, const float basefocal,
	const int N, const int N_dp, const int w, const int h,
	const char* config_pt,
	// outputs
	int& n_registered, float* innovation_pt, float* poses_pt, float* poses_covar_pt, float* depth_pt, float* depth_conf_pt) {

	Config cfg;
	std::istringstream iss(config_pt);
	std::vector<std::string> cfg_strs(
		std::istream_iterator<std::string>{iss},
		std::istream_iterator<std::string>());
	cfg.fx = fx;
	cfg.cx = cx;
	cfg.fy = fy;
	cfg.cy = cy;
	cfg.basefocal = basefocal;
	cfg.read_config(cfg_strs);


	vector<Mat> flows;
	Mat disparity;
	Mat disparity_pconf;
	vector<Mat> depth_priors;
	vector<Vec6f> depth_prior_poses;
	vector<Mat> depth_priors_pconfs;

	for (int i = 0; i < N; i++) {
		flows.push_back(Mat(Size(w, h), CV_32FC2, (void*)(flows_pt + i * w*h * 2)));
		depth_prior_poses.push_back(Vec6f(depth_prior_poses_pt + i * 6));
	}

	if (disparity_pt)
		disparity = Mat(Size(w, h), CV_32F, (void*)disparity_pt);
	if (disparity_pconf_pt)
		disparity_pconf = Mat(Size(w, h), CV_32F, (void*)disparity_pconf_pt);

	for (int i = 0; i < N_dp; i++) {
		depth_priors.push_back(Mat(Size(w, h), CV_32F, (void*)(depth_priors_pt + i * w*h)));
		if (depth_prior_pconfs_pt)
			depth_priors_pconfs.push_back(Mat(Size(w, h), CV_32F, (void*)(depth_prior_pconfs_pt + i * w*h)));
	}

	// Get the Estimated Depth Map
	VOLDOR voldor(cfg);
	voldor.init(flows, disparity, disparity_pconf, depth_priors, depth_prior_poses, depth_priors_pconfs);
	voldor.solve();

	

	n_registered = voldor.n_flows;

	if (depth_pt){
		memcpy(depth_pt, voldor.depth.data, w*h * sizeof(float));

		// Get the Innovation for Kinematic State
		float state_innovation_pt[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
		SOLVER::Manager kinematic_state_optimizer(flows_pt, depth_pt, kinematic_state_pt, w, h);	
		kinematic_state_optimizer.computeInnovation(state_innovation_pt);
		// kinematic_state_optimizer.visualize(img_frame_pt);

		std::cout << "Innovations : ";
		for (int i = 0; i < 5; ++i) {
			std::cout << state_innovation_pt[i] << ", ";
		}
		std::cout << std::endl;

		if(innovation_pt)
			memcpy(innovation_pt, state_innovation_pt, 5 * sizeof(float));
	};

	if (depth_conf_pt) {
		Mat depth_conf = Mat::zeros(Size(w, h), CV_32F);
		for (int i = 0; i < voldor.n_flows; i++)
			depth_conf += voldor.rigidnesses[i];
		for (int i = 0; i < voldor.n_depth_priors; i++)
			depth_conf += voldor.depth_prior_confs[i];
		depth_conf /= (float)(voldor.n_flows + voldor.n_depth_priors);
		memcpy(depth_conf_pt, depth_conf.data, w*h * sizeof(float));
	}

	return 0;
}