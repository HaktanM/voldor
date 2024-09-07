import threading # threading is only used for Visualizer
import time
import cv2
import numpy as np
import my_utils
import raft_utils
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from functools import partial


import sys
sys.path.append('slam_py/')
sys.path.append('slam_py/install')
from flow_utils import load_flow

from slam_utils import *
from voldor_viewer import VOLDOR_Viewer
from voldor_slam import VOLDOR_SLAM

try:
    import pyvoldor_vo as pyvoldor
    pyvoldor_module = 'vo'
    print('VO pyvoldor module loaded.')
except:
    raise 'Cannot load pyvoldor module.'

import argparse
parser = argparse.ArgumentParser(description='VOLDOR-SLAM demo script')
parser.add_argument('--mode', type=str, default='mono', help='One from stereo/mono-scaled/mono. For stereo and mono-scaled, disparity input will be required.')
parser.add_argument('--fx', type=float, default=320)
parser.add_argument('--fy', type=float, default=320)
parser.add_argument('--cx', type=float, default=320)
parser.add_argument('--cy', type=float, default=256)
parser.add_argument('--bf', type=float, default=0, help='Baseline x focal, which determines the world scale. If set to 0, default baseline is 0.')
parser.add_argument('--resize', type=float, default=1.0, help='resize input size')
parser.add_argument('--abs_resize', type=float, default=1.0, help='Resize factor related to the size that optical flow is estimated from. (useful to residual model)')
parser.add_argument('--enable_mapping', default=True, action='store_true')
parser.add_argument('--save_poses', type=str)
parser.add_argument('--save_depths', type=str)

opt = parser.parse_args()
if opt.abs_resize is None:
    opt.abs_resize = opt.resize
    
    
if __name__ == '__main__':
    # init slam instance and select mode from mono/mono-scaled/stereo
    slam = VOLDOR_SLAM(mode=opt.mode, path_to_data = "/home/hakito/python_scripts/AirSim/Data1")
    
    num_of_flows = slam.data.FlowList.itemCount()

    start_idx = 150
    # flow_idx = start_idx
    for flow_idx in range(start_idx, num_of_flows-slam.voldor_winsize-1):
        
        # Load the flows
        flows = []
        prior_poses = []
        
        for win_idx in range(slam.voldor_winsize):
            path_to_flow = slam.data.FlowList.getItemPath(flow_idx + win_idx)
            flow = load_flow(path_to_flow)
            flows.append(flow)
            
            curr_flow_timestamp = slam.data.FlowList.getItemID(flow_idx + win_idx)
            next_flow_timestamp = slam.data.FlowList.getItemID(flow_idx + win_idx + 1)
            
            dT = (next_flow_timestamp - curr_flow_timestamp) * (1e-9)
            
            # Current GT
            curr_flow_timestamp = slam.data.FlowList.getItemID(flow_idx + win_idx)
            q_b0_g, v_b0_g, t_b0_g = slam.data.get_gt(curr_flow_timestamp) 
            R_b0_g = R.from_quat(q_b0_g).as_matrix().reshape(3,3)
            R_g_b0 = R_b0_g.transpose()
            
            # Use IMU measurements to get iterative velocity
            Delta_R_b_g, Delta_v_b_g, Delta_t_b_g = slam.data.get_preintegrated_measurement(curr_flow_timestamp, next_flow_timestamp)
            Delta_v_b_g = Delta_v_b_g.reshape(3,1)
            Delta_t_b_g = Delta_t_b_g.reshape(3,1)
            
            R_b1_b0 = Delta_R_b_g
            t_b1_b0 = Delta_t_b_g + (R_g_b0 @ v_b0_g).reshape(3,1) * dT + 0.5 * (dT**2) * (R_g_b0 @ slam.gravity.reshape(3,1)).reshape(3,1)
            
            # Express the iterative pose with respect to camera 
            R_c1_c0 = slam.cam_model.R_b_c @ R_b1_b0 @ slam.cam_model.R_c_b
            t_c1_c0 = slam.cam_model.R_b_c @ (t_b1_b0.reshape(3,1) + (R_b1_b0 - np.eye(3))@(slam.cam_model.t_c_b.reshape(3,1))).reshape(3,1)
            
            R_c0_c1 = R_c1_c0.transpose()
            t_c0_v1 = - R_c0_c1 @ t_c1_c0.reshape(3,1)
        
            T_c0_c1 = np.eye(4, dtype=np.float32)
            T_c0_c1[:3,:3] = R_c0_c1
            T_c0_c1[:3,3] = t_c0_v1.reshape(3)
            
            prior_poses.append(T44_to_T6(T_c0_c1))
        
        depth_priors = []
        depth_prior_pconfs = []
        
        # Get current state
        curr_flow_timestamp = slam.data.FlowList.getItemID(flow_idx)
        q_b0_g, v_b0_g, t_b0_g = slam.data.get_gt(curr_flow_timestamp)   
        R_b0_g = R.from_quat(q_b0_g).as_matrix().astype(np.float32).reshape(3,3)
        v_gb_g = v_b0_g.copy().astype(np.float32).reshape(3,1)
        
        # Get calibration parameters
        T_c_b_ = slam.cam_model.T_c_b[:3,:].astype(np.float32).reshape(3,4)
        
        # Get preintegrated IMU measurement
        curr_flow_timestamp = slam.data.FlowList.getItemID(flow_idx + win_idx)
        next_flow_timestamp = slam.data.FlowList.getItemID(flow_idx + win_idx + 1)
        dT = (next_flow_timestamp - curr_flow_timestamp) * (1e-9)
        
        Delta_R_b_g, Delta_v_b_g, Delta_t_b_g = slam.data.get_preintegrated_measurement(curr_flow_timestamp, next_flow_timestamp)
        Delta_R_b_g = Delta_R_b_g.astype(np.float32).reshape(3,3)
        Delta_t_b_g = Delta_t_b_g.reshape(3,1).astype(np.float32).reshape(3,1)
        
        
        R_g_b = R_b0_g.transpose()
        v_gb_b = R_g_b @ v_gb_g.reshape(3,1)
        kinematic_state = np.concatenate((R_g_b, v_gb_b, T_c_b_, Delta_R_b_g, Delta_t_b_g), axis=1)
        kinematic_state = kinematic_state.transpose()  

        # Get the depth map and original image to feed into voldor in order to debug
        curr_timestamp = slam.data.FlowList.getItemID(flow_idx)
        next_timestamp = slam.data.FlowList.getItemID(flow_idx+1)
        path_to_depth_map = slam.data.DepthList.getItemPathFromTimeStamp(curr_timestamp)
        depth_map = my_utils.read_pfm(path_to_depth_map)
        
        
        path_to_curr_image = slam.data.ImageList.getItemPathFromTimeStamp(curr_timestamp)
        curr_gray_img_frame = cv2.imread(path_to_curr_image, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        path_to_next_image = slam.data.ImageList.getItemPathFromTimeStamp(next_timestamp)
        next_gray_img_frame = cv2.imread(path_to_next_image, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        img_arr = [curr_gray_img_frame, next_gray_img_frame]
        py_voldor_kwargs = {
            'flows': np.stack(flows, axis=0),
            'img_frame': np.stack(img_arr, axis=0),
            'kinematic_state': kinematic_state,
            'fx':slam.cam_model.fx, 'fy':slam.cam_model.fy, 'cx':slam.cam_model.cx, 'cy':slam.cam_model.cy, 'basefocal':1.0,
            'disparity' : None,
            'depth_priors' : np.stack(depth_priors, axis=0) if len(depth_priors)>0 else None,
            'depth_prior_pconfs' : np.stack(depth_prior_pconfs, axis=0) if len(depth_prior_pconfs)>0 else None,
            'depth_prior_poses' : np.stack(prior_poses, axis=0) if len(prior_poses)>0 else None,
            'config' : slam.voldor_config + ' ' + slam.voldor_user_config}
        
        py_voldor_funmap = partial(pyvoldor.voldor, **py_voldor_kwargs)
        vo_ret = slam.cython_process_pool.apply(py_voldor_funmap)
        
        estimated_depth_map = vo_ret['depth']
        innovation_term = vo_ret['innovation']
        print(innovation_term)
        # cv2.imshow("gray_img_frame",gray_img_frame.astype(np.uint8))
        # cv2.waitKey()
        
        print("-------------------")
            
        ############ VISUALIZATION ############
        visualize = False
        if visualize:
            path_to_flow = slam.data.FlowList.getItemPath(flow_idx)
            curr_flow = load_flow(path_to_flow)
            curr_timestamp = slam.data.FlowList.getItemID(flow_idx)

            path_to_depth_map = slam.data.DepthList.getItemPathFromTimeStamp(curr_timestamp)
            depth_map = my_utils.read_pfm(path_to_depth_map)
            
            path_to_image = slam.data.ImageList.getItemPathFromTimeStamp(curr_timestamp)
            original_image = cv2.imread(path_to_image)
            
            mask = vo_ret['depth_conf']>slam.depth_scaling_conf_thresh
            error = depth_map - estimated_depth_map
            weighted_error = error / depth_map
            
            ########### HISTOGRAM FOR ERROR IN DEPTH ESTIMATION ###########
            error_histogram = cv2.calcHist([error], [0], None, [20], [-10, 10])
            error_histogram = error_histogram / (512*640)
            
            # Plot the histogram
            x_axis = np.linspace(-10, 10, num=20)
            plt.figure()
            plt.title("PDF of Error in Depth Estimation")
            plt.xlabel("Error (meter)")
            plt.plot(x_axis, error_histogram)
            plt.xlim([-10, 10])
            plt.ylim([0, 1.0])
            
            # Convert the plot to an OpenCV image
            fig = plt.gcf()  # Get the current figure
            fig.canvas.draw()  # Render the figure
            error_histogram_fig = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            error_histogram_fig = error_histogram_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            error_histogram_fig = cv2.cvtColor(error_histogram_fig, cv2.COLOR_RGB2BGR).astype(np.uint8)
            plt.cla()

            ########### HISTOGRAM FOR PERCENTAGE ERROR IN DEPTH ESTIMATION ###########
            error_histogram = cv2.calcHist([weighted_error], [0], None, [20], [-10, 10])
            error_histogram = error_histogram / (512*640)
            
            # Plot the histogram
            x_axis = np.linspace(-10, 10, num=20)
            plt.close()
            plt.figure()
            plt.title("PDF of Percentage Error in Depth Estimation")
            plt.xlabel("Error Percentage")
            plt.plot(x_axis, error_histogram)
            plt.xlim([-10, 10])
            plt.ylim([0, 1.0])
            
            # Convert the plot to an OpenCV image
            fig = plt.gcf()  # Get the current figure
            fig.canvas.draw()  # Render the figure
            percentage_error_histogram_fig = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            percentage_error_histogram_fig = percentage_error_histogram_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            percentage_error_histogram_fig = cv2.cvtColor(percentage_error_histogram_fig, cv2.COLOR_RGB2BGR).astype(np.uint8)
            plt.cla()
            
            
            
            
            error_vis = cv2.normalize(error, None,
                                    alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

            flow_bgr = raft_utils.flow_to_image(curr_flow)
            
            normalizer = 100 / 255
            
            depth_map_vis = depth_map.copy()
            out_region = np.where(depth_map_vis > 100)
            depth_map_vis[out_region] = 100
            depth_map_vis = (depth_map_vis / normalizer).astype(np.uint8)
            
            estimated_depth_map_vis = estimated_depth_map.copy()
            out_region = np.where(estimated_depth_map_vis > 100)
            estimated_depth_map_vis[out_region] = 100
            estimated_depth_map_vis = (estimated_depth_map_vis / normalizer).astype(np.uint8)
            
            mask = vo_ret['depth_conf']>slam.depth_scaling_conf_thresh
            mask = mask.astype(np.uint8)*255
            
            depth_map_vis = cv2.cvtColor(depth_map_vis, cv2.COLOR_GRAY2BGR).astype(np.uint8)
            estimated_depth_map_vis = cv2.cvtColor(estimated_depth_map_vis, cv2.COLOR_GRAY2BGR).astype(np.uint8)
            error_histogram_fig = cv2.resize(error_histogram_fig, (640,512)) 
            percentage_error_histogram_fig = cv2.resize(percentage_error_histogram_fig, (640,512)) 
            
            # print(depth_map_vis.shape)
            # print(estimated_depth_map_vis.shape)
            # print(error_histogram_fig.shape)
            
            depth_maps = cv2.hconcat([depth_map_vis, estimated_depth_map_vis])
            histograms = cv2.hconcat([error_histogram_fig, percentage_error_histogram_fig])
            imgs = cv2.hconcat([original_image, flow_bgr])

            out_img = cv2.vconcat([imgs, depth_maps, histograms])
            # saving_path = os.path.join("DepthAnalysis2", str(curr_flow_timestamp) + ".png")
            # cv2.imwrite(saving_path, out_img)

            # Resize the image
            scale = 0.6
            new_size = (int(out_img.shape[1]*scale), int(out_img.shape[0]*scale))
            out_img = cv2.resize(out_img, new_size)
            cv2.imshow("out_img", out_img)

            key =cv2.waitKey(1)
            if key == ord("q"):
                break

