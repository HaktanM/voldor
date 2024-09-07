import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import os

import my_utils
import raft_utils
import warp_utils

path_to_data = "/home/hakito/python_scripts/AirSim/Data1"
data = my_utils.dataHandler(path_to_data)

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

if __name__ == '__main__':
    # init slam instance and select mode from mono/mono-scaled/stereo
    slam = VOLDOR_SLAM(mode='mono', path_to_data = "/home/hakito/python_scripts/AirSim/Data1")
    
    num_of_flows = slam.data.FlowList.itemCount()


    for flow_idx in range(100, num_of_flows-slam.voldor_winsize-1):
        
        # Load the flow      
        win_idx = 0
        path_to_flow = slam.data.FlowList.getItemPath(flow_idx + win_idx)
        curr_flow = load_flow(path_to_flow)
        
        # Get the timestamp
        curr_flow_timestamp = slam.data.FlowList.getItemID(flow_idx + win_idx)
        next_flow_timestamp = slam.data.FlowList.getItemID(flow_idx + win_idx + 1)
        
        print(curr_flow_timestamp)
        
        
        dT = (next_flow_timestamp - curr_flow_timestamp) * (1e-9)
        
        # Load the corresponding depth map
        path_to_depth_map = slam.data.DepthList.getItemPathFromTimeStamp(curr_flow_timestamp)
        depth_map = my_utils.read_pfm(path_to_depth_map)
        
        # Current GT
        curr_flow_timestamp = slam.data.FlowList.getItemID(flow_idx + win_idx)
        q_b0_g, v_b0_g, t_b0_g = slam.data.get_gt(curr_flow_timestamp) 
        R_b0_g = R.from_quat(q_b0_g).as_matrix().reshape(3,3)
        R_g_b0 = R_b0_g.transpose()
        
        # Get preintegrated IMU measurements
        Delta_R_b_g, Delta_v_b_g, Delta_t_b_g = slam.data.get_preintegrated_measurement(curr_flow_timestamp, next_flow_timestamp)

        import time 
        t_start = time.monotonic()
        estimated_flow = slam.cam_model.get_estimated_OF(depth_map=depth_map, Delta_R=Delta_R_b_g, Delta_v=Delta_v_b_g, Delta_t=Delta_t_b_g, R_bk_g=R_b0_g, v_bk_g=v_b0_g, dT=dT, gravity_mag=9.80665016174316)
        t_stop  = time.monotonic()
        t_elapsed = (t_stop - t_start)
        fps = 1 / t_elapsed
        print(f"FPS for getting estimated optical flow : {fps}")
        
        print(estimated_flow[100,100])
        break
        ############ VISUALIZATION ############
        path_to_flow = slam.data.FlowList.getItemPath(flow_idx)
        curr_flow = load_flow(path_to_flow)
        
        curr_timestamp = slam.data.FlowList.getItemID(flow_idx)
        next_timestamp = slam.data.FlowList.getItemID(flow_idx+1)

        
        # Get the current image
        path_to_curr_image = slam.data.ImageList.getItemPathFromTimeStamp(curr_timestamp)
        curr_img = cv2.imread(path_to_curr_image)
        curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Get the next image
        path_to_next_image = slam.data.ImageList.getItemPathFromTimeStamp(next_timestamp)
        next_img = cv2.imread(path_to_next_image)
        next_img_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        
        ############ RAFT OPTICAL FLOW ############
        # Get the warping error
        curr_flow = curr_flow.astype(np.float32)
        warped_img_raft = warp_utils.warp_flow(curr_img_gray, curr_flow).astype(np.float32)
        warping_err_raft = next_img_gray - warped_img_raft

        # Get the warping mask
        mask = np.ones_like(next_img_gray)
        warped_mask_raft = warp_utils.warp_flow(mask, curr_flow).astype(np.float32)
        
        # Get the masked warping error
        masked_warping_error_raft = warped_mask_raft * warping_err_raft
        abs_warping_err_raft = np.abs(masked_warping_error_raft)
        abs_warping_err_raft_vis = cv2.cvtColor(abs_warping_err_raft, cv2.COLOR_GRAY2BGR).astype(np.uint8)

        # Compute the histogram
        histogram_raft = cv2.calcHist([masked_warping_error_raft], [0], None, [100], [-50, 50])
        histogram_raft = histogram_raft / (512*640)
        
        # Plot the histogram
        plt.figure()
        plt.title("Raft OF and Warping Error")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.plot(histogram_raft)
        plt.xlim([0, 100])
        plt.ylim([0, 0.25])
        
        # Convert the plot to an OpenCV image
        fig = plt.gcf()  # Get the current figure
        fig.canvas.draw()  # Render the figure
        histogram_raft_fig = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        histogram_raft_fig = histogram_raft_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        histogram_raft_fig = cv2.cvtColor(histogram_raft_fig, cv2.COLOR_RGB2BGR)
        plt.cla()
        
        ############ ESTIMATED OPTICAL FLOW ############
        # Get the warping error
        estimated_flow = estimated_flow.astype(np.float32)
        warped_img_est = warp_utils.warp_flow(curr_img_gray, estimated_flow).astype(np.float32)
        warping_err_est = next_img_gray - warped_img_est

        # Get the warping mask
        mask = np.ones_like(next_img_gray)
        warped_mask_est = warp_utils.warp_flow(mask, estimated_flow).astype(np.float32)
        
        # Get the masked warping error
        masked_warping_error_est = warped_mask_est * warping_err_est
        abs_warping_err_est = np.abs(masked_warping_error_est)
        abs_warping_err_est_vis = cv2.cvtColor(abs_warping_err_est, cv2.COLOR_GRAY2BGR).astype(np.uint8)

        # Compute the histogram
        histogram_est = cv2.calcHist([masked_warping_error_est], [0], None, [100], [-50, 50])
        histogram_est = histogram_est / (512*640)
        
        # Plot the histogram
        plt.figure()
        plt.title("Estimated OF and Warping Error")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.plot(histogram_est)
        plt.xlim([0, 100])
        plt.ylim([0, 0.25])
        
        # Convert the plot to an OpenCV image
        fig = plt.gcf()  # Get the current figure
        fig.canvas.draw()  # Render the figure
        histogram_est_fig = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        histogram_est_fig = histogram_est_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        histogram_est_fig = cv2.cvtColor(histogram_est_fig, cv2.COLOR_RGB2BGR)

        depth_map_vis = cv2.normalize(depth_map, None,
                                alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

        flow_bgr = raft_utils.flow_to_image(curr_flow)
        
        estimated_flow_bgr = raft_utils.flow_to_image(estimated_flow)
        
        flows = cv2.hconcat([flow_bgr, estimated_flow_bgr])
        warping_errors = cv2.hconcat([abs_warping_err_raft_vis, abs_warping_err_est_vis])
        err_histograms = cv2.hconcat([histogram_raft_fig, histogram_est_fig])
        
        out_img = cv2.vconcat([err_histograms, warping_errors, flows])
        
        # saving_path = os.path.join("OF_Analysis", str(curr_flow_timestamp) + ".png")
        # cv2.imwrite(saving_path, out_img)

        # cv2.imshow("depth_map_vis", depth_map_vis)
        # cv2.imshow("original_image", curr_img)
        cv2.imshow("estimated_flow_bgr", estimated_flow_bgr)

        key =cv2.waitKey()
        if key == ord("q"):
            break