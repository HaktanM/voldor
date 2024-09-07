from scipy.spatial.transform import Rotation as R
import numpy as np
import os

import my_utils
import inertial_propagator
import visualization_utils

main_path = "/home/hakito/python_scripts/AirSim/Data1"
data = my_utils.dataHandler(main_path)

path_to_imu = os.path.join(main_path, "preintegrated_imu.csv")
path_to_gt = os.path.join(main_path, "synched_gt.csv")

# Load the gt and imu data
imu_data = np.loadtxt( path_to_imu,  delimiter=",",  comments="#" ).reshape(-1,16)
gt_data = np.loadtxt(  path_to_gt,   delimiter=",",  comments="#" ).reshape(-1,16)


# Determine the data interval to process
start_idx = 50
stop_idx  = start_idx + 100


# Initialize the state
timestamp = gt_data[start_idx,0]
q_b_g_gt, v_gb_g_gt, t_gb_g_gt = data.get_gt(timestamp)
R_b_g_gt = R.from_quat(q_b_g_gt).as_matrix()
state = inertial_propagator.State(timestamp=timestamp, R_b_g=R_b_g_gt, v_gb_g=v_gb_g_gt, t_gb_g=t_gb_g_gt)

# Allocate memory to save estimated states
estimated_states = np.zeros((stop_idx-start_idx,11))

# Allocate memory to save synchronized gt
sync_gt = np.zeros((stop_idx-start_idx,11))


# Save the initialization point
estimated_states[0,0] = timestamp
estimated_states[0,1:5] = q_b_g_gt.reshape(4)
estimated_states[0,5:8] = v_gb_g_gt.reshape(3)
estimated_states[0,8:11] = t_gb_g_gt.reshape(3)

sync_gt[0,0] = timestamp
sync_gt[0,1:5] = q_b_g_gt.reshape(4)
sync_gt[0,5:8] = v_gb_g_gt.reshape(3)
sync_gt[0,8:11] = t_gb_g_gt.reshape(3)

timestamp_init = timestamp
for idx in range(start_idx, stop_idx - 1):
    curr_timestamp = imu_data[idx, 0]
    next_timestamp = imu_data[idx+1, 0]
    
    delta_R = imu_data[idx,1:10].reshape(3,3)
    delta_v = imu_data[idx,10:13].reshape(3,1)
    delta_t = imu_data[idx,13:16].reshape(3,1)
    
    
    # Propagate the state using preintegrated IMU measurement
    state.propagate_using_preintegtrated_imu(next_timestamp,Delta_R=delta_R, Delta_v=delta_v,Delta_t=delta_t)
    
    
    # Save the estimated state       
    next_R_b_e_est = state.T[:3,:3]
    next_q_b_e_est = R.from_matrix(next_R_b_e_est).as_quat()
    next_v_eb_e_es = state.T[:3,3].reshape(3)
    next_t_eb_e_es = state.T[:3,4].reshape(3)
    
    estimated_states[idx+1-start_idx, 0] = next_timestamp
    estimated_states[idx+1-start_idx, 1:5]  =  next_q_b_e_est.reshape(4)
    estimated_states[idx+1-start_idx, 5:8]  =  next_v_eb_e_es.reshape(3)
    estimated_states[idx+1-start_idx, 8:11] =  next_t_eb_e_es.reshape(3)
    
    # Save the synchronized gt
    R_b_g_gt = gt_data[idx,1:10].reshape(3,3)
    q_b_g_gt = R.from_matrix(R_b_g_gt).as_quat()
    v_gb_g_gt = gt_data[idx,10:13].reshape(3,1)
    t_gb_g_gt = gt_data[idx,13:16].reshape(3,1)

    sync_gt[idx+1-start_idx,0] = next_timestamp
    sync_gt[idx+1-start_idx,1:5] = q_b_g_gt.reshape(4)
    sync_gt[idx+1-start_idx,5:8] = v_gb_g_gt.reshape(3)
    sync_gt[idx+1-start_idx,8:11] = t_gb_g_gt.reshape(3)
    
    print(v_gb_g_gt)
    
gt_traj = sync_gt[:, [0, 8, 9, 10]]
es_traj = estimated_states[:, [0, 8, 9, 10]]


_vis = visualization_utils.traj_visualizer()
_vis.add_gt(gt_traj)
_vis.add_es(es_traj)
_vis.ax1.legend(["Ground-Truth", "Estimated Traj"])
_vis.ax2.legend(["Ground-Truth", "Estimated Traj"])
_vis.show()


errors = inertial_propagator.compute_err(gt=sync_gt,es=estimated_states)
err_plot = visualization_utils.err_vis()
err_plot.add_err_plt(errors)
err_plot.ax[0,0].legend(["Errors"])
err_plot.show()

    