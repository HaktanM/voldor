from scipy.spatial.transform import Rotation as R
import numpy as np

import my_utils
import inertial_propagator
import visualization_utils

main_path = "/home/hakito/python_scripts/AirSim/Data1"
data = my_utils.dataHandler(main_path)


# Determine the data interval to process
start_idx = 50
stop_idx  = data.FlowList.itemCount() - 1

# Initialize the state
timestamp = data.FlowList.getItemID(start_idx)
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
    curr_timestamp = data.FlowList.getItemID(idx)
    next_timestamp = data.FlowList.getItemID(idx+1)
    imu_readings = data.select_imu_readings(curr_timestamp,next_timestamp)

    for imu_idx in range(imu_readings.shape[0]):
        imu_timestamp = imu_readings[imu_idx,0]
        gyro_meas = imu_readings[imu_idx,1:4]
        acce_meas = imu_readings[imu_idx,4:7]
        state.propagate(timestamp=imu_timestamp, gyr_data=gyro_meas, acc_data=acce_meas)
        
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
    q_b_g_gt, v_gb_g_gt, t_gb_g_gt = data.get_gt(next_timestamp)
    sync_gt[idx+1-start_idx,0] = next_timestamp
    sync_gt[idx+1-start_idx,1:5] = q_b_g_gt.reshape(4)
    sync_gt[idx+1-start_idx,5:8] = v_gb_g_gt.reshape(3)
    sync_gt[idx+1-start_idx,8:11] = t_gb_g_gt.reshape(3)
    
    
gt_traj = sync_gt[:, [0, 8, 9, 10]]
es_traj = estimated_states[:, [0, 8, 9, 10]]


_vis = visualization_utils.traj_visualizer()
_vis.add_gt(gt_traj)
_vis.add_es(es_traj)
_vis.ax1.legend(["Ground-Truth", "Estimated Traj"])
# _vis.ax2.legend(["Ground-Truth", "Estimated Traj"])
_vis.show()


errors = inertial_propagator.compute_err(gt=sync_gt,es=estimated_states)
err_plot = visualization_utils.err_vis()
err_plot.add_err_plt(errors)
err_plot.ax[0,0].legend(["Errors"])
err_plot.show()

    