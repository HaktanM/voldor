from scipy.spatial.transform import Rotation as R
import numpy as np

import my_utils
import inertial_propagator
import visualization_utils

main_path = "/home/hakito/python_scripts/AirSim/Data1"
data = my_utils.dataHandler(main_path)


# Determine the data interval to process
start_idx = 1000
stop_idx  = data.imu.shape[0] - 100
es_times = data.imu_times[start_idx:stop_idx]

# Get Synchronized GT
sync_gt = inertial_propagator.get_sync_gt(es_times=es_times, gt=data.gt)


# Initialize the state
timestamp = sync_gt[0,0]
q_b_g_gt  = sync_gt[0,1:5]
v_gb_g_gt = sync_gt[0,5:8] 
t_gb_g_gt = sync_gt[0,8:11]
R_b_g_gt = R.from_quat(q_b_g_gt).as_matrix()

# Allocate memory to save estimated states
estimated_states = np.zeros((stop_idx-start_idx,11))

# Save the initialization point
estimated_states[0,0] = timestamp
estimated_states[0,1:5] = q_b_g_gt.reshape(4)
estimated_states[0,5:8] = v_gb_g_gt.reshape(3)
estimated_states[0,8:11] = t_gb_g_gt.reshape(3)


state = inertial_propagator.State(timestamp=timestamp, R_b_g=R_b_g_gt, v_gb_g=v_gb_g_gt, t_gb_g=t_gb_g_gt)
print(state.T[:3,3])
print(state.T[:3,4])

# import sys
# sys.exit()
for idx in range(start_idx, stop_idx - 1):
    curr_timestamp, gyro_meas, acce_meas = data.get_imu_from_idx(idx)
    next_timestamp, _, _ = data.get_imu_from_idx(idx+1)
    dt = (next_timestamp - curr_timestamp) * (1e-9)
    
    state.propagate(timestamp=curr_timestamp, gyr_data=gyro_meas, acc_data=acce_meas)
    
    # Save the estimated state       
    curr_R_b_e_est = state.T[:3,:3]
    curr_q_b_e_est = R.from_matrix(curr_R_b_e_est).as_quat()
    curr_v_eb_e_es = state.T[:3,3].reshape(3)
    curr_t_eb_e_es = state.T[:3,4].reshape(3)
    
    estimated_states[idx+1-start_idx, 0] = curr_timestamp
    estimated_states[idx+1-start_idx, 1:5] = curr_q_b_e_est.reshape(4)
    estimated_states[idx+1-start_idx, 5:8] = curr_v_eb_e_es.reshape(3)
    estimated_states[idx+1-start_idx, 8:11] = curr_t_eb_e_es.reshape(3)


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
