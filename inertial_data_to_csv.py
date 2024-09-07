from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import my_utils
import inertial_propagator
import visualization_utils

main_path = "/home/hakito/python_scripts/AirSim/Data1"
data = my_utils.dataHandler(main_path)

# Create Files to Save Result
path_to_save = os.path.join(main_path,"preintegrated_imu.csv")
with open(path_to_save, "w") as imu_file:
    imu_file.write("#timestamp, Delta_R (9 params), Delta_v (3 params), Delta_t (3 params), DeltaT (1 param)\n")
imu_file = open(path_to_save, "a")

path_to_save = os.path.join(main_path, "synched_gt.csv")
with open(path_to_save, "w") as gt_file:
    gt_file.write("#timestamp, R_b_g (9 params), v_gb_g (3 params), t_gb_g (3 params)\n")
gt_file = open(path_to_save, "a")

# Determine the data interval to process
start_idx = 1
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

v_gb_g_gt = v_gb_g_gt.reshape(3,1)
t_gb_g_gt = t_gb_g_gt.reshape(3,1)

# Write the current gt into file
gt_file.write(f"{int(timestamp)},")

for row in range(3):
    for col in range(3):
        gt_file.write(f"{R_b_g_gt[row, col]: .10f},")

for row in range(3):
    gt_file.write(f"{v_gb_g_gt[row, 0]: .5f},")

for row in range(2):
    gt_file.write(f"{t_gb_g_gt[row, 0]: .5f},")

gt_file.write(f"{t_gb_g_gt[2, 0]: .5f}\n")

timestamp_init = timestamp
for idx in range(start_idx, stop_idx - 1):
    curr_timestamp = data.FlowList.getItemID(idx)
    next_timestamp = data.FlowList.getItemID(idx+1)
    imu_readings = data.select_imu_readings_start_synched(curr_timestamp,next_timestamp)

    
    ##################################################################################################################################
    ########################################### COMPUTE THE PREINTEGRATED IMU MEASUREMENT ############################################
    ##################################################################################################################################
    delta_R = np.eye(3)
    delta_v = np.zeros((3,1))
    delta_t = np.zeros((3,1))
    for imu_idx in range(imu_readings.shape[0]-1):
        imu_timestamp = imu_readings[imu_idx,0]
        next_imu_timestamp = imu_readings[imu_idx+1,0]
        dt = (next_imu_timestamp - imu_timestamp) * (1e-9)
        
        gyro_meas = imu_readings[imu_idx,1:4].reshape(3,1)
        acce_meas = imu_readings[imu_idx,4:7].reshape(3,1)
        
        delta_R = delta_R @ state.LA.exp_SO3(gyro_meas*dt)
        delta_v = delta_v.reshape(3,1) + (state.LA.integral_exp_SO3(gyro_meas, dt) @ acce_meas).reshape(3,1)
        delta_t = delta_t.reshape(3,1) + (state.LA.double_integral_exp_SO3(gyro_meas, dt) @ acce_meas).reshape(3,1)

    # Also use the last IMU measurement to propagate the state
    imu_timestamp = imu_readings[-1,0]
    next_imu_timestamp = next_timestamp
    dt = (next_imu_timestamp - imu_timestamp) * (1e-9)
    
    delta_R = delta_R @ state.LA.exp_SO3(gyro_meas*dt)
    delta_v = delta_v.reshape(3,1) + (state.LA.integral_exp_SO3(gyro_meas, dt) @ acce_meas).reshape(3,1)
    delta_t = delta_t.reshape(3,1) + (state.LA.double_integral_exp_SO3(gyro_meas, dt) @ acce_meas).reshape(3,1)
    
    # Apped to our txt
    imu_file.write(f"{int(curr_timestamp)},")
    
    for row in range(3):
        for col in range(3):
            imu_file.write(f"{delta_R[row, col]: .10f},")
    
    for row in range(3):
        imu_file.write(f"{delta_v[row, 0]: .5f},")
    
    for row in range(2):
        imu_file.write(f"{delta_t[row, 0]: .5f},")
    imu_file.write(f"{delta_t[2, 0]: .5f}\n")
    
    gyro_meas = imu_readings[-1,1:4].reshape(3,1)
    acce_meas = imu_readings[-1,4:7].reshape(3,1)

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
    q_b_g_gt, v_gb_g_gt, t_gb_g_gt = data.get_gt(next_timestamp)
    R_b_g_gt = R.from_quat(q_b_g_gt).as_matrix()
    v_gb_g_gt = v_gb_g_gt.reshape(3,1)
    t_gb_g_gt = t_gb_g_gt.reshape(3,1)
    
    gt_file.write(f"{int(next_timestamp)},")

    for row in range(3):
        for col in range(3):
            gt_file.write(f"{R_b_g_gt[row, col]: .10f},")
    
    for row in range(3):
        gt_file.write(f"{v_gb_g_gt[row, 0]: .5f},")
    
    for row in range(2):
        gt_file.write(f"{t_gb_g_gt[row, 0]: .5f},")
    
    gt_file.write(f"{t_gb_g_gt[2, 0]: .5f}\n")
    
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

    