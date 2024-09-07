import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import re

class _CameraModel:
    def __init__(self) -> None:
        
        # Intrinsic Calibration
        self.w = 640
        self.h = 512
        self.fx = 320
        self.fy = 320
        self.cx = 320
        self.cy = 256
        
        self.K = np.array([
            self.fx, 0.0, self.cx,
            0.0, self.fy, self.cy,
            0.0, 0.0, 1.0
        ]).reshape(3,3)
        self.K_inv = np.linalg.inv(self.K).reshape(3,3)
        
        # Extrinsic Calibration
        R_cam_gim = np.array([
            0.0, 0.0, 1.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0
        ]).reshape(3,3)
        
        R_gim_body = R.from_euler('y', -90.0, degrees=True).as_matrix()
        self.R_c_b = R_gim_body @ R_cam_gim
        self.t_c_b = np.array([0.0, 0.5, 0.0]).reshape(3,1)
        self.R_b_c = self.R_c_b.transpose()
        self.t_b_c = - ( self.R_b_c @ self.t_c_b.reshape(3,1) ).reshape(3,1)
        
        self.T_b_c = np.eye(4, dtype=np.float32)
        self.T_b_c[:3,:3] = self.R_b_c
        self.T_b_c[:3,3] = self.t_b_c.reshape(3)
        
        self.T_c_b = np.eye(4, dtype=np.float32)
        self.T_c_b[:3,:3] = self.R_c_b
        self.T_c_b[:3,3] = self.t_c_b.reshape(3)
        
        self.pixels = self.getPixelMat().reshape(3,self.w*self.h)
        self.bearings = (self.K_inv @ self.pixels).reshape(3,self.w*self.h)

    def getPixelMat(self):
        pixel_math = np.ones((3,self.w*self.h))
        for y in range(self.h):
            for x in range(self.w):
                curr_idx = (y*self.w + x)
                pixel_math[0,curr_idx] = x + 0.5
                pixel_math[1,curr_idx] = y + 0.5
        return pixel_math
    
    def get_estimated_OF(self, depth_map, Delta_R, Delta_v, Delta_t, R_bk_g, v_bk_g, dT, gravity_mag=9.80665016174316):
        gravity = np.array([0,0,1]).reshape(3,1) * gravity_mag
        R_g_bk = R_bk_g.transpose()
        v_bk_g = v_bk_g.reshape(3,1)
        
        # print(R_bk_g)
        # print(v_bk_g)
        # print(Delta_R)
        # print(Delta_t)

        
        # Compute the incremental pose for body frame using IMU measurements
        # Orientation
        R_bn_bc = Delta_R
        R_bc_bn = R_bn_bc.transpose()
        
        # Translation
        t_bn_bc = Delta_t.reshape(3,1)   +   R_g_bk @ v_bk_g * dT   +   (0.5 * (dT**2)) * R_g_bk @ gravity
        t_bn_bc = t_bn_bc.reshape(3,1)
        # t_bc_bn = - R_bc_bn @ t_bn_bc
        # t_bc_bn = t_bc_bn.reshape(3,1)
        
        # Express the iterative pose in camera frame
        # Orientation
        R_cc_cn = self.R_b_c @ R_bc_bn @ self.R_c_b
        
        # Translation
        t_cn_cc = self.R_b_c @ (t_bn_bc + (R_bn_bc - np.eye(3))@self.t_c_b).reshape(3,1)
        t_cn_cc = t_cn_cc.reshape(3,1)
        t_cc_cn = - R_cc_cn @ t_cn_cc
        
        # Compute omega
        w = self.K @ R_cc_cn @ self.bearings
        
        # print(self.K @ R_cc_cn)
        # print(self.bearings[:,100:105])
        # print(w[:,100:105])
        
        
        # Compute b,    --->   b term is the same for all pixels
        b = self.K @ t_cc_cn
        print(w[:,13300:13300+5])
        print(b.reshape(3,1))

        b = np.tile(b, (1, w.shape[1]))
        
            
        
        
        # Get the depth map
        depth_reshaped = depth_map.reshape(1,self.h*self.w) 
        print(depth_reshaped[:,13300:13300+5])

        depth_reshaped = np.tile(depth_reshaped, (3, 1))
        
        # Elementary-wise multiplication
        w_depth = w * depth_reshaped
        
        

        # Compute the propagated pixels
        propagated_pixels = w_depth + b
        print(propagated_pixels[:,13300:13300+5])
        propagated_pixels = propagated_pixels / propagated_pixels[2,:]
        
        # Compute the optical flow
        estimated_optical_flow = propagated_pixels - self.pixels
        
        # Reshape the optical flow into proper format
        estimated_optical_flow_x = estimated_optical_flow[0,:]
        estimated_optical_flow_y = estimated_optical_flow[1,:]
        
        print(self.pixels[:,13300:13300+5])
        print(propagated_pixels[:,13300:13300+5])
        print(estimated_optical_flow[:,13300:13300+5])
        
        estimated_optical_flow_x = estimated_optical_flow_x.reshape(self.h, self.w) 
        estimated_optical_flow_y = estimated_optical_flow_y.reshape(self.h, self.w)
        
        
        
        estimated_flow = np.zeros((self.h, self.w, 2))
        estimated_flow[:,:,0] = estimated_optical_flow_x
        estimated_flow[:,:,1] = estimated_optical_flow_y
        
        
        return estimated_flow
        
        
        
        
         
def skew_symmetric(vec):
    vec = vec.reshape(3,1)
    Vec = np.array([
        0.0, -vec[2,0], vec[1,0],
        vec[2,0], 0.0, -vec[0,0],
        -vec[1,0], vec[0,0], 0.0
    ]).reshape(3,3)
    return Vec
    
class LieUtils():
    def __init__(self) -> None:
        self.tolerance = 1e-12
        
    def exp_SO3(self, vec):
        vec = vec.reshape(3,1)
        angle = np.linalg.norm(vec)
        
        unit_vec = np.zeros((3,1))
        if angle > self.tolerance:
            unit_vec = vec / angle
            
        unit_norm_skew = skew_symmetric(unit_vec).reshape(3,3)
        R = np.eye(3) + np.sin(angle) * unit_norm_skew + (1-np.cos(angle))*(unit_norm_skew@unit_norm_skew).reshape(3,3)
        return R.reshape(3,3)
    
    def RightJacob_SO3(self, vec):
        vec = vec.reshape(3,1)
        angle = np.linalg.norm(vec)
        
        unit_vec = np.zeros((3,1))
        if angle > self.tolerance:
            unit_vec = vec / angle
            
        unit_norm_skew = skew_symmetric(unit_vec).reshape(3,3)
        
        if angle < self.tolerance:
            m1 = - 0.5
            m2 = 1.0 / 6.0
        else:
            m1 = - (1 - np.cos(angle)) / (angle**2)
            m2 = (angle - np.sin(angle)) / (angle**3)
            
        R = np.eye(3) + m1 * unit_norm_skew + m2 * unit_norm_skew @ unit_norm_skew
        return R.reshape(3,3)
        
        
    def integral_exp_SO3(self, vec, dt):
        # It is assumed that the integral limits are 0 and dt
        vec = vec.reshape(3,1)
        angle = np.linalg.norm(vec)
        
        unit_vec = np.zeros((3,1))
        if angle > self.tolerance:
            unit_vec = vec / angle
            
        unit_norm_skew = skew_symmetric(unit_vec).reshape(3,3)
        
        angle_dt = angle * dt
        
        if angle < self.tolerance:
            m1 = np.sin(angle_dt)
            m2 = (1.0 - np.cos(angle_dt)) * dt
        else:
            m1 = (1.0 - np.cos(angle_dt)) / angle
            m2 = dt - np.sin(angle_dt) / angle
            
        R = np.eye(3) * dt + m1 * unit_norm_skew + m2*(unit_norm_skew@unit_norm_skew).reshape(3,3)
        return R.reshape(3,3)

    def double_integral_exp_SO3(self, vec, dt):
        # It is assumed that the integral limits are 0 and dt
        vec = vec.reshape(3,1)
        angle = np.linalg.norm(vec)

        unit_vec = np.zeros((3,1))
        if angle > self.tolerance:
            unit_vec = vec / angle
            
        unit_norm_skew = skew_symmetric(unit_vec).reshape(3,3)
        
        angle_dt = angle * dt
        angle_square = angle**2
        dt_square = dt**2
        
        if angle < self.tolerance:
            m1 = 0.5 * dt_square * np.sin(angle_dt)
            m2 = 0.5 * dt_square * (1 - np.cos(angle_dt))
        else:
            m1 = (angle_dt - np.sin(angle_dt)) / angle_square
            m2 = 0.5*dt_square - ( (1.0 - np.cos(angle_dt)) / angle_square )
            
        R = 0.5 * np.eye(3) * dt_square + m1 * unit_norm_skew + m2*(unit_norm_skew@unit_norm_skew).reshape(3,3)
        return R.reshape(3,3)

        
class itemList:
    def __init__(self, path) -> None:
        self.path = path
        folder_items = os.listdir(path)
        folder_items.sort()
        first_item = folder_items[0]
        second_item = folder_items[1]
        self.extension = first_item.split(".")[-1]
        self.extension_length = len(self.extension) + 1
        
        # Check if we have zero padding in the namming
        self.zero_padding = False
        self.padding_length = 0
        if first_item[0] == "0" and second_item[0] == "0": 
            self.zero_padding = True
            self.padding_length = len(first_item[:-self.extension_length])

        self.item_ids = []
        for item in folder_items:
            try:
                item_id = int(item[:-self.extension_length])
                self.item_ids.append(item_id)
            except:
                print(f"Item ({item[:-self.extension_length]}) cannot be converted to int. It will be discarded.")

        self.item_ids.sort()
        
    def getItemPath(self,idx:int):
        if self.zero_padding:
            path = os.path.join(self.path, str(self.item_ids[idx]).zfill(self.padding_length)+"." + self.extension)
        else:
            path = os.path.join(self.path, str(self.item_ids[idx])+"." + self.extension)
        return path
    
    def getItemPathFromTimeStamp(self,timestamp):
        path = os.path.join(self.path, str(timestamp) + "." + self.extension)
        return path
    
    def getItemID(self,idx):
        return self.item_ids[idx]
    
    def getItemName(self,idx):
        if self.zero_padding:
            name = str(self.item_ids[idx]).zfill(self.padding_length)
        else:
            name = str(self.item_ids[idx])
        return name

    def itemCount(self):
        return len(self.item_ids)
    
    
def read_pfm(file):
    with open(file, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise ValueError('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise ValueError('Malformed PFM header.')

        scale = float(f.readline().decode('utf-8').rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)        
    return data

def load_flow(path):
    with open(path, 'rb') as f:
        magic = float(np.fromfile(f, np.float32, count = 1)[0])
        if magic == 202021.25:
            w, h = np.fromfile(f, np.int32, count = 1)[0], np.fromfile(f, np.int32, count = 1)[0]
            data = np.fromfile(f, np.float32, count = h*w*2)
            data.resize((h, w, 2))
            return data
        return None



class dataHandler:
    def __init__(self, main_path) -> None:
        self.main_path = main_path

        # Specify the path to the related data
        self.path_to_IMU = os.path.join(self.main_path, "imu.csv")
        self.path_to_GT = os.path.join(self.main_path, "gt.csv")  # Used for initialization and evaluation
        
        self.FlowFolder = os.path.join(self.main_path, "RaftOF")
        self.DepthFolder = os.path.join(self.main_path, "Depth")  # Used for Evaluation
        self.ImageFolder = os.path.join(self.main_path, "Images") # Used for Visualization Only
        
        # Load the gt and imu data
        self.imu = np.loadtxt(self.path_to_IMU,delimiter=",",comments="#").reshape(-1,7)
        self.gt = np.loadtxt(self.path_to_GT,delimiter=",",comments="#").reshape(-1,11)
        
        # Create a list of the flows, depth and image folder.
        self.FlowList = itemList(self.FlowFolder)
        self.DepthList = itemList(self.DepthFolder)
        self.ImageList = itemList(self.ImageFolder)

        self.imu_times = self.imu[:,0]
        self.gt_times = self.gt[:,0]
        
        self.LA = LieUtils()
    def select_imu_readings(self, start_timestamps, stop_timestamp):
        assert start_timestamps < stop_timestamp
        related_reading_indexes = np.where((self.imu_times >= (start_timestamps)) & (self.imu_times <= stop_timestamp))[0]
        
        # We assume that the fps of the imu is higher than camera for implementation
        assert len(related_reading_indexes)>0
        
        first_imu_idx  = related_reading_indexes[0]
        last_imu_index = related_reading_indexes[-1]
        
        # Create an array to hold related IMU readings
        imu_readings = self.imu[first_imu_idx:last_imu_index+2,:].reshape(-1,7)
        
        # We need to interpolate the last IMU measurement
        imu_timestamp_before_last = self.imu[last_imu_index,0]
        imu_timestamp_last = self.imu[last_imu_index+1,0]
        
        # Interpolation Regulator
        alpha = (stop_timestamp - imu_timestamp_before_last) / (imu_timestamp_last - imu_timestamp_before_last)

        # Interpolation
        imu_meas_before_last = self.imu[last_imu_index,:]
        imu_meas_last = self.imu[last_imu_index+1,:]
        imu_meas_interpolated = (1.0-alpha) * imu_meas_before_last + alpha * imu_meas_last
        imu_readings[-1,:] = imu_meas_interpolated
        
        # Make sure that the interpolation is done properly by checking the interpolated timestamp
        # Recall the fact that the unit of timestamp is nanosecond
        assert abs(imu_readings[-1,0] - stop_timestamp) < 1e4
        
        # Modify the Last IMU reading
        imu_readings[-1,:] = imu_meas_interpolated  
        # imu_readings[-1,0] = stop_timestamp     
        
        return imu_readings
    
    def select_imu_readings_start_synched(self, start_timestamps, stop_timestamp):
        assert start_timestamps < stop_timestamp
        related_reading_indexes = np.where((self.imu_times >= (start_timestamps)) & (self.imu_times <= stop_timestamp))[0]
        
        # We assume that the fps of the imu is higher than camera for implementation
        assert len(related_reading_indexes)>0
        
        first_imu_idx  = related_reading_indexes[0]
        last_imu_index = related_reading_indexes[-1]
        
        # Create an array to hold related IMU readings
        imu_readings = self.imu[first_imu_idx-1:last_imu_index+1,:].reshape(-1,7)
        
        # We need to interpolate the first IMU measurement
        imu_timestamp_before_start = self.imu[first_imu_idx-1,0]
        imu_timestamp_start = self.imu[first_imu_idx,0]
        
        # Interpolation Regulator
        alpha = (start_timestamps - imu_timestamp_before_start) / (imu_timestamp_start - imu_timestamp_before_start)

        # Interpolation
        imu_meas_before_start = self.imu[first_imu_idx-1,:]
        imu_meas_start = self.imu[first_imu_idx,:]
        imu_meas_interpolated = (1.0-alpha) * imu_meas_before_start + alpha * imu_meas_start
        imu_readings[0,:] = imu_meas_interpolated
        
        # Make sure that the interpolation is done properly by checking the interpolated timestamp
        # Recall the fact that the unit of timestamp is nanosecond
        assert abs(imu_readings[0,0] - start_timestamps) < 1e4
        
        return imu_readings       

              

    
    def get_imu_from_idx(self,idx):
        timestamp = self.imu[idx,0]
        gyro_meas = self.imu[idx,1:4]
        acce_meas = self.imu[idx,4:7]
        return timestamp, gyro_meas, acce_meas
    
    
    def get_gt(self,curr_t):
        abs_err = np.abs(self.gt_times - curr_t)
        min_index = np.argmin(abs_err)
        
        closest_t = self.gt_times[min_index]
        
        if closest_t < curr_t:
            second_closest_t_idx = min_index + 1
        else:
            second_closest_t_idx = min_index - 1
            
        
        second_closest_t = self.gt_times[second_closest_t_idx]
        
        total_dif = abs( float(second_closest_t-closest_t) )
        alpha = abs( float(closest_t-curr_t) ) / total_dif
        
        gt1 = self.gt[min_index,1:].reshape(10,1)
        gt2 = self.gt[second_closest_t_idx,1:].reshape(10,1)
        
        t1 = gt1[0:3,0]
        q1 = gt1[3:7,0]
        v1 = gt1[7:10,0]
        
        t2 = gt2[0:3,0]
        q2 = gt2[3:7,0]
        v2 = gt2[7:10,0]
        
        t_interp = (1-alpha) * t1 + alpha * t2
        v_interp = (1-alpha) * v1 + alpha * v2
        
        key_rots = R.from_quat([q1,q2])      
        key_times = [0,1]
        
        slerp = Slerp(key_times, key_rots)
        quat_interp = slerp(alpha).as_quat()

        return quat_interp, v_interp, t_interp
    
    def get_preintegrated_measurement(self, curr_timestamp, next_timestamp):
        imu_readings = self.select_imu_readings_start_synched(curr_timestamp, next_timestamp)

        Delta_R = np.eye(3)
        Delta_v = np.zeros((3,1))
        Delta_t = np.zeros((3,1))
        for imu_idx in range(imu_readings.shape[0]-1):
            imu_timestamp = imu_readings[imu_idx,0]
            next_imu_timestamp = imu_readings[imu_idx+1,0]
            dt = (next_imu_timestamp - imu_timestamp) * (1e-9)
            
            gyro_meas = imu_readings[imu_idx,1:4].reshape(3,1)
            acce_meas = imu_readings[imu_idx,4:7].reshape(3,1)
            
            Delta_R = Delta_R @ self.LA.exp_SO3(gyro_meas*dt)
            Delta_v = Delta_v.reshape(3,1) + (self.LA.integral_exp_SO3(gyro_meas, dt) @ acce_meas).reshape(3,1)
            Delta_t = Delta_t.reshape(3,1) + (self.LA.double_integral_exp_SO3(gyro_meas, dt) @ acce_meas).reshape(3,1)

        # Also use the last IMU measurement to propagate the state
        imu_timestamp = imu_readings[-1,0]
        next_imu_timestamp = next_timestamp
        dt = (next_imu_timestamp - imu_timestamp) * (1e-9)
        
        gyro_meas = imu_readings[-1,1:4].reshape(3,1)
        acce_meas = imu_readings[-1,4:7].reshape(3,1)
        
        Delta_R = Delta_R @ self.LA.exp_SO3(gyro_meas*dt)
        Delta_v = Delta_v.reshape(3,1) + (self.LA.integral_exp_SO3(gyro_meas, dt) @ acce_meas).reshape(3,1)
        Delta_t = Delta_t.reshape(3,1) + (self.LA.double_integral_exp_SO3(gyro_meas, dt) @ acce_meas).reshape(3,1)
        
        return Delta_R, Delta_v, Delta_t