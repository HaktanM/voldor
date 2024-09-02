import numpy as np
import sys
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

from my_utils import LieUtils
    

def get_curr_gt_idx(gt_times,curr_t):
    abs_err = np.abs(gt_times - curr_t)
    min_index = np.argmin(abs_err)
    if abs_err[min_index] > 0.5 * 1e9:
        print(f"We cannot find gt close enough to estimation in time domain")
        sys.exit()
        
    return min_index

def get_sync_gt(es_times, gt):
    gt = gt.reshape(-1,11)
    gt_sync = np.zeros((es_times.shape[0],11))
    
    gt_times = gt[:,0].reshape(-1)
    es_times = es_times.reshape(-1)
    
    for idx, curr_time in enumerate(es_times):
        gt_sync[idx,0] = curr_time
        closest_gt_idx = get_curr_gt_idx(gt_times, curr_time)
        
        # If the closest time index is on the edge of the data, we cannot interpolate
        if closest_gt_idx == 0 or closest_gt_idx == gt.shape[0] -1:
            gt_sync[idx,1:] = gt[closest_gt_idx,1:]
        else: # Interpolate the GT data for improved accuracy
            closest_time = gt_times[closest_gt_idx]
            if closest_time < curr_time:
                second_closest_gt_idx = closest_gt_idx + 1
            else:
                second_closest_gt_idx = closest_gt_idx - 1
            second_closest_time = gt_times[second_closest_gt_idx]
            
            total_dif = abs( float(second_closest_time-closest_time) )
            alpha = abs( float(closest_time-curr_time) ) / total_dif
            
            gt1 = gt[closest_gt_idx,:].reshape(11,1)
            gt2 = gt[second_closest_gt_idx,:].reshape(11,1)

            q1 = gt1[4:8,0]
            v1 = gt1[8:11,0]
            t1 = gt1[1:4,0]
            
            q2 = gt2[4:8,0]
            v2 = gt2[8:11,0]
            t2 = gt2[1:4,0]
            
            key_rots = R.from_quat([q1,q2])      
            key_times = [0,1]

            slerp = Slerp(key_times, key_rots)
            quat_interp = slerp(alpha).as_quat()
            
            v_interp = (1-alpha) * v1 + alpha * v2
            t_interp = (1-alpha) * t1 + alpha * t2
            
            gt_sync[idx,1:5] = quat_interp
            gt_sync[idx,5:8] = v_interp
            gt_sync[idx,8:11] = t_interp
    return gt_sync

class State:
    def __init__(self, timestamp, R_b_g, v_gb_g, t_gb_g) -> None:
        R_b_g = R_b_g.reshape(3,3)
        v_gb_g = v_gb_g.reshape(3)
        t_gb_g = t_gb_g.reshape(3)
        
        self.T = np.eye(5)
        self.T[:3, :3] = R_b_g
        self.T[:3,3] = v_gb_g
        self.T[:3,4] = t_gb_g
        
        self.timestamp = timestamp
        
        self.LA = LieUtils() # Linear Algebra (LA) Utils
        self.gravity = np.array([0.0, 0.0, 9.80665016174316]) # 9.80665016174316

    def propagate(self, timestamp, gyr_data, acc_data):
        dt = (timestamp - self.timestamp) * (1e-9) # Unit of dt should be sec
        
        # dt should be non-negative
        assert dt >= 0.0
        
        # We expect the the frequency of IMU is higher than 50Hz
        # Lower frequencies are still ok, but the performance degrades
        assert dt <  0.02 
        
        gyr_data = gyr_data.reshape(3,1)
        acc_data = acc_data.reshape(3,1)

        Gamma = self.get_Gamma(dt).reshape(5,5)
        Phi = self.get_Phi(dt).reshape(5,5)
        Upsilon = self.get_Upsilon(gyr_data, acc_data, dt).reshape(5,5)
        
        self.T = ((Gamma @ Phi) @ Upsilon).reshape(5,5)
        self.timestamp = timestamp
        
    def propagate_using_preintegtrated_imu(self, new_timestamp, Delta_R, Delta_v, Delta_t):
        dt = (new_timestamp - self.timestamp) * (1e-9) # Unit of dt should be sec
        
        # dt should be non-negative
        assert dt >= 0.0

        Gamma = self.get_Gamma(dt).reshape(5,5)
        Phi = self.get_Phi(dt).reshape(5,5)
        
        Upsilon = np.eye(5)
        Upsilon[:3,:3] = Delta_R
        Upsilon[:3,3] = Delta_v.reshape(3)
        Upsilon[:3,4] = Delta_t.reshape(3)
        
        self.T = ((Gamma @ Phi) @ Upsilon).reshape(5,5)
        self.timestamp = new_timestamp
            
 
    def get_Gamma(self, dt):        
        Gamma_v = self.gravity * dt
        Gamma_t = 0.5 * self.gravity * (dt**2)

        Gamma = np.eye(5)
        Gamma[:3,3] = Gamma_v.reshape(3)
        Gamma[:3,4] = Gamma_t.reshape(3)
        return Gamma


    def get_Phi(self, dt):
        Phi = self.T.copy()
        Phi[:3,4] = Phi[:3,4] + Phi[:3,3]*dt       
        return Phi


    def get_Upsilon(self,gyr_data,acc_data,dt):
        Delta_R = self.LA.exp_SO3(gyr_data * dt)
        Delta_v = self.LA.integral_exp_SO3(gyr_data, dt) @ acc_data
        Delta_t = self.LA.double_integral_exp_SO3(gyr_data,dt) @ acc_data

        Upsilon = np.eye(5)
        Upsilon[:3,:3] = Delta_R
        Upsilon[:3,3] = Delta_v.reshape(3)
        Upsilon[:3,4] = Delta_t.reshape(3)
        return Upsilon
    
    

def compute_err(gt, es):
    gt = gt.reshape(-1,11)
    es = es.reshape(-1,11)
    
    data_length = es.shape[0]
    errors = np.zeros((data_length,10))
    
    for idx in range(data_length):
        curr_time = es[idx, 0]
        
        q_b_g_es = es[idx, 1:5].reshape(4)
        q_b_g_es = R.from_quat(q_b_g_es)
        v_gb_g_es = es[idx, 5:8]
        t_gb_g_es = es[idx, 8:11]
        
        q_b_g_gt = gt[idx, 1:5].reshape(4)
        q_b_g_gt = R.from_quat(q_b_g_gt)
        v_gb_g_gt = gt[idx, 5:8]
        t_gb_g_gt = gt[idx, 8:11]
        
        t_gb_g_err = t_gb_g_gt - t_gb_g_es       
        v_gb_g_err = v_gb_g_gt - v_gb_g_es
        
        R_b_g_es = q_b_g_es.as_matrix()
        R_b_g_gt = q_b_g_gt.as_matrix()
        
        R_err = R_b_g_es.transpose() @ R_b_g_gt
        psi = R.from_matrix(R_err).as_euler("xyz", degrees=True)
        
        errors[idx,0] = curr_time
        errors[idx,1:4]  = psi
        errors[idx,4:7]  = v_gb_g_err
        errors[idx,7:10] = t_gb_g_err
        
    return errors