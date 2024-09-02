import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import re

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
        data = np.flipud(data)
        
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