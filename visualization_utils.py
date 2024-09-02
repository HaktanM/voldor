import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class traj_visualizer:
    def __init__(self) -> None:       
        # Create a figure
        self.fig = plt.figure(figsize=(12, 6))

        # Create a GridSpec with 2 rows and 2 columns
        self.gs = GridSpec(3, 2, figure=self.fig)
        
        self.ax1 = self.fig.add_subplot(self.gs[:3, 0])
        self.ax1.set_xlabel("North  (Meters)")
        self.ax1.set_ylabel("East (Meters)")
        
        self.ax2 = self.fig.add_subplot(self.gs[0, 1])
        self.ax2.set_xlabel("Time  (Sec)")
        self.ax2.set_ylabel("North (Meters)")
        
        self.ax3 = self.fig.add_subplot(self.gs[1, 1])
        self.ax3.set_xlabel("Time  (Sec)")
        self.ax3.set_ylabel("East (Meters)")
        
        self.ax4 = self.fig.add_subplot(self.gs[2, 1])
        self.ax4.set_xlabel("Time  (Sec)")
        self.ax4.set_ylabel("Down (Meters)")
        
        self.gt_color = "green"
        self.es_color = "red"
    
    def plot_traf(self, traj, color):
        traj = traj.reshape(-1,4)
        t = (traj[:,0] - traj[0,0]) * 1e-9
        x = traj[:,1]
        y = traj[:,2]
        z = traj[:,3]
        
        self.ax1.plot(x, y, color=color)
        self.ax2.plot(t, x, color=color)
        self.ax3.plot(t, y, color=color)
        self.ax4.plot(t, z, color=color)
        
    def add_gt(self, traj):
        self.plot_traf(traj, self.gt_color)
    
    def add_es(self, traj):
        self.plot_traf(traj, self.es_color)
        
    def show(self):
        plt.tight_layout()
        plt.show()
        
    def clear(self):
        plt.cla()
        plt.clf()
        plt.close()
        
    def save(self, path_to_save):
        # Save the figure
        plt.savefig(path_to_save)
        self.clear()
        
def get_error_norms(errors):
    pos = errors[:,1:4]
    ori = errors[:,4:7]
    vel = errors[:,7:10]
    
    data_length = errors.shape[0]
    error_norms = np.zeros((data_length,3))
    
    for idx in range(data_length):
        error_norms[idx,0] = np.linalg.norm(pos[idx,:])
        error_norms[idx,1] = np.linalg.norm(ori[idx,:])
        error_norms[idx,2] = np.linalg.norm(vel[idx,:])
        
    return error_norms

class err_vis():
    def __init__(self) -> None:
        self.fig, self.ax = plt.subplots(4,3,figsize=(18,8))

    def add_err_plt(self, errors):
        error_norms = get_error_norms(errors)
        time_arr = (errors[:,0] - errors[0,0]) * (1e-9)
        self.ax[0,0].plot(time_arr, errors[:,1],linewidth=1.5)
        self.ax[1,0].plot(time_arr, errors[:,2],linewidth=1.5)
        self.ax[2,0].plot(time_arr, errors[:,3],linewidth=1.5)
        self.ax[3,0].plot(time_arr, error_norms[:,0],linewidth=1.5)

        self.ax[0,1].plot(time_arr, errors[:,4],linewidth=1.5)
        self.ax[1,1].plot(time_arr, errors[:,5],linewidth=1.5)
        self.ax[2,1].plot(time_arr, errors[:,6],linewidth=1.5)
        self.ax[3,1].plot(time_arr, error_norms[:,1],linewidth=1.5)

        self.ax[0,2].plot(time_arr, errors[:,7],linewidth=1.5)
        self.ax[1,2].plot(time_arr, errors[:,8],linewidth=1.5)
        self.ax[2,2].plot(time_arr, errors[:,9],linewidth=1.5)
        self.ax[3,2].plot(time_arr, error_norms[:,2],linewidth=1.5)
        
        self.ax[0,0].set_title("Orientation Error (degree)")
        self.ax[0,1].set_title("Velocity Error (meter/sec)")
        self.ax[0,2].set_title("Position Error (meter)")
        
    def show(self):
        plt.tight_layout()
        plt.show()
        
    def clear(self):
        plt.cla()
        plt.clf()
        plt.close()
        
    def save(self, path_to_save):
        # Save the figure
        plt.savefig(path_to_save)
        self.clear()