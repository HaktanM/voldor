from scipy.spatial.transform import Rotation as R
R_gim_body = R.from_euler('y', -90.0, degrees=True).as_matrix()

print(R_gim_body)