from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from glob import glob
import os
import numpy
import subprocess
import shutil

# Change this if you target different host/device
# nvcc_machine_code = '' #'-m64 -arch=compute_61 -code=sm_61'

# gpu_sources_cpp = ' '.join(glob('../../gpu-kernels/*.cpp'))
# gpu_sources_cu = ' '.join(glob('../../gpu-kernels/*.cu'))

# gpu_kernel_build_cmd = f'/usr/local/cuda/bin/nvcc --compiler-options "-shared -fPIC" {gpu_sources_cpp} {gpu_sources_cu} -I/usr/include/opencv4 -lib -o libgpu-kernels.so -O3 {nvcc_machine_code}'
# os.system(gpu_kernel_build_cmd)

opencv_libs = subprocess.check_output('pkg-config --libs opencv4'.split())
opencv_libs = [name[2:] for name in str(opencv_libs, 'utf-8').split()][1:]


# ext = Extension('pyvoldor_vo',
#     sources = ['pyvoldor_vo.pyx'] + \
#             [x for x in glob('../../voldor/*.cpp') if 'main.cpp' not in x],
#     language = 'c++',
#     library_dirs = ['.', '/usr/local/lib', '/usr/local/cuda/lib64'],
#     libraries = ['gpu-kernels', 'cudart'] + opencv_libs,
#     include_dirs = [numpy.get_include(),'/usr/include/opencv4']
# )

# Define the extension
ext = Extension(
    'pyvoldor_vo',
    sources=['pyvoldor_vo.pyx'] + \
            [x for x in glob('../../voldor/*.cpp') if 'main.cpp' not in x],
    language='c++',
    library_dirs=['.', '/usr/local/lib', '/usr/local/cuda/lib64'],
    libraries=['gpu-kernels', 'cudart'] + opencv_libs + ['ceres'],
    include_dirs=[
        numpy.get_include(),
        '/usr/include/opencv4',
        '/usr/include/eigen3',  # Eigen include directory
        '/usr/local/include',  # General include directory for Ceres
    ],
    extra_compile_args=['-std=c++14'],  # Make sure your code is compatible with C++14 or later
)

setup(
    name='pyvoldor_vo',
    description='voldor visual odometry',
    author='Zhixiang Min',
    ext_modules=cythonize([ext])
)