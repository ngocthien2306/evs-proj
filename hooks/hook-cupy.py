import cupy
import sys
import os
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

# Auto collect all cupy related modules
hiddenimports = [x for x in sys.modules.keys() 
                 if x.startswith('cupy.') or 
                    x.startswith('cupyx.') or 
                    x.startswith('cupy_backends.')] + [
                    'fastrlock',
                    'fastrlock.rlock'
                ]
# Collect binaries and data files
binaries = collect_dynamic_libs('cupy')
datas = collect_data_files('cupy')

# Add the specific CUDA DLLs that are needed
cuda_path = os.environ.get('CUDA_PATH', 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1')
cuda_bin = os.path.join(cuda_path, 'bin')

additional_dlls = [
    'nvrtc64_121_0.dll',  # Also commonly needed
    'cublas64_12.dll',    # May be needed for CuPy operations
    'cufft64_11.dll',     # May be needed for CuPy operations
]

for dll in additional_dlls:
    dll_path = os.path.join(cuda_bin, dll)
    if os.path.exists(dll_path):
        binaries.append((dll_path, '.'))