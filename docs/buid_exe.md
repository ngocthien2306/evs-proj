# EVS Build Guide: Creating Executable with PyInstaller

This guide explains how to build a standalone executable for the Event-based Visual System (EVS) using PyInstaller.

## Prerequisites

- Python 3.9 environment with all dependencies installed
- CUDA Toolkit 12.1
- cuDNN 8.9.2+
- Metavision 4.6+ libraries
- All Python dependencies from requirements.txt

## Build Command

Use the following PyInstaller command to build the executable:

```batch
pyinstaller --noconfirm --onefile --clean ^
--name "EVS Combine" ^
--icon=public/assets/icon.ico ^
--add-data "public;public" ^
--additional-hooks-dir="hooks" ^
--paths "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin" ^
--paths "C:\Users\pclab321\miniconda3\envs\evs_build\lib\site-packages" ^
--paths "C:\Users\pclab321\miniconda3\envs\evs_build\lib\site-packages\numpy\core" ^
--add-binary "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\*.dll;." ^
--hidden-import multiprocessing ^
--hidden-import multiprocessing.pool ^
--hidden-import multiprocessing.managers ^
--hidden-import multiprocessing.shared_memory ^
--hidden-import multiprocessing.heap ^
--hidden-import multiprocessing.synchronize ^
--hidden-import cv2 ^
--hidden-import win32com.shell.shell ^
--hidden-import win32com ^
--hidden-import win32api ^
--hidden-import win32con ^
--hidden-import subprocess ^
--hidden-import cupy ^
--hidden-import onnxruntime ^
--hidden-import numpy.core._multiarray_umath ^
--hidden-import numpy.core.multiarray ^
--hidden-import numpy.core._dtype_ctypes ^
--hidden-import psutil ^
--hidden-import metavision_hal ^
--hidden-import metavision_core ^
--hidden-import metavision_core.event_io ^
--hidden-import metavision_core.event_io.raw_reader ^
--collect-all metavision_hal ^
--collect-all metavision_core ^
--exclude-module cutensor ^
--exclude-module torch ^
--exclude-module tensorflow ^
--exclude-module tensorflow-gpu ^
--exclude-module tensorboard ^
--exclude-module keras ^
--exclude-module cudnn ^
--exclude-module notebook ^
--exclude-module ipykernel ^
--exclude-module jedi ^
main.py
```

## Understanding the Build Configuration

### Core Options

- `--noconfirm`: Skip confirmation prompts
- `--onefile`: Create a single executable file
- `--clean`: Clean build files before each build
- `--name "EVS Combine"`: Set the output executable name
- `--icon=public/assets/icon.ico`: Set the application icon

### Data and File Inclusion

- `--add-data "public;public"`: Include the public directory in the package
- `--add-binary "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\*.dll;."`: Include CUDA DLLs

### Path Configuration

- `--paths`: Add additional paths for PyInstaller to search for modules
  - CUDA binary path
  - Python site-packages
  - NumPy core directory

### Custom Hooks and Hidden Imports

#### Why Use Custom Hooks?

PyInstaller uses hooks to correctly bundle complex packages. Custom hooks are needed when:

1. **Dynamic Imports**: Packages like CuPy use dynamic imports that PyInstaller can't detect statically
2. **C Extensions**: Libraries with C extensions require special handling
3. **Binary Dependencies**: GPU libraries (CUDA, cuDNN) have complex binary dependencies

Our custom hooks are in the `hooks` directory:
- `hook-cupy.py`: Handles CuPy's CUDA dependencies
- `hook-cv.py`: Configures OpenCV dependencies
- `hook-numpy.py`: Ensures NumPy's C extensions are included

#### Why Use Hidden Imports?

Hidden imports are modules that:
1. Are imported dynamically at runtime
2. Are imported using methods PyInstaller can't detect
3. Are part of complex packages with nested imports

Without specifying hidden imports, the executable would fail at runtime with "ModuleNotFoundError" errors when it tries to use these dynamically loaded modules.

Key hidden import categories:
- **Multiprocessing**: Required for parallel processing
- **CUDA/GPU Libraries**: CuPy, ONNX Runtime
- **Low-level interfaces**: Core NumPy components, Windows API functions
- **Event camera libraries**: Metavision modules

### Module Collection and Exclusion

- `--collect-all metavision_hal`: Collect all submodules from Metavision HAL
- `--collect-all metavision_core`: Collect all submodules from Metavision Core
- `--exclude-module`: Exclude unused large packages to reduce executable size

## Required Hook Files

Place these hook files in the `hooks` directory:

### hook-cupy.py
```python
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
```

### hook-cv.py
```python
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

hiddenimports = ['numpy']
binaries = collect_dynamic_libs('cv2')
datas = collect_data_files('cv2')
```

### hook-numpy.py
```python
import os
import numpy
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

numpy_dir = r'C:\Users\pclab321\miniconda3\envs\evs_build\lib\site-packages\numpy'
numpy_core_dir = os.path.join(numpy_dir, 'core')

hiddenimports = [
    'numpy',
    'numpy.core._multiarray_umath',
    'numpy.core.multiarray',
    'numpy.core.numeric',
    'numpy.core._dtype_ctypes',
    'numpy.lib.format',
    'numpy.core._multiarray_tests',
    'numpy.core.numerictypes',
    'numpy.core.fromnumeric',
    'numpy.core.arrayprint'
]

binaries = []
core_libs = []

# Collect all PYD files from numpy.core
for f in os.listdir(numpy_core_dir):
    if f.endswith('.pyd'):
        full_path = os.path.join(numpy_core_dir, f)
        rel_path = os.path.join('numpy', 'core')
        core_libs.append((full_path, rel_path))

binaries.extend(core_libs)
binaries.extend(collect_dynamic_libs('numpy'))
datas = collect_data_files('numpy')
```

## Building Process

1. Ensure all dependencies are installed in your environment
2. Create the `hooks` directory and add the hook files
3. Run the PyInstaller command above
4. The executable will be created in the `dist` directory

## Testing the Executable

After building, test the executable in these scenarios:
1. On a machine with CUDA drivers installed
2. With various input sources (camera, file)
3. Test all major features (detection, tracking, fall detection)

## Common Issues and Solutions

1. **Missing DLL errors**: Add the missing DLL to the `--add-binary` command
2. **ModuleNotFoundError**: Add the module to `--hidden-import`
3. **CUDA initialization errors**: Ensure CUDA Toolkit version matches the one used during development
4. **Metavision errors**: Make sure Metavision libraries are correctly installed and collected

## Optimizing Build Size

The current build excludes several large libraries to reduce size. If you need to further optimize:

1. Use UPX compression: Add `--upx-dir=path\to\upx` to compress the executable
2. Exclude more unused modules with `--exclude-module`
3. Use `--strip` to remove debugging symbols

## Distributing the Application

When distributing the application:

1. Ensure the target system has CUDA drivers installed (minimum version required: 12.0)
2. Include any additional DLLs that might be required
3. Provide clear installation instructions for dependencies that can't be bundled