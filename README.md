# Event-based Visual System (EVS)

A high-performance event-based visual processing system that combines event camera data with radar-like visualization for object tracking and monitoring.

## Features

- Real-time event processing using GPU acceleration (CUDA via CuPy)
- Radar-style visualization of detected objects
- Multiple object tracking with smooth trajectory prediction
- Configurable filtering and noise reduction
- Support for both filtered and unfiltered event processing modes
- FPS limiting for resource optimization
- Comprehensive visualization with angle and distance measurements
- Advanced peak detection and tracking algorithms
- Hungarian algorithm-based object matching
- Spring-mass-damper motion model for smooth trajectory prediction
- Adaptive noise filtering with spatial and temporal constraints
- Live visualization with configurable display options

## Requirements

- Python 3.7+ (Python 3.9 recommended)
- CUDA-compatible GPU
- Dependencies:
  - cupy-cuda12x==13.3.0
  - numpy==1.23.5
  - opencv-python==4.10.0.84
  - opencv-contrib-python==4.10.0.84
  - scipy==1.15.0
  - metavision_core

## Installation

### Step-by-Step Setup for Python 3.9 and CUDA 12.7

1. **Install Python 3.9**:
   - Download and install Python 3.9 from the [official Python website](https://www.python.org/downloads/).
   - Ensure that Python 3.9 is added to your system's PATH.

2. **Install CUDA Toolkit 12.7**:
   - Download and install the CUDA Toolkit 12.7 from the [NVIDIA website](https://developer.nvidia.com/cuda-toolkit).
   - Follow the installation instructions specific to your operating system.

3. **Set up a Virtual Environment**:
   - Open a terminal or command prompt.
   - Navigate to your project directory.
   - Create a virtual environment:
     ```bash
     python3.9 -m venv evs_env
     ```
   - Activate the virtual environment:
     - On Windows:
       ```bash
       evs_env\Scripts\activate
       ```
     - On macOS/Linux:
       ```bash
       source evs_env/bin/activate
       ```

4. **Install Python Dependencies**:
   - Ensure your virtual environment is activated.
   - Install the required packages using the `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
     ```

5. **Verify Installation**:
   - Check that all dependencies are installed correctly:
     ```bash
     python -m pip list
     ```
   - Ensure that `cupy-cuda12x`, `numpy`, `opencv-python`, `opencv-contrib-python`, and `scipy` are listed.

## Usage

Run the system with either a camera or event data file:

```bash
# Using event data file
python src/main.py --file_path path/to/events.dat

# Using camera input
python src/main.py --camera
```

### Command Line Arguments

- `--file_path`: Path to the event data file (required if not using camera)
- `--camera`: Use camera input instead of file
- `--label_dir`: Directory containing label files (optional)
- `--trim_angle`: Edge trimming angle in degrees (default: 1.0)
- `--max_objects`: Maximum number of objects to track (default: 3)
- `--use_filter`: Enable activity noise filter (default: False)
- `--neighborhood_size`: Filter neighborhood size (default: 3)
- `--time_tolerance`: Time tolerance in microseconds (default: 1000)
- `--buffer_size`: Size of processing buffer (default: 256)

## Performance Optimization

The system is optimized for performance through:
1. GPU acceleration for event processing
2. Multi-threaded operations for parallel processing
3. Efficient memory management with CuPy
4. Optimized visualization routines
5. Configurable FPS limiting
6. Batch processing of events
7. Dynamic buffer management
8. Adaptive filtering thresholds

## Architecture

The system consists of several key components:

- `main.py`: Core execution and coordination
- `config.py`: System configuration and parameters
- `radar.py`: Radar visualization and object tracking
- `filters.py`: Event filtering and noise reduction algorithms
- `utils.py`: Utility functions for visualization and data processing
- `logging_config.py`: Logging configuration and management
- Additional modules for event filtering and activity monitoring

### Key Features Implementation

- **Event Processing**: GPU-accelerated event processing with CuPy
- **Object Tracking**: Hungarian algorithm with dynamic gating thresholds
- **Motion Model**: Spring-mass-damper system for trajectory prediction
- **Visualization**: Combined event and radar display with real-time metrics
- **Performance**: Optimized memory management and parallel processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details