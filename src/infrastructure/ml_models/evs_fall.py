import time
import numpy as np
import onnxruntime as ort

class evs_fall_exist:
    def __init__(self, path, model_type='tflite', num_threads=1):
        """Initialize the fall detection model.
        
        Args:
            path (str): Path to the model file
            model_type (str): Type of model ('tflite' or 'onnx')
            num_threads (int): Number of threads for TFLite inference
        """
        print(f"Initializing model with type: {model_type}")
        # if model_type == 'onnx':
            # print(f"ONNX Runtime version: {ort.__version__}")
            # print(f"Available providers: {ort.get_available_providers()}")
        
        self.model_type = model_type
        self.model_path = path
        
        if model_type == 'onnx':
            self.initialize_onnx_model(path)
        else:
            raise ValueError("Unsupported model type. Use 'tflite' or 'onnx'.")
        
        # Perform warm-up during initialization
        self.is_warmed_up = False
        self.warm_up()
        
    def warm_up(self, num_iterations=3):
        """Warm up the model by running multiple inferences with dummy data.
        
        Args:
            num_iterations (int): Number of warm-up iterations
            
        Returns:
            bool: True if warm-up was successful, False otherwise
        """
        try:
            print(f"Warming up {self.model_type} model...")
            start_time = time.time()
            
            # Create dummy input based on expected input shape
            if self.model_type == 'tflite':
                input_shape = tuple(self.input_shape[1:])  # Remove batch dimension
                dummy_frame = np.random.random(input_shape).astype(np.float32)
                dummy_frames = [dummy_frame for _ in range(max(10, input_shape[0]))]
            else:  # onnx
                # For ONNX, create dummy data matching expected dimensions (320x30)
                batch, channels, width, height = self.input_shape  # Note: width is 320, height is 30
                dummy_frame = np.random.random((height, width, channels)).astype(np.float32)
                dummy_frames = [dummy_frame for _ in range(10)]  # Fixed number of frames for warm-up
            dummy_idx = 0
            
            # Run warm-up iterations
            for i in range(num_iterations):
                _ = self.detect_fall_exist(dummy_frames, dummy_idx)
            
            elapsed_time = time.time() - start_time
            print(f"Model warm-up completed: {num_iterations} iterations in {elapsed_time:.2f} seconds")
            
            self.is_warmed_up = True
            return True
            
        except Exception as e:
            print(f"Error during model warm-up: {str(e)}")
            return False
    
    def __call__(self, frame_array, idx_a):
        """Detect falls in the given frame array.
        
        Args:
            frame_array: Array of frames
            idx_a: Index for frame selection
            
        Returns:
            tuple: (fall_detection, existence_detection)
        """
        return self.detect_fall_exist(frame_array, idx_a)
    

    def initialize_onnx_model(self, path):
        """Initialize ONNX model with CUDA support and CPU fallback.
        
        Args:
            path (str): Path to ONNX model
        """
        try:
            # First try with CUDA
            providers = ['CUDAExecutionProvider', "CPUExecutionProvider"]
            provider_options = [
                {
                    'device_id': 0
                },
                {}
            ]
            
            self.session = ort.InferenceSession(
                path,
                providers=providers,
                provider_options=provider_options
            )
        
            print("ONNX model initialized with CUDA support")
            
        except Exception as e:
            print(f"Failed to initialize with CUDA, falling back to CPU. Error: {e}")
            try:
                self.session = ort.InferenceSession(
                    path,
                    providers=['CPUExecutionProvider']
                )
                print("ONNX model initialized with CPU support")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize ONNX model: {str(e)}")
        
        # Get model information
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"Input Name: {self.input_name}")
        print(f"Input Shape: {self.input_shape}")
        print(f"Output Names: {self.output_names}")
    
    def detect_fall_exist(self, frame_array, idx_a):
        """Detect falls and existence in frames.
        
        Args:
            frame_array: Array of frames
            idx_a: Index for frame selection
            
        Returns:
            tuple: (fall_detection, existence_detection)
        """
        try:
            x = self.prepare_input(frame_array, idx_a)
            
            outputs = self.session.run(self.output_names, {self.input_name: x})
            output_001 = outputs[0]
            output_002 = outputs[1] if len(outputs) > 1 else outputs[0]
        
            output_fall, output_exists = self.process_outputs(output_001, output_002)
            return output_fall, output_exists
            
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            return None, None
    
    def process_outputs(self, output_001, output_002):
        """Process model outputs to determine fall and existence detection.
        
        Args:
            output_001: First model output
            output_002: Second model output
            
        Returns:
            tuple: (fall_detection, existence_detection)
        """
        # Extract single values from outputs if they're arrays
        if isinstance(output_001, np.ndarray):
            output_001 = output_001.item()
        if isinstance(output_002, np.ndarray):
            output_002 = output_002.item()
        
        if output_001 < 0:
            output_fall = output_002
            output_exists = -output_001
        elif output_002 < 0:
            output_fall = output_001
            output_exists = -output_002
        elif output_001 > 0:
            output_fall = output_001
            output_exists = output_002
        else:
            output_fall = output_002
            output_exists = output_001
            
        return float(output_fall), float(output_exists)
    
    def prepare_input(self, frame_array, idx_a):
        """Prepare input frames for model inference.
        
        Args:
            frame_array: Array of frames
            idx_a: Index for frame selection
            
        Returns:
            numpy.ndarray: Processed input array
        """
        try:
            Frame_Array = np.array(frame_array)
            x = Frame_Array[idx_a]
            x = np.expand_dims(x, axis=0).astype('float32')
            
            if self.model_type == 'onnx':
                # Check and fix input shape if needed
                expected_shape = tuple(self.input_shape)
                current_shape = x.shape
                
                if current_shape != expected_shape:
                    # print(f"Input shape mismatch. Expected: {expected_shape}, Got: {current_shape}")
                    if len(expected_shape) == 4:
                        x = np.transpose(x, (0, 2, 3, 1))
                        # print(f"Reshaped input to: {x.shape}")
            
            return x
            
        except Exception as e:
            raise RuntimeError(f"Error preparing input: {str(e)}")

