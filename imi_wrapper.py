import cv2
import numpy as np
import os
from ctypes import *
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Union
from enum import Enum
import time

# SDK Structure definitions remain the same but removing UVC-specific structures
class ImiFrameMode(Structure):
    _fields_ = [
        ("pixelFormat", c_uint32),
        ("resolutionX", c_int16),
        ("resolutionY", c_int16),
        ("bitsPerPixel", c_int8),
        ("framerate", c_int8)
    ]

class ImiImageFrame(Structure):
    _fields_ = [
        ("pixelFormat", c_uint32),
        ("type", c_uint32),
        ("frameNum", c_uint32),
        ("timeStamp", c_uint64),
        ("fps", c_uint32),
        ("width", c_int32),
        ("height", c_int32),
        ("pData", c_void_p),
        ("pSkeletonData", c_void_p),
        ("size", c_uint32)
    ]

@dataclass
class FrameMode:
    """Camera frame mode settings"""
    width: int
    height: int
    fps: int
    pixel_format: 'PixelFormat'

class StreamType(Enum):
    """Available stream types"""
    DEPTH = 0x00
    COLOR = 0x01
    IR = 0x02

class PixelFormat(Enum):
    """Available pixel formats"""
    DEPTH_16BIT = 0x00000000
    RGB_888 = 0x00000001
    YUV420SP = 0x00000002
    YUV422 = 0x00000003
    IR_16BIT = 0x00000004

class ImiFrame:
    """Base class for camera frames"""
    def __init__(self, data: np.ndarray, timestamp: int, frame_number: int):
        self.data = data
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.height, self.width = data.shape[:2]

class DepthFrame(ImiFrame):
    """Depth frame data"""
    pass

class ColorFrame(ImiFrame):
    """Color frame data"""
    pass

class ImiCamera:
    """Enhanced interface for IMI depth camera with OpenCV color support"""
    
    # SDK Constants
    IMI_SUCCESS = 0
    IMI_PROPERTY_DEPTH_INTRINSICS = 0x36
        
    @staticmethod
    def list_available_cameras():
        """List all available OpenCV cameras on the system"""
        available_cameras = []
        for i in range(10):  # Check first 10 indices
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        # Get some basic properties
                        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        available_cameras.append({
                            'index': i,
                            'working': True,
                            'resolution': f"{int(width)}x{int(height)}",
                            'fps': fps
                        })
                    else:
                        available_cameras.append({
                            'index': i,
                            'working': False
                        })
                cap.release()
            except:
                continue
        return available_cameras

    def __init__(self, lib_path: Optional[str] = None, color_index: Optional[int] = None):
        """Initialize camera interface
        
        Args:
            lib_path: Optional path to IMI SDK library
            color_index: OpenCV camera index for color stream. If None, will auto-detect.
        """
        
        if lib_path is None:
            sdk_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.imi_lib_path = os.path.join(sdk_dir, 'libs', 'libiminect.so')
        
        if not os.path.exists(self.imi_lib_path):
            raise RuntimeError(f"IMI SDK library not found at {self.imi_lib_path}")
            
        self.lib = CDLL(self.imi_lib_path)
        self._setup_functions()
        
        self.device = None
        self.streams = {}
        self.intrinsics = {}
        
        # Auto-detect color camera if index not specified
        if color_index is None:
            cameras = self.list_available_cameras()
            working_cameras = [cam for cam in cameras if cam['working']]
            if working_cameras:
                self.color_index = working_cameras[0]['index']
                print(f"Auto-detected color camera at index {self.color_index}")
            else:
                self.color_index = 2  # Default fallback
                print("No working cameras detected, defaulting to index 2")
        else:
            self.color_index = color_index
        
        self.color_cap = None
        self.frame_count = 0

    def _setup_functions(self):
        """Setup C function interfaces from SDK"""
        # Basic IMI functions
        self.lib.imiInitialize.argtypes = []
        self.lib.imiInitialize.restype = c_int32
        
        # Device functions
        self.lib.imiOpenDevice.argtypes = [c_char_p, POINTER(c_void_p), c_int32]
        self.lib.imiOpenDevice.restype = c_int32
        
        self.lib.imiCloseDevice.argtypes = [c_void_p]
        self.lib.imiCloseDevice.restype = c_int32
        
        # Stream functions
        self.lib.imiOpenStream.argtypes = [c_void_p, c_uint32, c_void_p, c_void_p, POINTER(c_void_p)]
        self.lib.imiOpenStream.restype = c_int32
        
        self.lib.imiCloseStream.argtypes = [c_void_p]
        self.lib.imiCloseStream.restype = c_int32
        
        # Frame functions
        self.lib.imiReadNextFrame.argtypes = [c_void_p, POINTER(POINTER(ImiImageFrame)), c_int32]
        self.lib.imiReadNextFrame.restype = c_int32
        
        self.lib.imiReleaseFrame.argtypes = [POINTER(POINTER(ImiImageFrame))]
        self.lib.imiReleaseFrame.restype = c_int32

    def initialize(self) -> None:
        """Initialize the IMI SDK and open devices"""
        # Initialize IMI SDK
        ret = self.lib.imiInitialize()
        if ret != self.IMI_SUCCESS:
            raise RuntimeError(f"Failed to initialize IMI SDK: {ret}")
        
        # Open main IMI device
        device_ptr = c_void_p()
        ret = self.lib.imiOpenDevice(None, byref(device_ptr), 0)
        if ret != self.IMI_SUCCESS:
            raise RuntimeError(f"Failed to open IMI device: {ret}")
        self.device = device_ptr

    def open_stream(self, stream_type: StreamType) -> None:
        """Open stream of specified type"""
        if stream_type in self.streams:
            return  # Stream already open
            
        if stream_type == StreamType.COLOR:
            self._open_color_stream()
        else:
            self._open_regular_stream(stream_type)
            
    def _open_color_stream(self):
        """Open color stream using OpenCV with robust initialization"""
        try:
            print(f"Opening OpenCV color stream with index {self.color_index}...")
            
            # First try to open with default backend
            self.color_cap = cv2.VideoCapture(self.color_index)
            if not self.color_cap.isOpened():
                # Try with specific backend
                self.color_cap = cv2.VideoCapture(self.color_index, cv2.CAP_V4L2)
                
            if not self.color_cap.isOpened():
                raise RuntimeError(f"Failed to open OpenCV camera {self.color_index}")
            
            # Set camera properties
            self.color_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.color_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.color_cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Give the camera time to initialize
            time.sleep(1.0)
            
            # Warm up the camera by reading a few frames
            for _ in range(5):
                self.color_cap.read()
                time.sleep(0.1)
                
            # Store the capture object as the stream
            self.streams[StreamType.COLOR] = self.color_cap
            
            # Get and print camera properties
            width = self.color_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.color_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.color_cap.get(cv2.CAP_PROP_FPS)
            print(f"Color camera initialized: {width}x{height} @ {fps}fps")
            
        except Exception as e:
            if self.color_cap:
                self.color_cap.release()
                self.color_cap = None
            raise RuntimeError(f"Failed to open COLOR stream: {str(e)}")

    def _get_color_frame(self) -> Optional[ColorFrame]:
        """Get frame from OpenCV color camera with robust error handling"""
        if not self.color_cap or not self.color_cap.isOpened():
            return None
        
        # Try to read frame with retry
        for attempt in range(3):  # Try up to 3 times
            ret, frame = self.color_cap.read()
            if ret and frame is not None:
                self.frame_count += 1
                timestamp = int(time.time() * 1_000_000)  # Use system time in microseconds
                return ColorFrame(frame, timestamp, self.frame_count)
                
            if attempt < 2:  # Don't sleep on last attempt
                time.sleep(0.01)
        
        return None
    
            
    def _open_regular_stream(self, stream_type: StreamType):
        """Open a non-color stream (depth/IR)"""
        try:
            print(f"Opening {stream_type.name} stream...")
            stream_ptr = c_void_p()
            ret = self.lib.imiOpenStream(self.device, stream_type.value, None, None, 
                                    byref(stream_ptr))
            
            if ret != self.IMI_SUCCESS:
                raise RuntimeError(f"Failed to open {stream_type.name} stream: {ret}")
                
            self.streams[stream_type] = stream_ptr
            print(f"Successfully opened {stream_type.name} stream")
            
        except Exception as e:
            raise RuntimeError(f"Failed to open {stream_type.name} stream: {str(e)}")

    def get_frame(self, stream_type: StreamType, timeout_ms: int = 1000) -> Optional[Union[DepthFrame, ColorFrame]]:
        """Get frame from specified stream"""
        if stream_type not in self.streams:
            return None
            
        # Handle OpenCV color stream
        if stream_type == StreamType.COLOR:
            return self._get_color_frame()
                
        # Handle regular streams
        return self._get_regular_frame(stream_type, timeout_ms)

    def _get_regular_frame(self, stream_type: StreamType, timeout_ms: int) -> Optional[Union[DepthFrame, ColorFrame]]:
        """Get frame from regular stream"""
        frame_ptr = POINTER(ImiImageFrame)()
        ret = self.lib.imiReadNextFrame(self.streams[stream_type], 
                                      byref(frame_ptr), timeout_ms)
        
        if ret != self.IMI_SUCCESS or not frame_ptr:
            return None
            
        try:
            width = frame_ptr.contents.width
            height = frame_ptr.contents.height
            
            if stream_type == StreamType.DEPTH:
                data_ptr = cast(frame_ptr.contents.pData, POINTER(c_uint16))
                data = np.ctypeslib.as_array(data_ptr, shape=(height, width))
                frame = DepthFrame(np.copy(data),
                                 frame_ptr.contents.timeStamp,
                                 frame_ptr.contents.frameNum)
            else:
                data_ptr = cast(frame_ptr.contents.pData, POINTER(c_uint8))
                if frame_ptr.contents.pixelFormat == PixelFormat.YUV420SP.value:
                    # Handle YUV420SP to RGB conversion
                    yuv_data = np.ctypeslib.as_array(data_ptr, shape=(height * 3 // 2, width))
                    rgb_data = self._yuv420sp_to_rgb(yuv_data, width, height)
                    data = rgb_data
                else:
                    data = np.ctypeslib.as_array(data_ptr, shape=(height, width, 3))
                frame = ColorFrame(np.copy(data),
                                 frame_ptr.contents.timeStamp,
                                 frame_ptr.contents.frameNum)
            return frame
        finally:
            self.lib.imiReleaseFrame(byref(frame_ptr))

    def close(self) -> None:
        """Clean up all resources"""
        # Close OpenCV color stream if open
        if StreamType.COLOR in self.streams and self.color_cap:
            try:
                self.color_cap.release()
            except Exception as e:
                print(f"Warning: Error closing OpenCV camera: {str(e)}")
        self.color_cap = None
        
        # Close any regular streams
        for stream_type, stream in self.streams.items():
            if stream_type != StreamType.COLOR:
                try:
                    self.lib.imiCloseStream(stream)
                except Exception as e:
                    print(f"Warning: Error closing {stream_type.name} stream: {str(e)}")
        self.streams.clear()
        
        # Close main IMI device
        if self.device:
            try:
                self.lib.imiCloseDevice(self.device)
            except Exception as e:
                print(f"Warning: Error closing IMI device: {str(e)}")
        self.device = None
        
        # Cleanup SDK
        try:
            self.lib.imiDestroy()
        except Exception as e:
            print(f"Warning: Error cleaning up IMI SDK: {str(e)}")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Export necessary classes
__all__ = [
    'ImiCamera',
    'StreamType',
    'PixelFormat',
    'FrameMode',
    'DepthFrame',
    'ColorFrame',
    'ImiFrame'
]