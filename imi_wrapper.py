import cv2
import numpy as np
import os
from ctypes import *
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Union
from enum import Enum

# SDK Structure definitions
class ImiFrameMode(Structure):
    """Frame mode structure from IMI SDK"""
    _fields_ = [
        ("pixelFormat", c_uint32),
        ("resolutionX", c_int16),
        ("resolutionY", c_int16),
        ("bitsPerPixel", c_int8),
        ("framerate", c_int8)
    ]

class ImiImageFrame(Structure):
    """Image frame structure from IMI SDK"""
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

# class ImiCameraFrameMode(Structure):
#     """UVC camera frame mode structure"""
#     _fields_ = [
#         ("pixelFormat", c_uint32),
#         ("resolutionX", c_int16),
#         ("resolutionY", c_int16),
#         ("fps", c_int8)
#     ]

class ImiCameraFrameMode(Structure):
    """UVC camera frame mode structure - matches ImiCameraDefines.h"""
    _fields_ = [
        ("pixelFormat", c_uint32),  # Changed from c_uint32
        ("resolutionX", c_uint16),  # Changed from c_int16
        ("resolutionY", c_uint16),  # Changed from c_int16
        ("fps", c_uint8)            # Changed from c_int8
    ]

class ImiCameraFrame(Structure):
    """UVC camera frame structure"""
    _fields_ = [
        ("frameNum", c_uint32),
        ("timeStamp", c_uint64),
        ("width", c_int32),
        ("height", c_int32),
        ("pData", POINTER(c_ubyte)),
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

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    k1: float = 0.0  # Radial distortion 1
    k2: float = 0.0  # Radial distortion 2
    p1: float = 0.0  # Tangential distortion 1
    p2: float = 0.0  # Tangential distortion 2
    k3: float = 0.0  # Radial distortion 3
    width: int = 0   # Image width
    height: int = 0  # Image height

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
    """Enhanced interface for IMI depth camera with UVC support"""
    
    # SDK Constants
    IMI_SUCCESS = 0
    IMI_PROPERTY_DEPTH_INTRINSICS = 0x36
    IMI_PROPERTY_COLOR_INTRINSICS = 0x1d
        
    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            sdk_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.imi_lib_path = os.path.join(sdk_dir, 'libs', 'libiminect.so')
            # Update the camera library filename to match the actual file
            self.cam_lib_path = os.path.join(sdk_dir, 'libs', 'libImiCamera.so')
        
        if not os.path.exists(self.imi_lib_path):
            raise RuntimeError(f"IMI SDK library not found at {self.imi_lib_path}")
            
        self.lib = CDLL(self.imi_lib_path)
        
        # Try to load camera library (optional)
        try:
            self.cam_lib = CDLL(self.cam_lib_path)
            self.has_camera_lib = True
        except OSError as e:
            print(f"Note: UVC camera library not available: {str(e)}")
            self.cam_lib = None
            self.has_camera_lib = False
            
        self._setup_functions()
        
        self.device = None
        self.color_device = None  # UVC handle
        self.streams = {}
        self.intrinsics = {}
        self.is_uvc_color = False

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

        if self.has_camera_lib:
            self.cam_lib.imiCamGetCurrentFrameMode.argtypes = [c_void_p]
            self.cam_lib.imiCamGetCurrentFrameMode.restype = POINTER(ImiCameraFrameMode)
            
            # Error handling functions
            self.lib.imiGetLastError.argtypes = []
            self.lib.imiGetLastError.restype = c_int32
            
            self.lib.imiGetErrorString.argtypes = [c_int32]
            self.lib.imiGetErrorString.restype = c_char_p
            
            # Camera functions - original open
            self.cam_lib.imiCamOpen.argtypes = [POINTER(c_void_p)]
            self.cam_lib.imiCamOpen.restype = c_int32
            
            # Camera functions - open with device selection
            self.cam_lib.imiCamOpen2.argtypes = [
                c_int32,     # vid
                c_int32,     # pid
                c_int32,     # fd
                c_int32,     # busnum
                c_int32,     # devaddr
                c_char_p,    # usbfs
                POINTER(c_void_p)  # pCameraDevice
            ]
            self.cam_lib.imiCamOpen2.restype = c_int32
            
            self.cam_lib.imiCamClose.argtypes = [c_void_p]
            self.cam_lib.imiCamClose.restype = c_int32
            
            self.cam_lib.imiCamStartStream.argtypes = [c_void_p, POINTER(ImiCameraFrameMode)]
            self.cam_lib.imiCamStartStream.restype = c_int32
            
            self.cam_lib.imiCamStopStream.argtypes = [c_void_p]
            self.cam_lib.imiCamStopStream.restype = c_int32
            
            self.cam_lib.imiCamReadNextFrame.argtypes = [c_void_p, POINTER(POINTER(ImiCameraFrame)), c_int32]
            self.cam_lib.imiCamReadNextFrame.restype = c_int32
            
            self.cam_lib.imiCamReleaseFrame.argtypes = [POINTER(POINTER(ImiCameraFrame))]
            self.cam_lib.imiCamReleaseFrame.restype = c_int32

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
        
        # Try to open UVC color camera if available
        if self.has_camera_lib:
            try:
                print("Looking for color camera...")
                color_device_ptr = c_void_p()
                
                # Use the basic open function first
                ret = self.cam_lib.imiCamOpen(byref(color_device_ptr))
                
                if ret == self.IMI_SUCCESS and color_device_ptr:
                    self.color_device = color_device_ptr
                    print("Successfully opened UVC color camera")
                    
                    # Try to get device info
                    try:
                        # Get supported modes to verify device
                        modes_ptr = POINTER(ImiCameraFrameMode)()
                        num_modes = c_uint32()
                        ret = self.cam_lib.imiCamGetSupportFrameModes(self.color_device, 
                                                                    byref(modes_ptr), 
                                                                    byref(num_modes))
                        if ret == self.IMI_SUCCESS:
                            print(f"Successfully queried {num_modes.value} supported modes")
                        else:
                            print(f"Note: Could not query supported modes (error: {ret})")
                            
                    except Exception as e:
                        print(f"Warning: Error querying device info: {str(e)}")
                else:
                    print(f"Note: UVC color camera initialization failed (error code: {ret})")
                    try:
                        error_str = self.lib.imiGetErrorString(ret)
                        print(f"Error description: {error_str}")
                    except Exception:
                        pass
                        
            except Exception as e:
                print(f"Note: Failed to initialize UVC camera: {str(e)}")
                import traceback
                traceback.print_exc()

    def open_stream(self, stream_type: StreamType) -> None:
        """Open stream of specified type"""
        if stream_type in self.streams:
            return  # Stream already open
            
        if stream_type == StreamType.COLOR:
            self._open_color_stream()
        else:
            self._open_regular_stream(stream_type)
            
    def _open_color_stream(self):
        """Attempt to open color stream through UVC or regular interface"""
        # First try UVC
        if self.has_camera_lib and self.color_device:
            try:
                print("Attempting to start UVC color stream...")
                
                # Query supported frame modes first
                modes_ptr = POINTER(ImiCameraFrameMode)()
                num_modes = c_uint32()
                ret = self.cam_lib.imiCamGetSupportFrameModes(self.color_device, 
                                                            byref(modes_ptr), 
                                                            byref(num_modes))
                if ret != self.IMI_SUCCESS:
                    raise RuntimeError(f"Failed to query supported modes: {ret}")
                    
                print(f"Found {num_modes.value} supported modes:")
                
                # Choose a supported mode - prefer 640x480 RGB888 @ 30fps
                selected_mode = None
                for i in range(num_modes.value):
                    mode = modes_ptr[i]
                    print(f"Mode {i}: {mode.resolutionX}x{mode.resolutionY} @ {mode.fps}fps "
                        f"format={mode.pixelFormat}")
                        
                    if (mode.resolutionX == 640 and mode.resolutionY == 480 and
                        mode.fps == 30 and mode.pixelFormat == 0):  # RGB888
                        selected_mode = mode
                        break
                        
                if not selected_mode:
                    # Fall back to first available mode
                    selected_mode = modes_ptr[0]
                    
                print(f"Selected mode: {selected_mode.resolutionX}x{selected_mode.resolutionY} "
                    f"@ {selected_mode.fps}fps")
                    
                # Start stream with selected mode
                ret = self.cam_lib.imiCamStartStream(self.color_device, byref(selected_mode))
                if ret != self.IMI_SUCCESS:
                    raise RuntimeError(f"Failed to start stream with selected mode: {ret}")
                    
                print("Successfully started UVC color stream")
                self.streams[StreamType.COLOR] = self.color_device
                self.is_uvc_color = True
                
                # Store the selected mode for later use
                self._uvc_mode = selected_mode
                return

            except Exception as e:
                print(f"Failed to initialize UVC color stream: {str(e)}")
                print("Falling back to regular color stream...")

        # Fallback to regular color stream
        try:
            stream_ptr = c_void_p()
            ret = self.lib.imiOpenStream(self.device, StreamType.COLOR.value, None, None, 
                                    byref(stream_ptr))
            if ret == self.IMI_SUCCESS:
                self.streams[StreamType.COLOR] = stream_ptr
                self.is_uvc_color = False
                print("Successfully opened regular color stream")
                return
                
            raise RuntimeError(f"Failed to open color stream: {ret}")
        except Exception as e:
            raise RuntimeError(f"Failed to open COLOR stream: {str(e)}")

        
    def _open_regular_stream(self, stream_type: StreamType):
        """Open a non-color stream (depth/IR)"""
        try:
            print(f"Opening {stream_type.name} stream...")
            stream_ptr = c_void_p()
            ret = self.lib.imiOpenStream(self.device, stream_type.value, None, None, 
                                    byref(stream_ptr))
            
            if ret != self.IMI_SUCCESS:
                error_str = None
                try:
                    error_str = self.lib.imiGetErrorString(ret)
                except:
                    pass
                raise RuntimeError(f"Failed to open {stream_type.name} stream. Error: {ret} ({error_str})")
                
            self.streams[stream_type] = stream_ptr
            print(f"Successfully opened {stream_type.name} stream")
            
        except Exception as e:
            raise RuntimeError(f"Failed to open {stream_type.name} stream: {str(e)}")

    def get_frame(self, stream_type: StreamType, timeout_ms: int = 1000) -> Optional[Union[DepthFrame, ColorFrame]]:
        """Get frame from specified stream"""
        if stream_type not in self.streams:
            return None
            
        # Handle UVC color stream
        if stream_type == StreamType.COLOR and self.is_uvc_color:
            return self._get_uvc_color_frame(timeout_ms)
                
        # Handle regular streams (including non-UVC color)
        return self._get_regular_frame(stream_type, timeout_ms)

    def _get_uvc_color_frame(self, timeout_ms: int) -> Optional[ColorFrame]:
        """Get frame from UVC color camera with enhanced error handling"""
        if not self.color_device or not hasattr(self, '_uvc_mode'):
            return None
                
        frame_ptr = POINTER(ImiCameraFrame)()
        try:
            ret = self.cam_lib.imiCamReadNextFrame(self.color_device, byref(frame_ptr), timeout_ms)
                
            if ret != self.IMI_SUCCESS or not frame_ptr:
                if ret != self.IMI_SUCCESS:
                    print(f"Failed to read frame, error: {ret}")
                return None
                    
            if not frame_ptr.contents:
                print("Null frame contents")
                return None
                    
            # Use the stored mode information to validate frame
            width = self._uvc_mode.resolutionX
            height = self._uvc_mode.resolutionY
            expected_size = width * height * 3  # RGB888 format
                    
            if frame_ptr.contents.size != expected_size:
                print(f"Unexpected frame size. Expected {expected_size}, got {frame_ptr.contents.size}")
                return None
                    
            # Create numpy array from RGB data
            try:
                data = np.ctypeslib.as_array(frame_ptr.contents.pData, 
                                            shape=(height, width, 3)).copy()
                return ColorFrame(data, 
                                frame_ptr.contents.timeStamp,
                                frame_ptr.contents.frameNum)
            except Exception as e:
                print(f"Error creating frame array: {str(e)}")
                return None
                    
        except Exception as e:
            print(f"Exception in _get_uvc_color_frame: {str(e)}")
            return None
        finally:
            if frame_ptr:
                try:
                    self.cam_lib.imiCamReleaseFrame(byref(frame_ptr))
                except Exception as e:
                    print(f"Error releasing frame: {str(e)}")

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

    def _yuv420sp_to_rgb(self, yuv420sp: np.ndarray, width: int, height: int) -> np.ndarray:
        """Convert YUV420SP format to RGB"""
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        frameSize = width * height

        for j in range(height):
            yp = j * width
            uvp = frameSize + (j >> 1) * width
            u = v = 0

            for i in range(width):
                y = (0xff & yuv420sp[yp]) - 16
                if y < 0:
                    y = 0

                if (i & 1) == 0:
                    v = (0xff & yuv420sp[uvp]) - 128
                    u = (0xff & yuv420sp[uvp + 1]) - 128
                    uvp += 2

                y1192 = 1192 * y
                r = (y1192 + 1634 * v)
                g = (y1192 - 833 * v - 400 * u)
                b = (y1192 + 2066 * u)

                r = min(max(r, 0), 262143) >> 10
                g = min(max(g, 0), 262143) >> 10
                b = min(max(b, 0), 262143) >> 10

                rgb[j, i] = [r, g, b]
                yp += 1

        return rgb

    def close(self) -> None:
        """Clean up all resources"""
        # Close any open streams
        for stream_type, stream in self.streams.items():
            try:
                if stream_type == StreamType.COLOR and self.is_uvc_color:
                    if self.color_device and self.has_camera_lib:
                        self.cam_lib.imiCamStopStream(self.color_device)
                else:
                    self.lib.imiCloseStream(stream)
            except Exception as e:
                print(f"Warning: Error closing {stream_type.name} stream: {str(e)}")
        self.streams.clear()
        
        # Close UVC color device if open
        if self.color_device and self.has_camera_lib:
            try:
                self.cam_lib.imiCamClose(self.color_device)
            except Exception as e:
                print(f"Warning: Error closing UVC camera: {str(e)}")
        self.color_device = None
        
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
    'ImiFrame',
    'CameraIntrinsics'
]