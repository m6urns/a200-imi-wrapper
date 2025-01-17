import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from enum import Enum

class ColorMap(Enum):
    """Available colormaps for depth visualization"""
    JET = cv2.COLORMAP_JET
    VIRIDIS = cv2.COLORMAP_VIRIDIS
    MAGMA = cv2.COLORMAP_MAGMA
    PLASMA = cv2.COLORMAP_PLASMA
    TURBO = cv2.COLORMAP_TURBO

@dataclass
class VisualizationConfig:
    """Configuration for depth visualization"""
    min_depth: float = 0.0
    max_depth: float = 10000.0
    auto_range: bool = True
    colormap: ColorMap = ColorMap.TURBO
    show_histogram: bool = True
    show_info: bool = True
    window_name: str = "Depth"
    histogram_window: str = "Histogram"
    percentile_min: float = 1.0
    percentile_max: float = 99.0

class FrameVisualizer:
    """Visualizer for depth and RGB frames"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize visualizer with optional config"""
        self.config = config or VisualizationConfig()
        self.last_frame_time = time.time()
        self.running_fps = 0
        self.windows_created = False
        self._create_windows()
        
    def _create_windows(self):
        """Create visualization windows"""
        if not self.windows_created:
            # Create window with specific flags for better scaling
            cv2.namedWindow(self.config.window_name, 
                          cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
            
            # Set initial window size
            cv2.resizeWindow(self.config.window_name, 1280, 720)
            
            if self.config.show_histogram:
                cv2.namedWindow(self.config.histogram_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.config.histogram_window, 400, 200)
            
            self.windows_created = True
            
            # Move windows to convenient positions
            cv2.moveWindow(self.config.window_name, 0, 0)
            if self.config.show_histogram:
                cv2.moveWindow(self.config.histogram_window, 1280, 0)
            
    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        self.running_fps = 0.9 * self.running_fps + 0.1 * (1 / (current_time - self.last_frame_time))
        self.last_frame_time = current_time
        
    def _create_histogram(self, depth_frame: np.ndarray) -> np.ndarray:
        """Create depth histogram visualization
        
        Args:
            depth_frame: Raw depth data
            
        Returns:
            Histogram visualization as RGB image
        """
        hist = cv2.calcHist([depth_frame], [0], None, [256], [0, self.config.max_depth])
        hist_img = np.zeros((200, 256), dtype=np.uint8)
        cv2.normalize(hist, hist, 0, 200, cv2.NORM_MINMAX)
        
        # Draw histogram bars
        for i in range(256):
            if hist[i] > 0:
                cv2.line(hist_img, (i, 200), (i, 200 - int(hist[i])), 255)
        
        # Add color
        return cv2.applyColorMap(hist_img, cv2.COLORMAP_HOT)
        
    def _create_info_overlay(self, depth_frame: np.ndarray, valid_mask: np.ndarray) -> List[str]:
        """Create information overlay text
        
        Args:
            depth_frame: Raw depth data
            valid_mask: Boolean mask of valid pixels
            
        Returns:
            List of text strings to display
        """
        return [
            f"FPS: {self.running_fps:.1f}",
            f"Range: {self.config.min_depth:.0f}-{self.config.max_depth:.0f}mm",
            f"Valid pixels: {(np.sum(valid_mask)/valid_mask.size*100):.1f}%",
            f"Mean depth: {np.mean(depth_frame[valid_mask]):.0f}mm",
            "Auto range: ON" if self.config.auto_range else "Auto range: OFF",
            "'a': Toggle auto range",
            "'h': Toggle histogram",
            "'c': Change colormap",
            "'s': Save frame",
            "'q': Quit"
        ]
    
    def visualize_depth(self, depth_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create visualization of depth frame
        
        Args:
            depth_frame: Raw depth data
            
        Returns:
            Tuple of (colorized depth image, normalized depth image)
        """
        if depth_frame is None:
            return None, None

        self._update_fps()
        
        # Create mask for valid depth values
        valid_mask = (depth_frame > 0)
        if self.config.auto_range and np.sum(valid_mask) > 0:
            self.config.min_depth = np.percentile(depth_frame[valid_mask], 
                                                self.config.percentile_min)
            self.config.max_depth = np.percentile(depth_frame[valid_mask], 
                                                self.config.percentile_max)
        
        valid_mask &= (depth_frame >= self.config.min_depth) & (depth_frame <= self.config.max_depth)
        
        if np.sum(valid_mask) > 0:
            # Normalize valid depths
            normalized = np.zeros_like(depth_frame, dtype=np.float32)
            normalized[valid_mask] = ((depth_frame[valid_mask] - self.config.min_depth) * 255 / 
                                    (self.config.max_depth - self.config.min_depth))
            
            # Clip to 0-255 range and convert to uint8
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
            
            # Apply colormap
            depth_colormap = cv2.applyColorMap(normalized, self.config.colormap.value)
            
            # Make invalid pixels black
            depth_colormap[~valid_mask] = 0
            
            # Scale image to fit window (maintain aspect ratio)
            window_width = 1280
            window_height = 720
            scale = min(window_width / depth_colormap.shape[1],
                       window_height / depth_colormap.shape[0])
            dim = (int(depth_colormap.shape[1] * scale),
                  int(depth_colormap.shape[0] * scale))
            depth_colormap = cv2.resize(depth_colormap, dim, 
                                      interpolation=cv2.INTER_AREA)
            
            # Add information overlay
            if self.config.show_info:
                info_text = self._create_info_overlay(depth_frame, valid_mask)
                y = 30
                for text in info_text:
                    cv2.putText(depth_colormap, text, (10, y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(depth_colormap, text, (10, y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                    y += 25
            
            return depth_colormap, normalized
        return None, None

    def show(self, depth_frame: np.ndarray, rgb_frame: Optional[np.ndarray] = None):
        """Display depth and optional RGB frames
        
        Args:
            depth_frame: Raw depth data
            rgb_frame: Optional RGB frame to display
        
        Returns:
            Key pressed (if any) during visualization
        """
        depth_colormap, normalized = self.visualize_depth(depth_frame)
        
        if depth_colormap is not None:
            cv2.imshow(self.config.window_name, depth_colormap)
            
            if self.config.show_histogram:
                hist_img = self._create_histogram(depth_frame)
                cv2.imshow(self.config.histogram_window, hist_img)
                
            if rgb_frame is not None:
                cv2.imshow("RGB", rgb_frame)
                
        key = cv2.waitKey(1) & 0xFF
        self._handle_key(key)
        return key
    
    def _handle_key(self, key: int):
        """Handle keyboard input
        
        Args:
            key: Key code from cv2.waitKey()
        """
        if key == ord('a'):
            self.config.auto_range = not self.config.auto_range
        elif key == ord('h'):
            self.config.show_histogram = not self.config.show_histogram
            if not self.config.show_histogram:
                cv2.destroyWindow(self.config.histogram_window)
        elif key == ord('c'):
            # Cycle through colormaps
            colormaps = list(ColorMap)
            current_idx = colormaps.index(self.config.colormap)
            self.config.colormap = colormaps[(current_idx + 1) % len(colormaps)]
            
    def save_frame(self, depth_frame: np.ndarray, depth_colormap: np.ndarray, 
                  rgb_frame: Optional[np.ndarray] = None):
        """Save current frames to disk
        
        Args:
            depth_frame: Raw depth data
            depth_colormap: Colorized depth visualization
            rgb_frame: Optional RGB frame
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        np.save(f'depth_frame_{timestamp}.npy', depth_frame)
        cv2.imwrite(f'depth_viz_{timestamp}.png', depth_colormap)
        if rgb_frame is not None:
            cv2.imwrite(f'rgb_frame_{timestamp}.png', rgb_frame)
        print(f"Frames saved with timestamp {timestamp}")
        
    def close(self):
        """Clean up visualization windows"""
        cv2.destroyAllWindows()
        self.windows_created = False

# Example usage:
if __name__ == "__main__":
    from imi_camera import ImiCamera
    
    config = VisualizationConfig(
        min_depth=0,
        max_depth=10000,
        auto_range=True,
        colormap=ColorMap.TURBO,
        show_histogram=True
    )
    
    viz = FrameVisualizer(config)
    
    try:
        with ImiCamera() as camera:
            while True:
                frame = camera.get_frame()
                if frame is not None:
                    key = viz.show(frame.data)
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        depth_colormap, _ = viz.visualize_depth(frame.data)
                        if depth_colormap is not None:
                            viz.save_frame(frame.data, depth_colormap)
                            
    finally:
        viz.close()