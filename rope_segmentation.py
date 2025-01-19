import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List

@dataclass
class SegmentationConfig:
    """Configuration parameters for rope segmentation"""
    # Background removal parameters
    plane_distance_threshold: float = 10.0  # mm
    min_plane_points: int = 1000
    
    # Color segmentation parameters
    blur_kernel_size: Tuple[int, int] = (5, 5)
    white_threshold: int = 240  # for background detection
    rope_color_threshold: int = 50  # for rope detection
    
    # Depth processing parameters
    min_depth: float = 100.0  # mm
    max_depth: float = 1000.0  # mm
    depth_scale: float = 1.0  # conversion factor to mm
    
    # Region growing parameters
    region_threshold: float = 5.0  # mm
    min_region_size: int = 50
    max_region_size: int = 5000

class RopeSegmenter:
    def __init__(self, config: Optional[SegmentationConfig] = None):
        """Initialize the rope segmenter with configuration"""
        self.config = config or SegmentationConfig()
        
    def remove_background(self, depth_frame: np.ndarray) -> np.ndarray:
        """Remove the background plane using RANSAC plane fitting"""
        # Convert depth to 3D points
        rows, cols = depth_frame.shape
        y, x = np.mgrid[0:rows, 0:cols]
        valid_points = depth_frame > 0
        points = np.column_stack((
            x[valid_points],
            y[valid_points],
            depth_frame[valid_points] * self.config.depth_scale
        ))
        
        if len(points) < self.config.min_plane_points:
            return np.zeros_like(depth_frame)
        
        # RANSAC plane fitting
        best_inliers = None
        best_plane = None
        n_iterations = 100
        
        for _ in range(n_iterations):
            # Randomly sample 3 points
            sample_indices = np.random.choice(len(points), 3, replace=False)
            sample_points = points[sample_indices]
            
            # Calculate plane equation ax + by + cz + d = 0
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            normal = np.cross(v1, v2)
            if np.all(normal == 0):
                continue
            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, sample_points[0])
            
            # Find inliers
            distances = np.abs(np.dot(points, normal) + d)
            inliers = distances < self.config.plane_distance_threshold
            
            if best_inliers is None or np.sum(inliers) > np.sum(best_inliers):
                best_inliers = inliers
                best_plane = (normal, d)
        
        # Create background mask
        background_mask = np.zeros_like(depth_frame)
        background_mask[y[valid_points][best_inliers], x[valid_points][best_inliers]] = 1
        
        # Remove background from depth frame
        filtered_depth = depth_frame.copy()
        filtered_depth[background_mask.astype(bool)] = 0
        return filtered_depth

    def segment_rope_color(self, color_frame: np.ndarray) -> np.ndarray:
        """Segment rope using color information"""
        # Convert to grayscale if needed
        if len(color_frame.shape) == 3:
            gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = color_frame
            
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, self.config.blur_kernel_size, 0)
        
        # Detect white background
        _, background_mask = cv2.threshold(
            blurred, 
            self.config.white_threshold, 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Invert to get potential rope pixels
        rope_mask = cv2.bitwise_not(background_mask)
        
        # Apply threshold to isolate rope
        _, rope_mask = cv2.threshold(
            blurred,
            self.config.rope_color_threshold,
            255,
            cv2.THRESH_BINARY_INV
        )
        
        return rope_mask

    def region_growing(self, depth_frame: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
        """Perform region growing to segment the rope into regions"""
        visited = np.zeros_like(mask, dtype=bool)
        regions = []
        rows, cols = mask.shape
        
        def valid_neighbor(r: int, c: int) -> bool:
            return (0 <= r < rows and 0 <= c < cols and 
                   mask[r, c] and not visited[r, c])
        
        def grow_region(start_r: int, start_c: int) -> np.ndarray:
            region = np.zeros_like(mask, dtype=bool)
            stack = [(start_r, start_c)]
            ref_depth = depth_frame[start_r, start_c]
            
            while stack:
                r, c = stack.pop()
                if not valid_neighbor(r, c):
                    continue
                    
                depth_diff = abs(depth_frame[r, c] - ref_depth)
                if depth_diff > self.config.region_threshold:
                    continue
                    
                visited[r, c] = True
                region[r, c] = True
                
                # Add neighbors to stack
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if valid_neighbor(nr, nc):
                        stack.append((nr, nc))
            
            return region
        
        # Find seed points and grow regions
        for r in range(rows):
            for c in range(cols):
                if mask[r, c] and not visited[r, c]:
                    region = grow_region(r, c)
                    if self.config.min_region_size <= np.sum(region) <= self.config.max_region_size:
                        regions.append(region)
        
        return regions

    def segment_rope(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Main method to segment the rope using both color and depth information"""
        # Remove background plane
        filtered_depth = self.remove_background(depth_frame)
        
        # Get rope mask from color image
        rope_mask = self.segment_rope_color(color_frame)
        
        # Combine depth and color information
        combined_mask = np.logical_and(
            rope_mask,
            np.logical_and(
                filtered_depth > self.config.min_depth,
                filtered_depth < self.config.max_depth
            )
        ).astype(np.uint8) * 255
        
        # Perform region growing
        regions = self.region_growing(filtered_depth, combined_mask)
        
        return regions, combined_mask

    def visualize_segments(self, color_frame: np.ndarray, regions: List[np.ndarray]) -> np.ndarray:
        """Visualize the segmented regions"""
        visualization = color_frame.copy()
        
        # Generate random colors for regions
        colors = np.random.randint(0, 255, (len(regions), 3))
        
        # Draw each region with a different color
        for i, region in enumerate(regions):
            visualization[region] = colors[i]
            
        return visualization