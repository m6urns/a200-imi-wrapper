import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List

@dataclass
class SegmentationConfig:
    """Configuration parameters for rope segmentation"""
    # Background removal parameters
    plane_distance_threshold: float = 5 # mm
    min_plane_points: int = 500
    
    # Color segmentation parameters
    blur_kernel_size: Tuple[int, int] = (7, 7)
    white_threshold: int = 50  # for background detection
    rope_color_threshold: int = 50  # for rope detection
    
    # Depth processing parameters
    min_depth: float = 100.0  # mm
    max_depth: float = 500.0  # mm
    depth_scale: float = 1.0  # conversion factor to mm
    
    # Region growing parameters
    region_threshold: float = 5.0  # mm
    min_region_size: int = 50
    max_region_size: int = 8000

class RopeSegmenter:
    def __init__(self, config: Optional[SegmentationConfig] = None, buffer_size: int = 5):
        """Initialize the rope segmenter with configuration"""
        self.config = config or SegmentationConfig()
        self.buffer_size = buffer_size
        self.skeleton_buffer = []
        self.ordered_points_buffer = []
        
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

    def segment_rope(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[Tuple[int, int]]]:
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
        
        # Generate and smooth skeleton
        current_skeleton = self.skeletonize(combined_mask)
        smoothed_skeleton = self.get_averaged_skeleton(current_skeleton)
        
        # Get endpoints and ordered points
        endpoints = self.find_endpoints(smoothed_skeleton)
        current_points = self.order_skeleton_points(smoothed_skeleton, endpoints)
        smoothed_points = self.smooth_ordered_points(current_points)
        
        return regions, combined_mask, smoothed_skeleton, smoothed_points

    def skeletonize(self, mask: np.ndarray) -> np.ndarray:
        """Generate skeleton from binary mask using morphological operations"""
        # Ensure binary image
        mask = mask.astype(np.uint8)
        if mask.max() > 1:
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
            
        # Initialize skeleton
        skeleton = np.zeros_like(mask)
        
        # Create a cross-shaped kernel for erosion and dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        
        while True:
            # Erode the image
            eroded = cv2.erode(mask, kernel)
            # Dilate the eroded image
            dilated = cv2.dilate(eroded, kernel)
            # Get the difference between the original and the dilated image
            diff = cv2.subtract(mask, dilated)
            # Add the difference to the skeleton
            skeleton = cv2.bitwise_or(skeleton, diff)
            # Update the mask
            mask = eroded.copy()
            
            # If no white pixels are left in the mask, we're done
            if cv2.countNonZero(mask) == 0:
                break
                
        return skeleton * 255  # Convert back to 0-255 range

    def find_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Find endpoints in the skeleton"""
        # Get coordinates of all skeleton points
        y, x = np.nonzero(skeleton)
        endpoints = []
        
        for i, j in zip(y, x):
            # Get 3x3 neighborhood
            patch = skeleton[max(0, i-1):min(i+2, skeleton.shape[0]),
                           max(0, j-1):min(j+2, skeleton.shape[1])]
            # Count neighbors
            neighbors = np.sum(patch) - 1  # subtract center point
            # If only one neighbor, it's an endpoint
            if neighbors == 1:
                endpoints.append((i, j))
                
        return endpoints

    def get_averaged_skeleton(self, current_skeleton: np.ndarray) -> np.ndarray:
        """Average the skeleton over recent frames"""
        # Add current skeleton to buffer
        self.skeleton_buffer.append(current_skeleton)
        
        # Keep buffer at specified size
        if len(self.skeleton_buffer) > self.buffer_size:
            self.skeleton_buffer.pop(0)
        
        # Average all skeletons in buffer
        if len(self.skeleton_buffer) > 0:
            # Stack all skeletons and take mean
            stacked = np.stack(self.skeleton_buffer)
            averaged = np.mean(stacked, axis=0)
            # Threshold to get binary skeleton
            _, averaged = cv2.threshold(averaged.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
            return averaged
        
        return current_skeleton

    def smooth_ordered_points(self, current_points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Smooth the ordered points over recent frames"""
        if not current_points:
            return current_points
            
        # Add current points to buffer
        self.ordered_points_buffer.append(current_points)
        
        # Keep buffer at specified size
        if len(self.ordered_points_buffer) > self.buffer_size:
            self.ordered_points_buffer.pop(0)
            
        # Need at least 2 frames to smooth
        if len(self.ordered_points_buffer) < 2:
            return current_points
            
        # Find the frame with closest number of points to current
        current_len = len(current_points)
        best_frame = min(self.ordered_points_buffer[:-1], 
                        key=lambda points: abs(len(points) - current_len))
        
        # If point counts match, do averaging
        if len(best_frame) == current_len:
            smoothed_points = []
            for i in range(current_len):
                y1, x1 = best_frame[i]
                y2, x2 = current_points[i]
                # Average the coordinates
                smoothed_points.append((
                    int((y1 + y2) / 2),
                    int((x1 + x2) / 2)
                ))
            return smoothed_points
            
        return current_points

    def order_skeleton_points(self, skeleton: np.ndarray, endpoints: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Order skeleton points from one endpoint to another"""
        if not endpoints:
            return []
            
        # Start from first endpoint
        ordered_points = [endpoints[0]]
        current = endpoints[0]
        visited = {endpoints[0]}
        
        while True:
            y, x = current
            # Get 3x3 neighborhood
            patch = skeleton[max(0, y-1):min(y+2, skeleton.shape[0]),
                           max(0, x-1):min(x+2, skeleton.shape[1])]
            py, px = max(0, y-1), max(0, x-1)
            
            # Find unvisited neighbors
            neighbors = []
            for i in range(patch.shape[0]):
                for j in range(patch.shape[1]):
                    if patch[i, j]:
                        point = (py + i, px + j)
                        if point != current and point not in visited:
                            neighbors.append(point)
            
            if not neighbors:
                break
                
            # Choose the closest neighbor
            current = min(neighbors, key=lambda p: abs(p[0]-y) + abs(p[1]-x))
            ordered_points.append(current)
            visited.add(current)
            
        return ordered_points

    def visualize_segments(self, color_frame: np.ndarray, regions: List[np.ndarray], 
                         skeleton: Optional[np.ndarray] = None, 
                         ordered_points: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
        """Visualize the segmented regions with optional skeleton overlay"""
        visualization = color_frame.copy()
        
        # Draw regions
        colors = np.random.randint(0, 255, (len(regions), 3))
        for i, region in enumerate(regions):
            visualization[region] = colors[i]
            
        # Draw skeleton if provided
        if skeleton is not None:
            visualization[skeleton > 0] = [0, 255, 0]  # Green skeleton
            
        # Draw ordered points if provided
        if ordered_points:
            for i in range(len(ordered_points) - 1):
                pt1 = ordered_points[i][::-1]  # Convert (y,x) to (x,y) for OpenCV
                pt2 = ordered_points[i+1][::-1]
                cv2.line(visualization, pt1, pt2, (255, 0, 0), 2)
                
            # Draw endpoints in red
            if ordered_points:
                start = ordered_points[0][::-1]
                end = ordered_points[-1][::-1]
                cv2.circle(visualization, start, 5, (0, 0, 255), -1)
                cv2.circle(visualization, end, 5, (0, 0, 255), -1)
            
        return visualization