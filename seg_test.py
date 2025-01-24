#!/usr/bin/env python3

import cv2
import numpy as np
import time
import argparse
from imi_wrapper import ImiCamera, StreamType
from imi_visualization import FrameVisualizer, VisualizationConfig, ColorMap
from rope_segmentation import RopeSegmenter, SegmentationConfig

class RopeSegmentationTest:
    def __init__(self, color_index=None):
        """Initialize test suite"""
        self.camera = None
        self.viz = None
        self.segmenter = None
        self.recording = False
        self.record_frames = []
        self.color_index = color_index
        self.show_skeleton = True
        
        # Window names
        self.WINDOW_ORIGINAL = "Original Feed"
        self.WINDOW_SEGMENTED = "Segmented Rope"
        self.WINDOW_SKELETON = "Skeleton View"

    def initialize_camera(self):
        """Initialize and test camera connection"""
        print("\n=== Initializing Camera ===")
        try:
            self.camera = ImiCamera(color_index=self.color_index)
            self.camera.initialize()
            print("✓ Camera initialization successful")
            return True
        except Exception as e:
            print(f"✗ Camera initialization failed: {str(e)}")
            return False

    def initialize_segmenter(self):
        """Initialize the rope segmenter"""
        print("\n=== Initializing Rope Segmenter ===")
        try:
            config = SegmentationConfig(
                white_threshold=240,
                rope_color_threshold=250,
                region_threshold=5.0,
                min_depth=25,
                max_depth=1000,
            )
            self.segmenter = RopeSegmenter(config)
            print("✓ Segmenter initialization successful")
            return True
        except Exception as e:
            print(f"✗ Segmenter initialization failed: {str(e)}")
            return False

    def setup_visualization(self):
        """Initialize visualization windows and trackbars"""
        cv2.namedWindow(self.WINDOW_ORIGINAL)
        cv2.namedWindow(self.WINDOW_SEGMENTED)
        cv2.namedWindow(self.WINDOW_SKELETON)
        
        # Segmentation parameters
        cv2.createTrackbar('White Thresh', self.WINDOW_SEGMENTED, 
                          self.segmenter.config.white_threshold, 255, 
                          lambda x: self.update_config('white_threshold', x))
        cv2.createTrackbar('Rope Thresh', self.WINDOW_SEGMENTED,
                          self.segmenter.config.rope_color_threshold, 255,
                          lambda x: self.update_config('rope_color_threshold', x))
        cv2.createTrackbar('Region Thresh', self.WINDOW_SEGMENTED,
                          int(self.segmenter.config.region_threshold), 20,
                          lambda x: self.update_config('region_threshold', float(x)))

    def update_config(self, param_name, value):
        """Update segmentation configuration parameters"""
        setattr(self.segmenter.config, param_name, value)

    def process_frames(self, depth_frame, color_frame):
        """Process frames and show segmentation results"""
        if depth_frame is None or color_frame is None:
            return None

        try:
            # Get segmentation results
            regions, mask, skeleton, ordered_points = self.segmenter.segment_rope(
                color_frame.data, depth_frame.data)
            
            # Create visualizations
            vis_img = self.segmenter.visualize_segments(color_frame.data, regions)
            skeleton_vis = self.segmenter.visualize_segments(
                color_frame.data, regions, 
                skeleton=skeleton, 
                ordered_points=ordered_points
            )
            
            # Show original feed
            cv2.imshow(self.WINDOW_ORIGINAL, color_frame.data)
            
            # Show segmentation results
            debug_images = np.hstack([
                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                vis_img
            ])
            cv2.imshow(self.WINDOW_SEGMENTED, debug_images)
            
            # Show skeleton visualization
            cv2.imshow(self.WINDOW_SKELETON, skeleton_vis)
            
            return cv2.waitKey(1)
            
        except Exception as e:
            print(f"\nError processing frame: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def print_controls(self):
        """Print available keyboard controls"""
        print("\nStarting segmentation test...")
        print("Controls:")
        print("  'q': Quit")
        print("  'r': Toggle recording")
        print("  's': Save current frame and segmentation")
        print("  't': Toggle skeleton visibility")
        print("\nUse trackbars to adjust segmentation parameters")

    def save_results(self, depth_frame, color_frame):
        """Save the current frame and segmentation results"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Get current results
        regions, mask, skeleton, ordered_points = self.segmenter.segment_rope(
            color_frame.data, depth_frame.data)
        
        # Save original frames
        cv2.imwrite(f"color_{timestamp}.png", color_frame.data)
        cv2.imwrite(f"depth_{timestamp}.png", depth_frame.data)
        
        # Save segmentation results
        cv2.imwrite(f"mask_{timestamp}.png", mask)
        
        # Save visualization
        skeleton_vis = self.segmenter.visualize_segments(
            color_frame.data, regions,
            skeleton=skeleton,
            ordered_points=ordered_points
        )
        cv2.imwrite(f"skeleton_{timestamp}.png", skeleton_vis)
        
        print(f"\nSaved results with timestamp {timestamp}")

    def run_test(self):
        """Run the segmentation test"""
        try:
            if not self.initialize_camera():
                return
            if not self.initialize_segmenter():
                return
                
            self.setup_visualization()
            
            self.camera.open_stream(StreamType.DEPTH)
            self.camera.open_stream(StreamType.COLOR)
            
            self.print_controls()
            
            while True:
                depth_frame = self.camera.get_frame(StreamType.DEPTH)
                color_frame = self.camera.get_frame(StreamType.COLOR)
                
                key = self.process_frames(depth_frame, color_frame)
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_results(depth_frame, color_frame)
                elif key == ord('r'):
                    self.recording = not self.recording
                    print(f"\nRecording: {'Started' if self.recording else 'Stopped'}")
                elif key == ord('t'):
                    self.show_skeleton = not self.show_skeleton
                    
            print("\nTest completed successfully")
            
        except Exception as e:
            print(f"\n✗ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            if self.camera:
                self.camera.close()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Rope Segmentation Test Suite')
    parser.add_argument('--color-index', type=int, help='Specify color camera index')
    args = parser.parse_args()

    print("Starting Rope Segmentation Test Suite")
    print("====================================")
    
    test = RopeSegmentationTest(color_index=args.color_index)
    test.run_test()

if __name__ == "__main__":
    main()