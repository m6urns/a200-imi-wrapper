#!/usr/bin/env python3

import cv2
import numpy as np
import time
from imi_wrapper import ImiCamera, StreamType
from imi_visualization import FrameVisualizer, VisualizationConfig, ColorMap

class CameraTest:
    def __init__(self):
        """Initialize test suite"""
        self.camera = None
        self.viz = None
        self.recording = False
        self.record_frames = []

    def setup_visualization(self):
        """Initialize visualization configuration"""
        config = VisualizationConfig(
            min_depth=0,
            max_depth=10000,
            auto_range=True,
            colormap=ColorMap.TURBO,
            show_histogram=True,
            show_info=True
        )
        self.viz = FrameVisualizer(config)

    def initialize_camera(self):
        """Initialize and test camera connection"""
        print("\n=== Testing Camera Initialization ===")
        try:
            self.camera = ImiCamera()
            self.camera.initialize()
            print("✓ Camera initialization successful")
            return True
        except Exception as e:
            print(f"✗ Camera initialization failed: {str(e)}")
            return False

    def open_streams(self):
        """Open depth and color streams"""
        # Open depth stream (required)
        try:
            self.camera.open_stream(StreamType.DEPTH)
        except Exception as e:
            raise RuntimeError(f"Failed to open depth stream: {str(e)}")

        # Try to open color stream (optional)
        try:
            self.camera.open_stream(StreamType.COLOR)
            return True
        except Exception as e:
            print(f"Note: Color stream not available: {str(e)}")
            return False

    def process_frame(self, depth_frame, color_frame, fps_stats):
        """Process and visualize camera frames"""
        if depth_frame is None:
            return None

        # Update FPS calculation
        fps_stats['frames_processed'] += 1
        if fps_stats['frames_processed'] % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - fps_stats['last_time'])
            fps_stats['history'].append(fps)
            if len(fps_stats['history']) > 10:
                fps_stats['history'].pop(0)
            fps_stats['last_time'] = current_time
            avg_fps = sum(fps_stats['history'])/len(fps_stats['history'])
            print(f"\rFPS: {avg_fps:.1f}", end="")

        # Record if enabled
        if self.recording and color_frame is not None:
            self.record_frames.append({
                'depth': depth_frame.data.copy(),
                'color': color_frame.data.copy(),
                'timestamp': depth_frame.timestamp
            })

        # Visualize frames
        return self.viz.show(depth_frame.data, 
                           color_frame.data if color_frame else None)

    def handle_user_input(self, key, depth_frame, color_frame):
        """Handle keyboard input during streaming"""
        if key == ord('r'):
            self.recording = not self.recording
            if self.recording:
                print("\nStarted recording...")
            else:
                print(f"\nStopped recording. Captured {len(self.record_frames)} frames")
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            depth_colormap, _ = self.viz.visualize_depth(depth_frame.data)
            if depth_colormap is not None:
                self.viz.save_frame(depth_frame.data, depth_colormap,
                                  color_frame.data if color_frame else None)
                print(f"\nSaved frames with timestamp {timestamp}")
        return key == ord('q')

    def save_recording(self):
        """Save recorded frames if any exist"""
        if self.record_frames:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f'recorded_session_{timestamp}.npy'
            np.save(filename, self.record_frames)
            print(f"\nSaved recording to {filename}")

    def print_controls(self):
        """Print available keyboard controls"""
        print("\nStarting capture loop...")
        print("Controls:")
        print("  'q': Quit")
        print("  'r': Toggle recording")
        print("  's': Save current frame")
        print("  'a': Toggle auto-range")
        print("  'c': Cycle colormaps")

    def test_streaming(self):
        """Run streaming test with visualization"""
        print("\n=== Testing Streaming ===")
        try:
            self.setup_visualization()
            has_color = self.open_streams()
            self.print_controls()

            fps_stats = {
                'history': [],
                'last_time': time.time(),
                'frames_processed': 0
            }

            while True:
                # Get frames
                depth_frame = self.camera.get_frame(StreamType.DEPTH)
                color_frame = self.camera.get_frame(StreamType.COLOR) if has_color else None

                # Process frames
                key = self.process_frame(depth_frame, color_frame, fps_stats)
                if key is None:
                    continue

                # Handle user input
                if self.handle_user_input(key, depth_frame, color_frame):
                    print("\nStreaming test completed successfully")
                    return True

        except Exception as e:
            print(f"\n✗ Streaming test failed: {str(e)}")
            return False
        finally:
            if self.viz:
                self.viz.close()

    def run_all_tests(self):
        """Run complete test suite"""
        try:
            if not self.initialize_camera():
                return

            if not self.test_streaming():
                return

        finally:
            if self.camera:
                self.camera.close()
            if self.viz:
                self.viz.close()

            self.save_recording()

def main():
    print("Starting IMI Camera Test Suite")
    print("==============================")
    
    test = CameraTest()
    test.run_all_tests()

if __name__ == "__main__":
    main()