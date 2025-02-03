#!/usr/bin/env python3
import cv2
import argparse
import time

def test_camera(index):
    """Test opening and reading from a camera"""
    print(f"\nTesting camera index {index}")
    
    # Try with default backend
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        # Try with V4L2 backend explicitly
        print("Trying V4L2 backend...")
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"Failed to open camera {index}")
        return
    
    # Get camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera opened successfully:")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    
    # Try to read a few frames
    print("\nTrying to read frames...")
    frames_read = 0
    start_time = time.time()
    
    try:
        while frames_read < 100:  # Read 100 frames or until 'q' pressed
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
                
            frames_read += 1
            elapsed = time.time() - start_time
            current_fps = frames_read / elapsed
            
            # Add FPS counter to frame
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(f"Camera {index}", frame)
            
            # Break if 'q' pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        print(f"Successfully read {frames_read} frames")
        print(f"Average FPS: {frames_read/elapsed:.1f}")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()

def scan_cameras(max_index=10):
    """Scan for available cameras"""
    print("Scanning for cameras...")
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"Camera {i}: {width}x{height} @ {fps}fps - WORKING")
            else:
                print(f"Camera {i}: Opens but cannot read frames")
        else:
            print(f"Camera {i}: Not available")
        cap.release()

def main():
    parser = argparse.ArgumentParser(description='Test OpenCV camera access')
    parser.add_argument('--scan', action='store_true', 
                       help='Scan for available cameras')
    parser.add_argument('--index', type=int, 
                       help='Test specific camera index')
    args = parser.parse_args()
    
    if args.scan:
        scan_cameras()
    elif args.index is not None:
        test_camera(args.index)
    else:
        print("Please specify --scan or --index")
        print("Example usage:")
        print("  ./opencv_test.py --scan")
        print("  ./opencv_test.py --index 0")

if __name__ == "__main__":
    main()