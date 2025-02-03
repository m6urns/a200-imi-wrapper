import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import cv2
import numpy as np
import torch
from torchvision import transforms
from datetime import datetime
import threading
import queue
import time
import argparse
from typing import Optional, Dict
from pydantic import BaseModel, Field
from pathlib import Path

from knot_classifier import DualStreamKnotClassifier
from imi_wrapper import ImiCamera, StreamType
from imi_visualization import FrameVisualizer, VisualizationConfig, ColorMap

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseModel):
    """API settings model with explicit parameter names and descriptions"""
    # Camera and hardware settings
    camera_index: int = Field(
        default=4,
        description="Index of the color camera to use",
        ge=0
    )
    
    # Model settings
    model_path: str = Field(
        default='best_model.pth',
        description="Path to the trained model weights file"
    )
    confidence_threshold: float = Field(
        default=0.4,
        description="Minimum confidence threshold for classifications",
        ge=0.0,
        le=1.0
    )
    
    # Visualization settings
    view_mode: str = Field(
        default="side-by-side",
        description="Visualization mode ('side-by-side' or 'overlay')"
    )
    auto_range: bool = Field(
        default=True,
        description="Automatically adjust depth range"
    )
    min_depth: int = Field(
        default=100,
        description="Minimum depth value in mm",
        ge=0
    )
    max_depth: int = Field(
        default=1000,
        description="Maximum depth value in mm",
        ge=0
    )
    
    # Image alignment settings
    vertical_shift: int = Field(
        default=71,
        description="Vertical shift for depth-color alignment in pixels"
    )
    horizontal_shift: int = Field(
        default=45,
        description="Horizontal shift for depth-color alignment in pixels"
    )
    alignment_mode: bool = Field(
        default=False,
        description="Enable alignment adjustment mode"
    )
    
    # Server settings
    host: str = Field(
        default="0.0.0.0",
        description="Host address to bind the server"
    )
    port: int = Field(
        default=8000,
        description="Port number for the server",
        ge=0,
        le=65535
    )

class KnotClassifierAPI:
    """API wrapper for knot classifier"""
    
    STAGES = ["loose", "loop", "complete", "tightened"]
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize API wrapper with settings"""
        self.settings = settings or Settings()
        self.latest_frame = None
        self.latest_classification = {
            "stage": "unknown",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        self.fps = 0
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Initialize visualization config
        self.viz_config = VisualizationConfig(
            min_depth=self.settings.min_depth,
            max_depth=self.settings.max_depth,
            auto_range=self.settings.auto_range,
            colormap=ColorMap.TURBO,
            show_histogram=False,
            show_info=False,
            view_mode=self.settings.view_mode,
            window_width=800,
            window_height=600
        )
        
        # Initialize transforms
        self.rgb_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.depth_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        # Initialize classifier
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        try:
            weights_path = Path(self.settings.model_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"Model weights not found at {self.settings.model_path}")
                
            self.classifier = DualStreamKnotClassifier(num_classes=len(self.STAGES))
            logger.info("Loading model weights...")
            model_state = torch.load(self.settings.model_path, map_location=self.device)
            self.classifier.load_state_dict(model_state)
            self.classifier.to(self.device)
            self.classifier.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Initialize camera
        try:
            self.camera = ImiCamera(color_index=self.settings.camera_index)
            self.camera.initialize()
            self.camera.open_stream(StreamType.DEPTH)
            self.camera.open_stream(StreamType.COLOR)
            logger.info(f"Camera initialized with index {self.settings.camera_index}")
        except Exception as e:
            logger.error(f"Camera initialization failed: {str(e)}")
            raise
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.is_running = True
        self.processing_thread.start()
        
    def preprocess_frames(self, rgb_frame, depth_frame):
        """Preprocess frames for model input"""
        # Process RGB
        rgb = cv2.cvtColor(rgb_frame.data, cv2.COLOR_BGR2RGB)
        rgb_tensor = self.rgb_transform(rgb).unsqueeze(0).to(self.device)
        
        # Process depth
        depth = depth_frame.data
        depth_min = depth[depth > 0].min() if np.any(depth > 0) else 0
        depth_max = depth.max()
        depth_normalized = np.zeros_like(depth, dtype=np.uint8)
        if depth_max > depth_min:
            valid_mask = depth > 0
            depth_normalized[valid_mask] = ((depth[valid_mask] - depth_min) * 255 / 
                                          (depth_max - depth_min))
        depth_tensor = self.depth_transform(depth_normalized).unsqueeze(0).to(self.device)
        
        return rgb_tensor, depth_tensor
        
    def process_frames(self):
        """Main processing loop"""
        last_time = time.time()
        frame_count = 0
        logger.info("Starting frame processing loop")
        
        while self.is_running:
            try:
                # Get frames
                depth_frame = self.camera.get_frame(StreamType.DEPTH)
                color_frame = self.camera.get_frame(StreamType.COLOR)
                
                if depth_frame is not None and color_frame is not None:
                    # Update FPS
                    frame_count += 1
                    if frame_count % 30 == 0:
                        current_time = time.time()
                        self.fps = 30 / (current_time - last_time)
                        last_time = current_time
                        logger.info(f"Processing frames at {self.fps:.1f} FPS")
                    
                    # Process frames for visualization
                    depth_viz = depth_frame.data.copy()
                    color_viz = color_frame.data.copy()
                    
                    # Get model prediction
                    with torch.no_grad():
                        rgb_tensor, depth_tensor = self.preprocess_frames(color_frame, depth_frame)
                        self.last_rgb_tensor = rgb_tensor
                        self.last_depth_tensor = depth_tensor
                        
                        outputs = self.classifier(rgb_tensor, depth_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        confidence = confidence.item()
                        predicted_idx = predicted.item()
                        
                        if confidence >= self.settings.confidence_threshold:
                            predicted_stage = self.STAGES[predicted_idx]
                        else:
                            predicted_stage = "unknown"
                    
                    self.latest_classification = {
                        "stage": predicted_stage,
                        "confidence": confidence,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Create visualization
                    confidence_color = (0, 255, 0) if confidence >= self.settings.confidence_threshold else (0, 165, 255)
                    cv2.putText(color_viz, f"Stage: {predicted_stage}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              confidence_color, 2)
                    cv2.putText(color_viz, f"Confidence: {confidence:.2f}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              confidence_color, 2)
                    cv2.putText(color_viz, f"FPS: {self.fps:.1f}", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              (255, 255, 255), 2)
                    
                    # Normalize and colorize depth frame
                    depth_min = depth_viz[depth_viz > 0].min() if np.any(depth_viz > 0) else 0
                    depth_max = depth_viz.max()
                    depth_normalized = np.zeros_like(depth_viz, dtype=np.uint8)
                    if depth_max > depth_min:
                        valid_mask = depth_viz > 0
                        depth_normalized[valid_mask] = ((depth_viz[valid_mask] - depth_min) * 255 / 
                                                      (depth_max - depth_min))
                    depth_colormap = cv2.applyColorMap(depth_normalized, self.viz_config.colormap.value)
                    
                    # Ensure same size before combining
                    if depth_colormap.shape[:2] != color_viz.shape[:2]:
                        depth_colormap = cv2.resize(depth_colormap, 
                                                  (color_viz.shape[1], color_viz.shape[0]))
                        
                            # Apply alignment shift
                    rows, cols = depth_colormap.shape[:2]
                    shift_matrix = np.float32([[1, 0, self.settings.horizontal_shift],
                                            [0, 1, self.settings.vertical_shift]])
                    depth_colormap = cv2.warpAffine(depth_colormap,
                                                shift_matrix,
                                                (cols, rows),
                                                borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=[0, 0, 0])
                    
                    if self.viz_config.view_mode == "overlay":
                        alpha = 0.7
                        combined_frame = cv2.addWeighted(depth_colormap, alpha, color_viz, 1-alpha, 0)
                    else:  # side-by-side
                        combined_frame = np.hstack((color_viz, depth_colormap))
                        
                    # Add alignment info if in alignment mode
                    if self.settings.alignment_mode:
                        cv2.putText(combined_frame, 
                                f"Alignment Mode: v={self.settings.vertical_shift} h={self.settings.horizontal_shift}",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)
                    
                    # Update frame queue
                    try:
                        self.frame_queue.put_nowait(combined_frame)
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(combined_frame)
                        except queue.Empty:
                            pass
                            
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                time.sleep(0.1)
    
    def encode_frame(self):
        """Generator for MJPEG streaming"""
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if frame is not None:
                    ret, encoded_frame = cv2.imencode('.jpg', frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               encoded_frame.tobytes() + b'\r\n')
            except queue.Empty:
                continue
    
    def get_status(self) -> Dict:
        """Get current system status"""
        return {
            "camera_connected": True,
            "model_loaded": True,
            "fps": self.fps,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_classification(self) -> Dict:
        """Get latest classification with detailed probabilities"""
        with torch.no_grad():
            outputs = self.classifier(self.last_rgb_tensor, self.last_depth_tensor) if hasattr(self, 'last_rgb_tensor') else None
            if outputs is not None:
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                all_probs = {stage: float(prob) for stage, prob in zip(self.STAGES, probabilities)}
            else:
                all_probs = {stage: 0.0 for stage in self.STAGES}
                
        return {
            "stage": self.latest_classification["stage"],
            "confidence": self.latest_classification["confidence"],
            "timestamp": self.latest_classification["timestamp"],
            "probabilities": all_probs
        }
    
    def get_stream(self):
        """Get MJPEG stream"""
        return StreamingResponse(
            self.encode_frame(),
            media_type='multipart/x-mixed-replace; boundary=frame'
        )
    
    def get_settings(self) -> Settings:
        """Get current settings"""
        return self.settings
    
    def update_settings(self, new_settings: Settings):
        """Update settings"""
        self.settings = new_settings
        self.viz_config.view_mode = new_settings.view_mode
        self.viz_config.auto_range = new_settings.auto_range
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join()
        if self.camera:
            self.camera.close()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Knot Classifier API Server')
    
    # Camera settings
    parser.add_argument('--camera-index', type=int, default=7,
                      help='Index of the color camera to use (default: 7)')
    
    # Model settings
    parser.add_argument('--model-path', type=str, default='best_model.pth',
                      help='Path to the trained model weights file (default: best_model.pth)')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                      help='Minimum confidence threshold for classifications (default: 0.7)')
    
    # Visualization settings
    parser.add_argument('--view-mode', type=str, default='side-by-side',
                      choices=['side-by-side', 'overlay'],
                      help='Visualization mode (default: side-by-side)')
    parser.add_argument('--auto-range', type=bool, default=True,
                      help='Automatically adjust depth range (default: True)')
    parser.add_argument('--min-depth', type=int, default=100,
                      help='Minimum depth value in mm (default: 100)')
    parser.add_argument('--max-depth', type=int, default=1000,
                      help='Maximum depth value in mm (default: 1000)')
    
    # Image alignment settings
    parser.add_argument('--vertical-shift', type=int, default=47,
                      help='Vertical shift for depth-color alignment (default: 71)')
    parser.add_argument('--horizontal-shift', type=int, default=28,
                      help='Horizontal shift for depth-color alignment (default: 45)')
    parser.add_argument('--alignment-mode', action='store_true',
                      help='Enable alignment adjustment mode')
    
    # Server settings
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host address to bind the server (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8002,
                      help='Port number for the server (default: 8002)')
    
    return parser.parse_args()

# Global instance for API
classifier_api = None

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier_api
    args = parse_args()
    settings = Settings(
        camera_index=args.camera_index,
        model_path=args.model_path,
        confidence_threshold=args.confidence_threshold,
        view_mode=args.view_mode,
        auto_range=args.auto_range,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        host=args.host,
        port=args.port
    )
    classifier_api = KnotClassifierAPI(settings=settings)
    yield
    if classifier_api:
        classifier_api.cleanup()

app = FastAPI(
    title="Knot Classifier API",
    description="API for real-time knot tying classification using RGB-D camera",
    lifespan=lifespan
)

@app.get("/status")
async def get_status():
    """
    Get current system status.
    
    Returns:
        dict: Contains information about:
            - Camera connection status
            - Model loading status
            - Current FPS
            - Current timestamp
    """
    return classifier_api.get_status()

@app.get("/classification")
async def get_classification():
    """
    Get the latest knot classification result.
    
    Returns:
        dict: Classification details including:
            - Predicted stage ("loose", "loop", "complete", "tightened", or "unknown")
            - Confidence score (0.0 to 1.0)
            - Timestamp of classification
            - Individual probability scores for each possible stage
    """
    return classifier_api.get_classification()

@app.get("/stream")
async def get_stream():
    """
    Get live video stream of the classification visualization.
    
    Returns:
        StreamingResponse: MJPEG stream containing:
            - Color camera feed
            - Depth visualization
            - Current classification results
            - FPS counter
            
    Raises:
        500: If stream cannot be initialized or encounters an error
    """
    try:
        return StreamingResponse(
            classifier_api.encode_frame(),
            media_type='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        logger.error(f"Stream error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Stream error occurred"}
        )

@app.post("/alignment")
async def update_alignment(
    vertical_shift: Optional[int] = None,
    horizontal_shift: Optional[int] = None,
    mode: Optional[bool] = None
):
    """
    Update depth-color camera alignment parameters.
    
    Args:
        vertical_shift: Vertical offset in pixels for depth image alignment.
            Positive values move depth image down, negative values move it up.
        horizontal_shift: Horizontal offset in pixels for depth image alignment.
            Positive values move depth image right, negative values move it left.
        mode: Enable/disable alignment adjustment mode. When enabled, shows current
            alignment values in the video stream.
    
    Returns:
        dict: Current alignment settings after update:
            - vertical_shift: Current vertical offset
            - horizontal_shift: Current horizontal offset
            - alignment_mode: Whether alignment mode is enabled
    """
    if vertical_shift is not None:
        classifier_api.settings.vertical_shift = vertical_shift
    if horizontal_shift is not None:
        classifier_api.settings.horizontal_shift = horizontal_shift
    if mode is not None:
        classifier_api.settings.alignment_mode = mode
        
    return {
        "vertical_shift": classifier_api.settings.vertical_shift,
        "horizontal_shift": classifier_api.settings.horizontal_shift,
        "alignment_mode": classifier_api.settings.alignment_mode
    }

@app.get("/settings")
async def get_settings():
    """
    Get current system settings.
    
    Returns:
        Settings: Complete settings object containing:
            - Camera settings (index)
            - Model settings (path, confidence threshold)
            - Visualization settings (view mode, depth range)
            - Server settings (host, port)
            - Alignment settings (shifts, mode)
    """
    return classifier_api.get_settings()

@app.post("/settings")
async def update_settings(settings: Settings):
    """
    Update system settings.
    
    Args:
        settings: Complete Settings object with new values.
            All fields are optional and only provided values will be updated.
    
    Returns:
        dict: Status of update operation
    """
    classifier_api.update_settings(settings)
    return {"status": "success"}

def main():
    """Main entry point with command line arguments"""
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()