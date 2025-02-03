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
from typing import Optional, Dict
from pydantic import BaseModel
from pathlib import Path

from knot_classifier import DualStreamKnotClassifier
from imi_wrapper import ImiCamera, StreamType
from imi_visualization import FrameVisualizer, VisualizationConfig, ColorMap

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseModel):
    """API settings model"""
    view_mode: Optional[str] = "side-by-side"
    auto_range: Optional[bool] = True
    confidence_threshold: Optional[float] = 0.7

class KnotClassifierAPI:
    """API wrapper for knot classifier"""
    
    STAGES = ["loose", "loop", "complete", "tightened"]
    
    def __init__(self, model_path: str = 'best_model.pth', color_index: int = 5):
        """Initialize API wrapper"""
        self.settings = Settings()
        self.latest_frame = None
        self.latest_classification = {
            "stage": "unknown",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        self.fps = 0
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)  # Small queue for latest frames
        
        # Initialize visualization config
        self.viz_config = VisualizationConfig(
            min_depth=100,
            max_depth=1000,
            auto_range=True,
            colormap=ColorMap.TURBO,
            show_histogram=False,  # Disable histogram
            show_info=False,      # Disable info overlay
            view_mode="side-by-side",
            window_width=800,
            window_height=600
        )
        self.viz = FrameVisualizer(self.viz_config)
        
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
            weights_path = Path(model_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"Model weights not found at {model_path}")
                
            self.classifier = DualStreamKnotClassifier()
            logger.info("Loading model weights...")
            model_state = torch.load(model_path, map_location=self.device)
            self.classifier.load_state_dict(model_state)
            self.classifier.to(self.device)
            self.classifier.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Initialize camera
        self.camera = ImiCamera(color_index=color_index)
        self.camera.initialize()
        self.camera.open_stream(StreamType.DEPTH)
        self.camera.open_stream(StreamType.COLOR)
        
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
                    
                    # Normalize depth for visualization
                    depth_min = depth_viz[depth_viz > 0].min() if np.any(depth_viz > 0) else 0
                    depth_max = depth_viz.max()
                    depth_normalized = np.zeros_like(depth_viz, dtype=np.uint8)
                    if depth_max > depth_min:
                        valid_mask = depth_viz > 0
                        depth_normalized[valid_mask] = ((depth_viz[valid_mask] - depth_min) * 255 / 
                                                      (depth_max - depth_min))
                    
                    # Get model prediction
                    with torch.no_grad():
                        rgb_tensor, depth_tensor = self.preprocess_frames(color_frame, depth_frame)
                        # Store tensors for reuse
                        self.last_rgb_tensor = rgb_tensor
                        self.last_depth_tensor = depth_tensor
                        
                        outputs = self.classifier(rgb_tensor, depth_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        confidence = confidence.item()
                        predicted_idx = predicted.item()
                        
                        logger.debug(f"Raw prediction: {predicted_idx}, Confidence: {confidence:.3f}")
                        
                        if confidence >= self.settings.confidence_threshold:
                            predicted_stage = self.STAGES[predicted_idx]
                            logger.info(f"Classified as {predicted_stage} with confidence {confidence:.3f}")
                        else:
                            predicted_stage = "unknown"
                            logger.info(f"Low confidence ({confidence:.3f}) - marked as unknown")
                    
                    # Update latest classification
                    self.latest_classification = {
                        "stage": predicted_stage,
                        "confidence": confidence,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Add visualization overlay
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
                    
                    # Create combined visualization
                    # Convert depth visualization to 3-channel (applying colormap)
                    depth_colormap = cv2.applyColorMap(depth_normalized, self.viz_config.colormap.value)
                    
                    # Ensure same size before stacking
                    if depth_colormap.shape[:2] != color_viz.shape[:2]:
                        depth_colormap = cv2.resize(depth_colormap, 
                                                  (color_viz.shape[1], color_viz.shape[0]))
                    
                    combined_frame = np.hstack((color_viz, depth_colormap))
                    
                    # Update frame queue
                    try:
                        self.frame_queue.put_nowait(combined_frame)
                        if frame_count % 30 == 0:
                            logger.info(f"Queue size: {self.frame_queue.qsize()}")
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()  # Remove old frame
                            self.frame_queue.put_nowait(combined_frame)
                        except queue.Empty:
                            pass
                else:
                    logger.warning("Received None frame from camera")
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                time.sleep(0.1)  # Avoid tight loop on error
    
    def encode_frame(self):
        """Generator for MJPEG streaming"""
        logger.info("Starting frame encoding")
        frame_count = 0
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if frame is not None:
                    frame_count += 1
                    ret, encoded_frame = cv2.imencode('.jpg', frame)
                    if ret:
                        if frame_count % 30 == 0:
                            logger.info(f"Encoded {frame_count} frames")
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               encoded_frame.tobytes() + b'\r\n')
                    else:
                        logger.error("Failed to encode frame")
            except queue.Empty:
                logger.warning("Frame queue empty")
    
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
        if self.viz:
            self.viz.close()

# Global instance for API
classifier_api = None

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global classifier_api
    classifier_api = KnotClassifierAPI()
    yield
    # Shutdown
    if classifier_api:
        classifier_api.cleanup()

# Create FastAPI app
app = FastAPI(
    title="Knot Classifier API",
    lifespan=lifespan
)

@app.get("/status")
async def get_status():
    return classifier_api.get_status()

@app.get("/classification")
async def get_classification():
    return classifier_api.get_classification()

@app.get("/stream")
async def get_stream():
    logger.info("Stream requested")
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

@app.get("/settings")
async def get_settings():
    return classifier_api.get_settings()

@app.post("/settings")
async def update_settings(settings: Settings):
    classifier_api.update_settings(settings)
    return {"status": "success"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    main()