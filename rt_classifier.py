import torch
import cv2
import numpy as np
from pathlib import Path
import time
from torchvision import transforms
from imi_wrapper import ImiCamera, StreamType
from imi_visualization import FrameVisualizer, VisualizationConfig, ColorMap
from knot_classifier import DualStreamKnotClassifier

class RealtimeKnotClassifier:
    """Real-time knot classifier using RGB-D camera"""
    
    STAGES = ["loose", "loop", "complete", "tightened"]
    
    def __init__(self, model_path='best_model.pth', color_index=4, confidence_threshold=0.7):
        """Initialize classifier
        
        Args:
            model_path: Path to trained model weights
            color_index: Index of color camera
            confidence_threshold: Minimum confidence for classification
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize camera and visualization
        self.viz_config = VisualizationConfig(
            min_depth=100,
            max_depth=1000,
            auto_range=True,
            colormap=ColorMap.TURBO,
            show_histogram=True,
            show_info=True,
            view_mode="side-by-side",
            window_width=800,
            window_height=600
        )
        self.viz = FrameVisualizer(self.viz_config)
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        
        # Setup transforms
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
        
        # Initialize camera
        self.camera = ImiCamera(color_index=color_index)
        self.camera.initialize()
        
        # Performance tracking
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        
    def _load_model(self, model_path):
        """Load and prepare model for inference"""
        model = DualStreamKnotClassifier(num_classes=len(self.STAGES))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
        
    def _preprocess_frames(self, rgb_frame, depth_frame):
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
        
    def _update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = time.time()
            self.fps = 30 / (current_time - self.last_time)
            self.last_time = current_time
        
    def run(self):
        """Run real-time classification"""
        print("Starting real-time classification...")
        print("Controls:")
        print("  'q': Quit")
        print("  'v': Toggle view mode")
        print("  'r': Toggle auto-range")
        
        try:
            # Open streams
            self.camera.open_stream(StreamType.DEPTH)
            self.camera.open_stream(StreamType.COLOR)
            
            running = True
            while running:
                # Get frames
                depth_frame = self.camera.get_frame(StreamType.DEPTH)
                color_frame = self.camera.get_frame(StreamType.COLOR)
                
                if depth_frame is not None and color_frame is not None:
                    # Update FPS
                    self._update_fps()
                    
                    # Prepare visualization frames
                    depth_viz = depth_frame.data.copy()
                    color_viz = color_frame.data.copy()
                    
                    # Get model prediction
                    with torch.no_grad():
                        rgb_tensor, depth_tensor = self._preprocess_frames(color_frame, depth_frame)
                        outputs = self.model(rgb_tensor, depth_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        confidence = confidence.item()
                        predicted_stage = self.STAGES[predicted.item()]
                    
                    # Add prediction overlay
                    confidence_color = (0, 255, 0) if confidence >= self.confidence_threshold else (0, 165, 255)
                    cv2.putText(color_viz, f"Stage: {predicted_stage}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              confidence_color, 2)
                    cv2.putText(color_viz, f"Confidence: {confidence:.2f}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              confidence_color, 2)
                    cv2.putText(color_viz, f"FPS: {self.fps:.1f}", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              (255, 255, 255), 2)
                    
                    # Show frames
                    key = self.viz.show(depth_viz, color_viz)
                    
                    if key == ord('q'):
                        running = False
                        
        finally:
            if self.camera:
                self.camera.close()
            if self.viz:
                self.viz.close()

def main():
    # Initialize and run classifier
    classifier = RealtimeKnotClassifier(
        model_path='best_model.pth',
        color_index=4,
        confidence_threshold=0.7
    )
    classifier.run()

if __name__ == '__main__':
    main()