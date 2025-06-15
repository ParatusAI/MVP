# model_predictor.py - Real-time spectral image prediction for CsPbBr3 synthesis
import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
from PIL import Image
from torchvision import transforms
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedSpectralCNN(nn.Module):
    """Your exact CNN architecture from spectral_image_cnn_v3.py (line 105)"""
    
    def __init__(self, dropout_rate=0.3):
        super(ImprovedSpectralCNN, self).__init__()
        
        # Your proven architecture
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), 
            nn.Dropout2d(0.1),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Your regression head
        self.regressor = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Linear(256, 3)  # Output: normalized plqy, emission_peak, fwhm
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Your weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        regression = self.regressor(features)
        return regression

class CsPbBr3Predictor:
    """Real-time predictor for CsPbBr3 properties from spectral images"""
    
    def __init__(self, model_path: str = "cspbbr3_final_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.confidence = 0.944  # From your stratified k-fold results
        
        # Your exact normalization ranges from training
        self.normalization_ranges = {
            "plqy": [0.108, 0.920],
            "emission_peak": [500.3, 523.8], 
            "fwhm": [17.2, 60.0]
        }
        
        # Load model
        self._load_model(model_path)
        self._setup_transforms()
        
        logger.info(f"✅ CsPbBr3 Predictor initialized on {self.device}")
        logger.info(f"🎯 Model confidence: {self.confidence:.3f}")
    
    def _load_model(self, model_path: str):
        """Load your trained CNN model"""
        try:
            # Try multiple possible locations
            possible_paths = [
                model_path,
                f"../{model_path}",
                f"../../{model_path}",
                f"ml-models/{model_path}",
                f"../ml-models/{model_path}"
            ]
            
            model_loaded = False
            for path in possible_paths:
                if os.path.exists(path):
                    self.model = ImprovedSpectralCNN(dropout_rate=0.3)
                    self.model.load_state_dict(torch.load(path, map_location=self.device))
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info(f"✅ Model loaded from {path}")
                    model_loaded = True
                    break
            
            if not model_loaded:
                raise FileNotFoundError(f"Could not find model file: {model_path}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms (same as training)"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _denormalize_predictions(self, normalized_pred: np.ndarray) -> Dict[str, float]:
        """Convert normalized model outputs back to real property values"""
        
        # Your exact denormalization from training
        plqy = float(normalized_pred[0] * (0.920 - 0.108) + 0.108)
        emission_peak = float(normalized_pred[1] * (523.8 - 500.3) + 500.3)
        fwhm = float(normalized_pred[2] * (60.0 - 17.2) + 17.2)
        
        # Clamp to reasonable ranges
        plqy = max(0.0, min(1.0, plqy))
        emission_peak = max(400.0, min(800.0, emission_peak))
        fwhm = max(5.0, min(100.0, fwhm))
        
        return {
            "plqy": round(plqy, 4),
            "emission_peak": round(emission_peak, 2),
            "fwhm": round(fwhm, 2)
        }
    
    def predict_from_image_path(self, image_path: str) -> Dict[str, Any]:
        """Predict properties from spectral image file path"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            return self._predict_from_pil_image(image, image_path)
            
        except Exception as e:
            logger.error(f"❌ Prediction failed for {image_path}: {e}")
            return self._error_response(str(e))
    
    def predict_from_image_data(self, image_data: bytes, filename: str = "spectral_image.png") -> Dict[str, Any]:
        """Predict properties from image bytes"""
        try:
            # Convert bytes to PIL Image
            from io import BytesIO
            image = Image.open(BytesIO(image_data)).convert('RGB')
            return self._predict_from_pil_image(image, filename)
            
        except Exception as e:
            logger.error(f"❌ Prediction failed for image data: {e}")
            return self._error_response(str(e))
    
    def _predict_from_pil_image(self, image: Image.Image, source: str) -> Dict[str, Any]:
        """Core prediction logic"""
        try:
            # Preprocess image
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run prediction
            with torch.no_grad():
                normalized_output = self.model(input_tensor)
                normalized_pred = normalized_output.cpu().numpy()[0]
            
            # Denormalize to real values
            properties = self._denormalize_predictions(normalized_pred)
            
            # Create response
            response = {
                "predicted_plqy": properties["plqy"],
                "predicted_emission_peak": properties["emission_peak"],
                "predicted_fwhm": properties["fwhm"],
                "confidence": self.confidence,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model_version": "spectral_cnn_v3",
                "source_image": source,
                "status": "success"
            }
            
            logger.info(f"🔮 Prediction: PLQY={properties['plqy']:.3f}, Peak={properties['emission_peak']:.1f}nm, FWHM={properties['fwhm']:.1f}nm")
            return response
            
        except Exception as e:
            logger.error(f"❌ Core prediction failed: {e}")
            return self._error_response(str(e))
    
    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """Generate error response for Ryan's RL agent"""
        return {
            "predicted_plqy": 0.0,
            "predicted_emission_peak": 515.0,  # Default CsPbBr3 peak
            "predicted_fwhm": 25.0,           # Default FWHM
            "confidence": 0.0,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_version": "spectral_cnn_v3",
            "status": "error",
            "error_message": error_msg
        }
    
    def batch_predict(self, image_paths: list) -> list:
        """Predict properties for multiple images"""
        results = []
        for image_path in image_paths:
            result = self.predict_from_image_path(image_path)
            results.append(result)
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for monitoring"""
        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "device": str(self.device),
            "confidence": self.confidence,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

# Global predictor instance
predictor = None

def initialize_predictor(model_path: str = "cspbbr3_final_model.pth"):
    """Initialize the global predictor instance"""
    global predictor
    predictor = CsPbBr3Predictor(model_path)
    return predictor

def predict_properties(image_path: str) -> str:
    """Main prediction function that returns JSON string for Ryan's RL"""
    global predictor
    
    if predictor is None:
        initialize_predictor()
    
    result = predictor.predict_from_image_path(image_path)
    return json.dumps(result, indent=2)

def predict_properties_from_bytes(image_data: bytes, filename: str = "spectral.png") -> str:
    """Prediction from image bytes"""
    global predictor
    
    if predictor is None:
        initialize_predictor()
    
    result = predictor.predict_from_image_data(image_data, filename)
    return json.dumps(result, indent=2)

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing CsPbBr3 Spectral Image Predictor...")
    
    try:
        # Initialize predictor
        predictor = initialize_predictor()
        
        # Health check
        health = predictor.health_check()
        print(f"Health: {json.dumps(health, indent=2)}")
        
        # Test with a dummy image (for development)
        print("\n🔮 Creating test prediction...")
        
        # Create a dummy spectral image for testing
        dummy_image = Image.new('RGB', (224, 224), color='red')
        dummy_image.save('test_spectral_image.png')
        
        # Test prediction
        result = predict_properties('test_spectral_image.png')
        print(f"\n📊 Prediction Result:")
        print(result)
        
        # Clean up
        os.remove('test_spectral_image.png')
        
        print("\n✅ Predictor is ready for real-time synthesis!")
        print("🔗 Ready to integrate with Ryan's RL agent!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("🔧 Make sure cspbbr3_final_model.pth is available")