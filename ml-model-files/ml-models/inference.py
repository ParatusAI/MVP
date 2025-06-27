import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import numpy as np
import io
import base64

class ImprovedSpectralCNN(nn.Module):
    """Same CNN architecture from your training file"""
    
    def __init__(self, dropout_rate=0.3):
        super(ImprovedSpectralCNN, self).__init__()
        
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
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        regression = self.regressor(features)
        return regression

def model_fn(model_dir):
    """Load the PyTorch model for SageMaker"""
    device = torch.device('cpu')  # SageMaker CPU inference
    model = ImprovedSpectralCNN(dropout_rate=0.3)
    
    # Load your trained model weights
    model_path = f"{model_dir}/cspbbr3_final_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def input_fn(request_body, content_type):
    """Process input data for the model"""
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        
        # Decode base64 image
        image_data = base64.b64decode(input_data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Apply same preprocessing as training
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor
    
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Make predictions with the model"""
    device = torch.device('cpu')
    
    with torch.no_grad():
        # Get normalized predictions from model
        normalized_outputs = model(input_data)
        normalized_outputs = normalized_outputs.cpu().numpy()[0]  # Remove batch dimension
        
        # Denormalize predictions to original scales
        plqy = normalized_outputs[0] * (0.920 - 0.108) + 0.108
        emission_peak = normalized_outputs[1] * (523.8 - 500.3) + 500.3  
        fwhm = normalized_outputs[2] * (60.0 - 17.2) + 17.2
        
        # Clamp to reasonable ranges
        plqy = max(0.0, min(1.0, plqy))
        emission_peak = max(480.0, min(540.0, emission_peak))
        fwhm = max(10.0, min(80.0, fwhm))
        
        return {
            'plqy': float(plqy),
            'emission_peak': float(emission_peak),
            'fwhm': float(fwhm)
        }

def output_fn(prediction, accept):
    """Format the prediction output"""
    if accept == 'application/json':
        # Add decision logic for synthesis parameter adjustment
        recommendations = get_synthesis_recommendations(prediction)
        
        response = {
            'predictions': prediction,
            'synthesis_recommendations': recommendations,
            'quality_assessment': assess_quality(prediction)
        }
        
        return json.dumps(response), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

def assess_quality(prediction):
    """Assess the quality of predicted CsPbBr3 properties"""
    plqy = prediction['plqy']
    emission_peak = prediction['emission_peak']
    fwhm = prediction['fwhm']
    
    # Target values for high-quality CsPbBr3
    target_plqy = 0.80
    target_emission = 515.0  # Green emission
    target_fwhm = 20.0  # Narrow bandwidth
    
    # Calculate quality score (0-100)
    plqy_score = min(100, (plqy / target_plqy) * 100)
    
    # Emission peak score (penalty for deviation from 515nm)
    emission_deviation = abs(emission_peak - target_emission)
    emission_score = max(0, 100 - (emission_deviation * 5))  # 5% penalty per nm deviation
    
    # FWHM score (lower is better)
    fwhm_score = max(0, 100 - ((fwhm - target_fwhm) * 2))  # 2% penalty per nm above target
    
    overall_quality = (plqy_score + emission_score + fwhm_score) / 3
    
    if overall_quality >= 80:
        quality_grade = "EXCELLENT"
    elif overall_quality >= 60:
        quality_grade = "GOOD"
    elif overall_quality >= 40:
        quality_grade = "FAIR"
    else:
        quality_grade = "POOR"
    
    return {
        'overall_score': round(overall_quality, 1),
        'grade': quality_grade,
        'plqy_score': round(plqy_score, 1),
        'emission_score': round(emission_score, 1),
        'fwhm_score': round(fwhm_score, 1)
    }

def get_synthesis_recommendations(prediction):
    """Provide synthesis parameter adjustment recommendations"""
    plqy = prediction['plqy']
    emission_peak = prediction['emission_peak']
    fwhm = prediction['fwhm']
    
    recommendations = {
        'temperature_adjustment': 0,  # °C change
        'cs_flow_adjustment': 0,      # mL/min change
        'pb_flow_adjustment': 0,      # mL/min change
        'reasoning': []
    }
    
    # PLQY optimization
    if plqy < 0.6:
        recommendations['temperature_adjustment'] += 5  # Increase temperature
        recommendations['reasoning'].append("Low PLQY: Increase temperature for better crystallization")
    elif plqy > 0.9:
        recommendations['temperature_adjustment'] -= 2  # Slightly decrease
        recommendations['reasoning'].append("Very high PLQY: Slight temperature reduction to maintain stability")
    
    # Emission peak optimization (target: 515nm)
    if emission_peak < 510:
        # Blue-shifted, need larger particles
        recommendations['cs_flow_adjustment'] -= 0.1
        recommendations['pb_flow_adjustment'] += 0.1
        recommendations['reasoning'].append("Blue-shifted emission: Increase Pb:Cs ratio for larger particles")
    elif emission_peak > 520:
        # Red-shifted, need smaller particles  
        recommendations['cs_flow_adjustment'] += 0.1
        recommendations['pb_flow_adjustment'] -= 0.1
        recommendations['reasoning'].append("Red-shifted emission: Increase Cs:Pb ratio for smaller particles")
    
    # FWHM optimization (target: narrow bandwidth)
    if fwhm > 25:
        recommendations['temperature_adjustment'] += 3  # Better size distribution
        recommendations['reasoning'].append("Broad FWHM: Increase temperature for better size uniformity")
    
    # Safety limits
    recommendations['temperature_adjustment'] = max(-10, min(15, recommendations['temperature_adjustment']))
    recommendations['cs_flow_adjustment'] = max(-0.3, min(0.3, recommendations['cs_flow_adjustment']))
    recommendations['pb_flow_adjustment'] = max(-0.3, min(0.3, recommendations['pb_flow_adjustment']))
    
    if not recommendations['reasoning']:
        recommendations['reasoning'].append("Properties look good - maintain current parameters")
    
    return recommendations