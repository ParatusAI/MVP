import json
import base64
import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import boto3
import os

# Initialize S3 client
s3_client = boto3.client('s3')

# Global model variable for reuse across invocations
model = None

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

def load_model():
    """Load the PyTorch model from S3 (cached globally)"""
    global model
    
    if model is None:
        print("Loading model from S3...")
        device = torch.device('cpu')  # Lambda uses CPU
        model = ImprovedSpectralCNN(dropout_rate=0.3)
        
        # Download model from S3 to /tmp folder
        bucket_name = 'spectraldatacspbbr3'
        model_key = 'models/cspbbr3_final_model.pth'
        local_model_path = '/tmp/cspbbr3_final_model.pth'
        
        # Check if already downloaded in this Lambda instance
        if not os.path.exists(local_model_path):
            print("Downloading model from S3...")
            s3_client.download_file(bucket_name, model_key, local_model_path)
            print("Model downloaded successfully!")
        else:
            print("Using cached model from /tmp")
        
        # Load model weights
        model.load_state_dict(torch.load(local_model_path, map_location=device))
        model.eval()
        print("Model loaded successfully!")
    
    return model

def process_image(image_data):
    """Process base64 image for model input"""
    # Decode base64 image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Apply same preprocessing as training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def make_prediction(image_tensor):
    """Make predictions with the model"""
    model = load_model()
    
    with torch.no_grad():
        # Get normalized predictions from model
        normalized_outputs = model(image_tensor)
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

def lambda_handler(event, context):
    """Main Lambda handler"""
    try:
        print("Lambda function started")
        
        # Parse input event
        if 'body' in event:
            # API Gateway format
            body = json.loads(event['body'])
        else:
            # Direct invocation format
            body = event
        
        # Extract image data
        if 'image' not in body:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing image data'})
            }
        
        image_data = body['image']
        
        # Process image and make prediction
        print("Processing image...")
        image_tensor = process_image(image_data)
        
        print("Making prediction...")
        prediction = make_prediction(image_tensor)
        
        # Assess quality and get recommendations
        quality_assessment = assess_quality(prediction)
        recommendations = get_synthesis_recommendations(prediction)
        
        # Store results in DynamoDB
        store_results(prediction, quality_assessment, recommendations)
        
        # Prepare response
        response = {
            'predictions': prediction,
            'quality_assessment': quality_assessment,
            'synthesis_recommendations': recommendations,
            'timestamp': context.aws_request_id
        }
        
        print("Prediction completed successfully")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response)
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def store_results(prediction, quality_assessment, recommendations):
    """Store results in DynamoDB"""
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('synthesis-parameters')
        
        import time
        timestamp = str(int(time.time()))
        
        item = {
            'timestamp': timestamp,
            'plqy': prediction['plqy'],
            'emission_peak': prediction['emission_peak'],
            'fwhm': prediction['fwhm'],
            'quality_score': quality_assessment['overall_score'],
            'quality_grade': quality_assessment['grade'],
            'temp_adjustment': recommendations['temperature_adjustment'],
            'cs_flow_adjustment': recommendations['cs_flow_adjustment'],
            'pb_flow_adjustment': recommendations['pb_flow_adjustment']
        }
        
        table.put_item(Item=item)
        print("Results stored in DynamoDB")
        
    except Exception as e:
        print(f"DynamoDB error: {str(e)}")
        # Don't fail the whole function if DynamoDB fails