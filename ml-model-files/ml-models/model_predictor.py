# ml-models/model_predictor.py
import torch
import pandas as pd
import numpy as np
from advanced_cnn_trainer import AdvancedQDCNN
import pickle

class QDPredictor:
    """Production model for quantum dot property prediction"""
    
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load saved model and components
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = AdvancedQDCNN()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.scaler = checkpoint['scaler']
        self.label_encoder = checkpoint['label_encoder']
        
        print(f"Model loaded from {model_path}")
        print(f"Quality classes: {self.label_encoder.classes_}")
    
    def predict(self, cs_flow_rate, pb_flow_rate, temperature, residence_time, 
                return_confidence=True):
        """Predict quantum dot properties"""
        
        # Prepare input
        features = np.array([[cs_flow_rate, pb_flow_rate, temperature, residence_time]])
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features_tensor)
            
            # Regression predictions
            reg_pred = outputs['regression'].cpu().numpy()[0]
            
            # Classification predictions
            clf_logits = outputs['classification'].cpu().numpy()[0]
            clf_probs = torch.softmax(torch.tensor(clf_logits), dim=0).numpy()
            predicted_class_idx = np.argmax(clf_probs)
            predicted_class = self.label_encoder.classes_[predicted_class_idx]
            confidence = clf_probs[predicted_class_idx]
        
        results = {
            'plqy': float(reg_pred[0]),
            'emission_peak': float(reg_pred[1]),
            'fwhm': float(reg_pred[2]),
            'predicted_quality': predicted_class,
            'quality_confidence': float(confidence)
        }
        
        if return_confidence:
            # Add confidence intervals (simple approach using model uncertainty)
            results['confidence_intervals'] = {
                'plqy_range': [float(reg_pred[0] * 0.9), float(reg_pred[0] * 1.1)],
                'peak_range': [float(reg_pred[1] - 3), float(reg_pred[1] + 3)],
                'fwhm_range': [float(reg_pred[2] * 0.8), float(reg_pred[2] * 1.2)]
            }
        
        return results
    
    def predict_batch(self, conditions_df):
        """Predict for multiple conditions"""
        results = []
        
        for _, row in conditions_df.iterrows():
            pred = self.predict(
                row['cs_flow_rate'], 
                row['pb_flow_rate'],
                row['temperature'], 
                row['residence_time']
            )
            pred['input_conditions'] = row.to_dict()
            results.append(pred)
        
        return results
    
    def optimization_suggestions(self, current_conditions, target_plqy=0.8):
        """Suggest parameter changes to improve PLQY"""
        
        base_pred = self.predict(**current_conditions)
        current_plqy = base_pred['plqy']
        
        if current_plqy >= target_plqy:
            return {"message": f"Current PLQY ({current_plqy:.3f}) already meets target ({target_plqy})"}
        
        suggestions = []
        
        # Test parameter variations
        test_conditions = [
            # Increase Cs flow rate
            {**current_conditions, 'cs_flow_rate': current_conditions['cs_flow_rate'] * 1.2},
            # Increase temperature
            {**current_conditions, 'temperature': current_conditions['temperature'] + 20},
            # Increase residence time
            {**current_conditions, 'residence_time': current_conditions['residence_time'] + 30},
            # Combination
            {**current_conditions, 'cs_flow_rate': current_conditions['cs_flow_rate'] * 1.1,
             'temperature': current_conditions['temperature'] + 10}
        ]
        
        for i, test_cond in enumerate(test_conditions):
            pred = self.predict(**test_cond)
            improvement = pred['plqy'] - current_plqy
            
            if improvement > 0.05:  # Significant improvement
                suggestions.append({
                    'conditions': test_cond,
                    'predicted_plqy': pred['plqy'],
                    'improvement': improvement,
                    'predicted_quality': pred['predicted_quality']
                })
        
        # Sort by improvement
        suggestions = sorted(suggestions, key=lambda x: x['improvement'], reverse=True)
        
        return {
            'current_plqy': current_plqy,
            'target_plqy': target_plqy,
            'suggestions': suggestions[:3]  # Top 3 suggestions
        }

# Example usage
if __name__ == "__main__":
    # Load your trained model (update path as needed)
    predictor = QDPredictor("training_results/best_model_20250528_120000.pth")
    
    # Single prediction
    result = predictor.predict(
        cs_flow_rate=1.1,
        pb_flow_rate=1.0,
        temperature=155,
        residence_time=110
    )
    
    print("Prediction Results:")
    print(f"PLQY: {result['plqy']:.3f}")
    print(f"Emission Peak: {result['emission_peak']:.1f} nm")
    print(f"FWHM: {result['fwhm']:.1f} nm")
    print(f"Predicted Quality: {result['predicted_quality']} (confidence: {result['quality_confidence']:.3f})")
    
    # Optimization suggestions
    current_conditions = {
        'cs_flow_rate': 0.8,
        'pb_flow_rate': 1.0,
        'temperature': 140,
        'residence_time': 100
    }
    
    suggestions = predictor.optimization_suggestions(current_conditions, target_plqy=0.8)
    print(f"\nOptimization Suggestions:")
    print(f"Current PLQY: {suggestions['current_plqy']:.3f}")
    
    for i, suggestion in enumerate(suggestions['suggestions']):
        print(f"\nSuggestion {i+1}:")
        print(f"  Predicted PLQY: {suggestion['predicted_plqy']:.3f} (+{suggestion['improvement']:.3f})")
        print(f"  Conditions: {suggestion['conditions']}")