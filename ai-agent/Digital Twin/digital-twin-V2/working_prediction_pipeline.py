#!/usr/bin/env python3
"""
Working Prediction Pipeline - Proper Implementation
Correctly implements the robust system with proper feature scaling
"""

import numpy as np
import pandas as pd
import pickle
import json
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkingPredictionPipeline:
    """Properly implemented prediction pipeline with scaling"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_loaded = False
    
    def load_model(self):
        """Load the trained model and preprocessing components"""
        logger.info("📂 Loading trained model components...")
        
        # Load model (use base model for better performance)
        model_path = "robust_final_results/best_model.pkl"
        if Path(model_path).exists():
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"✅ Model loaded: {type(self.model).__name__}")
        else:
            logger.error(f"❌ Model not found: {model_path}")
            return False
        
        # Load scaler
        scaler_path = "robust_final_results/scaler.pkl"
        if Path(scaler_path).exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"✅ Scaler loaded: {type(self.scaler).__name__}")
        else:
            logger.error(f"❌ Scaler not found: {scaler_path}")
            return False
        
        # Load feature names from results
        results_path = "robust_final_results/robust_training_results.json"
        if Path(results_path).exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            self.feature_names = results.get('feature_columns', [])
            logger.info(f"✅ Features loaded: {len(self.feature_names)} features")
        else:
            logger.warning("⚠️  Feature names not found, using default order")
        
        self.is_loaded = True
        return True
    
    def predict(self, X):
        """Make predictions with proper preprocessing"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Ensure X is DataFrame with correct columns
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        
        # Ensure feature order matches training
        if self.feature_names:
            if not all(col in X.columns for col in self.feature_names):
                missing = [col for col in self.feature_names if col not in X.columns]
                raise ValueError(f"Missing features: {missing}")
            X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def predict_single(self, sample_dict):
        """Predict a single sample"""
        predictions, probabilities = self.predict(sample_dict)
        
        prediction = predictions[0]
        confidence = np.max(probabilities[0])
        class_probabilities = dict(enumerate(probabilities[0]))
        
        return {
            'predicted_class': int(prediction),
            'confidence': float(confidence),
            'class_probabilities': {int(k): float(v) for k, v in class_probabilities.items()}
        }
    
    def validate_system(self):
        """Validate the complete system"""
        logger.info("🔍 VALIDATING COMPLETE WORKING SYSTEM")
        logger.info("=" * 45)
        
        # Load test dataset
        df = pd.read_csv("robust_final_results/robust_dataset.csv")
        X = df.drop('phase_label', axis=1)
        y = df['phase_label']
        
        logger.info(f"📊 Dataset: {len(X):,} samples, {len(X.columns)} features")
        
        # Split for proper validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"🔄 Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Test predictions
        predictions, probabilities = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        confidences = np.max(probabilities, axis=1)
        avg_confidence = np.mean(confidences)
        
        logger.info(f"🎯 Test Accuracy: {accuracy:.4f}")
        logger.info(f"📈 Test F1-Score: {f1:.4f}")
        logger.info(f"🔮 Average Confidence: {avg_confidence:.4f}")
        
        # High confidence accuracy
        for threshold in [0.8, 0.9, 0.95]:
            high_conf_mask = confidences > threshold
            if np.sum(high_conf_mask) > 0:
                high_conf_acc = accuracy_score(y_test[high_conf_mask], predictions[high_conf_mask])
                pct_samples = 100 * np.sum(high_conf_mask) / len(y_test)
                logger.info(f"⭐ Confidence >{threshold}: {high_conf_acc:.4f} "
                           f"({pct_samples:.1f}% of samples)")
        
        # Cross-validation
        logger.info("\n🔄 Cross-Validation Performance:")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, self.scaler.transform(X_train), y_train, cv=skf)
        logger.info(f"📊 CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Per-class performance
        logger.info("\n📋 Per-Class Performance:")
        class_report = classification_report(y_test, predictions, output_dict=True)
        for i in range(5):
            metrics = class_report[str(i)]
            logger.info(f"Class {i}: P={metrics['precision']:.3f}, "
                       f"R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            logger.info("\n🔬 Top 10 Feature Importance:")
            importance_df = pd.DataFrame({
                'feature': self.feature_names or X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for i, row in importance_df.head(10).iterrows():
                logger.info(f"   {row['feature']}: {row['importance']:.4f}")
        
        # System status
        status = "EXCELLENT" if accuracy > 0.93 else "GOOD" if accuracy > 0.90 else "NEEDS_IMPROVEMENT"
        
        validation_results = {
            'system_status': status,
            'test_accuracy': float(accuracy),
            'test_f1_score': float(f1),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'avg_confidence': float(avg_confidence),
            'feature_count': len(self.feature_names or X.columns)
        }
        
        logger.info(f"\n🎉 SYSTEM STATUS: {status}")
        logger.info(f"✅ Working system validated with {accuracy:.1%} accuracy!")
        
        return validation_results

def demonstrate_predictions():
    """Demonstrate working predictions"""
    logger.info("\n🚀 DEMONSTRATING WORKING PREDICTIONS")
    logger.info("=" * 40)
    
    pipeline = WorkingPredictionPipeline()
    if not pipeline.load_model():
        logger.error("❌ Failed to load model")
        return
    
    # Test with sample data
    sample_conditions = [
        {
            'cs_br_concentration': 1.5,
            'pb_br2_concentration': 1.0,
            'temperature': 160.0,
            'oa_concentration': 0.4,
            'oam_concentration': 0.3,
            'reaction_time': 30.0,
            'solvent_type': 1,
            'cs_pb_ratio': 1.5,
            'temp_normalized': 0.47,
            'ligand_ratio': 0.35,
            'supersaturation': 0.8,
            'nucleation_rate': 0.15,
            'growth_rate': 120.0,
            'solvent_effect': 1.1,
            'cs_pb_temp_interaction': 0.7,
            'ligand_temp_interaction': 0.15,
            'concentration_product': 1.5
        },
        {
            'cs_br_concentration': 2.5,
            'pb_br2_concentration': 0.5,
            'temperature': 200.0,
            'oa_concentration': 0.8,
            'oam_concentration': 0.6,
            'reaction_time': 60.0,
            'solvent_type': 2,
            'cs_pb_ratio': 5.0,
            'temp_normalized': 0.71,
            'ligand_ratio': 0.56,
            'supersaturation': 1.2,
            'nucleation_rate': 0.3,
            'growth_rate': 200.0,
            'solvent_effect': 1.2,
            'cs_pb_temp_interaction': 1.4,
            'ligand_temp_interaction': 0.43,
            'concentration_product': 1.25
        }
    ]
    
    synthesis_outcomes = {
        0: "Mixed Phase",
        1: "0D Perovskite", 
        2: "2D Perovskite",
        3: "3D Perovskite",
        4: "Failed Synthesis"
    }
    
    for i, sample in enumerate(sample_conditions):
        logger.info(f"\n📝 Sample {i+1}:")
        logger.info(f"   Temperature: {sample['temperature']}°C")
        logger.info(f"   Cs:Pb ratio: {sample['cs_pb_ratio']:.1f}")
        logger.info(f"   Ligand ratio: {sample['ligand_ratio']:.2f}")
        
        result = pipeline.predict_single(sample)
        predicted_outcome = synthesis_outcomes[result['predicted_class']]
        
        logger.info(f"🎯 Prediction: {predicted_outcome} (Class {result['predicted_class']})")
        logger.info(f"🔮 Confidence: {result['confidence']:.1%}")
        
        # Show top 3 probabilities
        sorted_probs = sorted(result['class_probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        logger.info("📊 Top 3 probabilities:")
        for class_id, prob in sorted_probs:
            outcome_name = synthesis_outcomes[class_id]
            logger.info(f"   {outcome_name}: {prob:.1%}")
    
    return pipeline

if __name__ == "__main__":
    # Load and validate system
    pipeline = WorkingPredictionPipeline()
    
    if pipeline.load_model():
        # Validate the system
        results = pipeline.validate_system()
        
        # Demonstrate predictions
        demonstrate_predictions()
        
        # Save validation results
        with open("working_system_validation.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n💾 Validation results saved to: working_system_validation.json")
        logger.info("🎉 WORKING SYSTEM SUCCESSFULLY VALIDATED!")
    else:
        logger.error("❌ Failed to initialize working system")