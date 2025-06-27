#!/usr/bin/env python3
"""
Investigate Model Issue - Debug Why "Working" Model Shows 20% Accuracy
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def investigate_model_issue():
    """Investigate why the model is showing poor performance"""
    logger.info("🔍 INVESTIGATING MODEL ISSUE")
    logger.info("=" * 40)
    
    # Load dataset
    df = pd.read_csv("robust_final_results/robust_dataset.csv")
    X = df.drop('phase_label', axis=1)
    y = df['phase_label']
    
    logger.info(f"📊 Dataset shape: {df.shape}")
    logger.info(f"🔬 Features: {list(X.columns)}")
    logger.info(f"📈 Target distribution: {y.value_counts().sort_index().to_dict()}")
    
    # Load model and scaler
    with open("robust_final_results/calibrated_model.pkl", 'rb') as f:
        model = pickle.load(f)
    
    with open("robust_final_results/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    logger.info(f"🤖 Model type: {type(model)}")
    logger.info(f"📐 Scaler type: {type(scaler)}")
    
    # Check if we need to scale the data
    logger.info("\n🧪 TESTING WITH AND WITHOUT SCALING")
    logger.info("-" * 35)
    
    # Test 1: Raw data
    try:
        y_pred_raw = model.predict(X)
        acc_raw = accuracy_score(y, y_pred_raw)
        logger.info(f"📊 Raw data accuracy: {acc_raw:.4f}")
    except Exception as e:
        logger.error(f"❌ Raw data prediction failed: {e}")
    
    # Test 2: Scaled data
    try:
        X_scaled = scaler.transform(X)
        y_pred_scaled = model.predict(X_scaled)
        acc_scaled = accuracy_score(y, y_pred_scaled)
        logger.info(f"📊 Scaled data accuracy: {acc_scaled:.4f}")
        
        # Check predictions distribution
        unique_preds, counts = np.unique(y_pred_scaled, return_counts=True)
        pred_dist = dict(zip(unique_preds, counts))
        logger.info(f"🎯 Prediction distribution: {pred_dist}")
        
    except Exception as e:
        logger.error(f"❌ Scaled data prediction failed: {e}")
    
    # Test 3: Check if model is actually the base model
    logger.info("\n🔍 INVESTIGATING MODEL STRUCTURE")
    logger.info("-" * 30)
    
    if hasattr(model, 'base_estimator'):
        logger.info(f"🏗️  Base estimator: {type(model.base_estimator)}")
        
        # Try predicting with base estimator directly
        try:
            if hasattr(model, 'calibrated_classifiers_'):
                for i, cal_clf in enumerate(model.calibrated_classifiers_):
                    logger.info(f"🔧 Calibrated classifier {i}: {type(cal_clf.base_estimator)}")
        except Exception as e:
            logger.error(f"❌ Error inspecting calibrated classifiers: {e}")
    
    # Test 4: Load and test the base model (non-calibrated)
    logger.info("\n🧪 TESTING BASE MODEL")
    logger.info("-" * 20)
    
    base_model_path = "robust_final_results/best_model.pkl"
    if Path(base_model_path).exists():
        try:
            with open(base_model_path, 'rb') as f:
                base_model = pickle.load(f)
            
            logger.info(f"🎯 Base model type: {type(base_model)}")
            
            # Test base model with scaled data
            y_pred_base = base_model.predict(X_scaled)
            acc_base = accuracy_score(y, y_pred_base)
            logger.info(f"🎯 Base model accuracy: {acc_base:.4f}")
            
            # Check base model predictions
            unique_base, counts_base = np.unique(y_pred_base, return_counts=True)
            base_dist = dict(zip(unique_base, counts_base))
            logger.info(f"🎯 Base prediction distribution: {base_dist}")
            
        except Exception as e:
            logger.error(f"❌ Base model testing failed: {e}")
    
    # Test 5: Check feature scaling impact
    logger.info("\n📐 CHECKING FEATURE SCALING")
    logger.info("-" * 25)
    
    logger.info(f"📊 Original feature ranges:")
    for col in X.columns[:5]:  # First 5 features
        logger.info(f"   {col}: [{X[col].min():.3f}, {X[col].max():.3f}]")
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    logger.info(f"📊 Scaled feature ranges:")
    for col in X_scaled_df.columns[:5]:  # First 5 features
        logger.info(f"   {col}: [{X_scaled_df[col].min():.3f}, {X_scaled_df[col].max():.3f}]")
    
    # Test 6: Check for data leakage in training
    logger.info("\n🔍 CHECKING FOR POTENTIAL ISSUES")
    logger.info("-" * 30)
    
    # Check if there are any infinite or NaN values
    inf_count = np.isinf(X_scaled).sum().sum()
    nan_count = np.isnan(X_scaled).sum().sum()
    logger.info(f"🔢 Infinite values: {inf_count}")
    logger.info(f"🔢 NaN values: {nan_count}")
    
    # Check prediction probabilities
    try:
        y_proba = model.predict_proba(X_scaled)
        logger.info(f"🎲 Probability shape: {y_proba.shape}")
        logger.info(f"🎲 Probability ranges: [{y_proba.min():.3f}, {y_proba.max():.3f}]")
        logger.info(f"🎲 Average max probability: {np.mean(np.max(y_proba, axis=1)):.3f}")
        
        # Check if probabilities are reasonable
        avg_probs_per_class = np.mean(y_proba, axis=0)
        logger.info(f"🎲 Average probabilities per class: {avg_probs_per_class}")
        
    except Exception as e:
        logger.error(f"❌ Probability analysis failed: {e}")

if __name__ == "__main__":
    investigate_model_issue()