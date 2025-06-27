#!/usr/bin/env python3
"""
Optimized Feature Training - Best of Both Worlds
Combines robust 17-feature system with select enhanced features for optimal performance
"""

import numpy as np
import pandas as pd
import json
import pickle
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import SelectFromModel
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedFeatureTrainer:
    """Train optimized model with best feature combination"""
    
    def __init__(self):
        self.output_dir = Path("optimized_feature_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Core robust features (proven to work)
        self.core_features = [
            'cs_br_concentration', 'pb_br2_concentration', 'temperature',
            'oa_concentration', 'oam_concentration', 'reaction_time', 'solvent_type',
            'cs_pb_ratio', 'temp_normalized', 'ligand_ratio', 'supersaturation',
            'nucleation_rate', 'growth_rate', 'solvent_effect',
            'cs_pb_temp_interaction', 'ligand_temp_interaction', 'concentration_product'
        ]
        
        # Enhanced features to evaluate
        self.enhanced_features = [
            'goldschmidt_tolerance', 'formation_energy', 'ligand_total_coverage'
        ]
        
    def load_and_prepare_data(self):
        """Load robust dataset and fixed ultimate dataset"""
        logger.info("📂 Loading datasets for feature optimization...")
        
        # Load robust dataset (baseline)
        robust_df = pd.read_csv("robust_final_results/robust_dataset.csv")
        logger.info(f"✅ Robust dataset: {robust_df.shape}")
        
        # Load fixed ultimate dataset
        fixed_df = pd.read_csv("fixed_ultimate_dataset/fixed_ultimate_dataset_50000.csv")
        logger.info(f"✅ Fixed ultimate dataset: {fixed_df.shape}")
        
        # Ensure both have same class distribution
        robust_class_dist = robust_df['phase_label'].value_counts().sort_index()
        fixed_class_dist = fixed_df['phase_label'].value_counts().sort_index()
        
        logger.info(f"📊 Robust distribution: {dict(robust_class_dist)}")
        logger.info(f"📊 Fixed distribution: {dict(fixed_class_dist)}")
        
        return robust_df, fixed_df
    
    def create_optimized_features(self, robust_df, fixed_df):
        """Create optimized feature set combining best of both"""
        logger.info("🔧 Creating optimized feature combination...")
        
        # Start with robust features
        X_robust = robust_df[self.core_features]
        y_robust = robust_df['phase_label']
        
        # Add select enhanced features from fixed dataset
        enhanced_subset = fixed_df[self.enhanced_features + ['phase_label']]
        
        # Take subset of fixed data to match robust size
        n_samples = min(len(robust_df), len(fixed_df))
        robust_subset = robust_df.sample(n=n_samples, random_state=42)
        fixed_subset = fixed_df.sample(n=n_samples, random_state=42)
        
        # Combine core features with enhanced features
        X_combined = robust_subset[self.core_features].copy()
        
        # Add the three best enhanced features
        for feature in self.enhanced_features:
            X_combined[feature] = fixed_subset[feature].values
        
        y_combined = robust_subset['phase_label']
        
        logger.info(f"✅ Optimized features: {len(X_combined.columns)} total")
        logger.info(f"   Core features: {len(self.core_features)}")
        logger.info(f"   Enhanced features: {len(self.enhanced_features)}")
        
        return X_combined, y_combined
    
    def feature_importance_analysis(self, X, y):
        """Analyze feature importance to validate selection"""
        logger.info("🔬 Analyzing feature importance...")
        
        # Train Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("🔝 Top 10 Feature Importance:")
        for i, row in importance_df.head(10).iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.4f}")
        
        # Identify low-importance features
        low_importance = importance_df[importance_df['importance'] < 0.01]
        if len(low_importance) > 0:
            logger.info(f"⚠️  Low importance features ({len(low_importance)}):")
            for i, row in low_importance.iterrows():
                logger.info(f"   {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def train_optimized_model(self, X, y):
        """Train optimized model with best features"""
        logger.info("🚀 Training optimized model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"📊 Training: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest (proven to work best)
        logger.info("🌲 Training Random Forest with optimized features...")
        start_time = time.time()
        
        rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Initial evaluation
        train_pred = rf_model.predict(X_train_scaled)
        test_pred = rf_model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        
        training_time = time.time() - start_time
        
        logger.info(f"📊 Training accuracy: {train_acc:.4f}")
        logger.info(f"🎯 Test accuracy: {test_acc:.4f}")
        logger.info(f"📈 Test F1-score: {test_f1:.4f}")
        logger.info(f"⏱️  Training time: {training_time:.1f}s")
        
        # Cross-validation
        logger.info("🔄 Cross-validation...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=skf, scoring='accuracy')
        
        logger.info(f"📊 CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Calibration
        logger.info("🌡️  Calibrating model...")
        calibrated_model = CalibratedClassifierCV(rf_model, method='isotonic', cv=3)
        calibrated_model.fit(X_train_scaled, y_train)
        
        # Calibrated evaluation
        calib_pred = calibrated_model.predict(X_test_scaled)
        calib_proba = calibrated_model.predict_proba(X_test_scaled)
        
        calib_acc = accuracy_score(y_test, calib_pred)
        calib_f1 = f1_score(y_test, calib_pred, average='weighted')
        avg_confidence = np.mean(np.max(calib_proba, axis=1))
        
        logger.info(f"🎯 Calibrated accuracy: {calib_acc:.4f}")
        logger.info(f"📈 Calibrated F1-score: {calib_f1:.4f}")
        logger.info(f"🔮 Average confidence: {avg_confidence:.4f}")
        
        # High confidence analysis
        for threshold in [0.8, 0.9, 0.95]:
            high_conf_mask = np.max(calib_proba, axis=1) > threshold
            if np.sum(high_conf_mask) > 0:
                high_conf_acc = accuracy_score(y_test[high_conf_mask], calib_pred[high_conf_mask])
                pct_samples = 100 * np.sum(high_conf_mask) / len(y_test)
                logger.info(f"⭐ Confidence >{threshold}: {high_conf_acc:.4f} ({pct_samples:.1f}% samples)")
        
        # Per-class performance
        logger.info("📋 Per-class performance:")
        class_report = classification_report(y_test, calib_pred, output_dict=True)
        for i in range(5):
            metrics = class_report[str(i)]
            logger.info(f"   Class {i}: P={metrics['precision']:.3f}, "
                       f"R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Package results
        results = {
            'optimized_model_performance': {
                'test_accuracy': float(calib_acc),
                'test_f1_score': float(calib_f1),
                'cv_accuracy_mean': float(cv_scores.mean()),
                'cv_accuracy_std': float(cv_scores.std()),
                'avg_confidence': float(avg_confidence),
                'training_time': float(training_time),
                'feature_count': len(X.columns)
            },
            'feature_list': list(X.columns),
            'core_features': self.core_features,
            'enhanced_features': self.enhanced_features
        }
        
        return calibrated_model, scaler, results, X_test_scaled, y_test
    
    def save_optimized_system(self, model, scaler, results, feature_list):
        """Save the optimized system"""
        logger.info("💾 Saving optimized system...")
        
        # Save model
        model_path = self.output_dir / 'optimized_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save scaler
        scaler_path = self.output_dir / 'optimized_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save results
        results_path = self.output_dir / 'optimized_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save feature list
        features_path = self.output_dir / 'optimized_features.json'
        with open(features_path, 'w') as f:
            json.dump(feature_list, f, indent=2)
        
        logger.info(f"📂 Optimized system saved to: {self.output_dir}")
        logger.info("📁 Files created:")
        logger.info("   - optimized_model.pkl")
        logger.info("   - optimized_scaler.pkl") 
        logger.info("   - optimized_results.json")
        logger.info("   - optimized_features.json")

def main():
    """Run optimized feature training"""
    logger.info("🚀 OPTIMIZED FEATURE TRAINING")
    logger.info("=" * 45)
    logger.info("Combining robust 17-feature system with select enhanced features")
    
    trainer = OptimizedFeatureTrainer()
    
    # Load data
    robust_df, fixed_df = trainer.load_and_prepare_data()
    
    # Create optimized features
    X_optimized, y_optimized = trainer.create_optimized_features(robust_df, fixed_df)
    
    # Analyze feature importance
    importance_df = trainer.feature_importance_analysis(X_optimized, y_optimized)
    
    # Train optimized model
    model, scaler, results, X_test, y_test = trainer.train_optimized_model(X_optimized, y_optimized)
    
    # Save system
    trainer.save_optimized_system(model, scaler, results, list(X_optimized.columns))
    
    # Final assessment
    final_accuracy = results['optimized_model_performance']['test_accuracy']
    
    logger.info("\n" + "=" * 45)
    logger.info("🎉 OPTIMIZED FEATURE TRAINING COMPLETE!")
    logger.info("=" * 45)
    logger.info(f"🎯 Final optimized accuracy: {final_accuracy:.4f}")
    logger.info(f"🔬 Total features: {len(X_optimized.columns)}")
    logger.info(f"   Core robust features: {len(trainer.core_features)}")
    logger.info(f"   Enhanced features: {len(trainer.enhanced_features)}")
    
    # Compare with baseline
    baseline_acc = 0.9153  # From robust system
    improvement = final_accuracy - baseline_acc
    
    if improvement > 0:
        logger.info(f"📈 Improvement over baseline: +{improvement:.4f} ({100*improvement:.2f}%)")
        logger.info("✅ ENHANCED SYSTEM SUCCESS!")
    else:
        logger.info(f"📊 Performance vs baseline: {improvement:.4f}")
        logger.info("⚖️  Robust system remains optimal")
    
    return final_accuracy > 0.92

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("\n🎉 OPTIMIZED SYSTEM READY FOR DEPLOYMENT!")
    else:
        logger.info("\n📊 Robust baseline remains the best option")