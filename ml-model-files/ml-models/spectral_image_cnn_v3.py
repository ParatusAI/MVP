# ml-models/stratified_kfold_improved_cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score
import copy

class SpectralImageDataset(Dataset):
    """Enhanced dataset with better data augmentation for spectral images"""
    
    def __init__(self, metadata, transform=None, is_training=False):
        self.metadata = metadata
        self.is_training = is_training
        
        if transform is None:
            if is_training:
                # Spectral-specific augmentations
                self.transform = transforms.Compose([
                    transforms.RandomRotation(3),  # Slight rotation
                    transforms.ColorJitter(brightness=0.05, contrast=0.05),  # Subtle lighting changes
                    transforms.RandomHorizontalFlip(0.3),  # Less aggressive flipping
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                # Validation/test (no augmentation)
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load image
        image = Image.open(item['filename']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Target values - PROPERLY NORMALIZED using actual data ranges
        targets = torch.FloatTensor([
            (item['plqy'] - 0.108) / (0.920 - 0.108),                    # 0-1 range: (0.108-0.920) -> (0-1)
            (item['emission_peak'] - 500.3) / (523.8 - 500.3),          # 0-1 range: (500.3-523.8) -> (0-1)  
            (item['fwhm'] - 17.2) / (60.0 - 17.2)                       # 0-1 range: (17.2-60.0) -> (0-1)
        ])
        
        return image, targets

class ImprovedSpectralCNN(nn.Module):
    """Your original CNN architecture with small improvements"""
    
    def __init__(self, dropout_rate=0.3):
        super(ImprovedSpectralCNN, self).__init__()
        
        # Keep your proven architecture but add batch normalization
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
        
        # Keep your regression head but improve regularization
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
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Better weight initialization"""
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
        # Extract features
        features = self.features(x)
        
        # Flatten for fully connected layers
        features = features.view(features.size(0), -1)
        
        # Regression output
        regression = self.regressor(features)
        
        return regression

def denormalize_predictions(predictions, targets):
    """Convert normalized predictions back to original scales for evaluation"""
    
    # Clone to avoid modifying originals
    pred_denorm = predictions.copy()
    target_denorm = targets.copy()
    
    # Denormalize PLQY: (0-1) -> (0.108-0.920)
    pred_denorm[:, 0] = pred_denorm[:, 0] * (0.920 - 0.108) + 0.108
    target_denorm[:, 0] = target_denorm[:, 0] * (0.920 - 0.108) + 0.108
    
    # Denormalize emission peak: (0-1) -> (500.3-523.8)
    pred_denorm[:, 1] = pred_denorm[:, 1] * (523.8 - 500.3) + 500.3
    target_denorm[:, 1] = target_denorm[:, 1] * (523.8 - 500.3) + 500.3
    
    # Denormalize FWHM: (0-1) -> (17.2-60.0)
    pred_denorm[:, 2] = pred_denorm[:, 2] * (60.0 - 17.2) + 17.2
    target_denorm[:, 2] = target_denorm[:, 2] * (60.0 - 17.2) + 17.2
    
    return pred_denorm, target_denorm

def train_single_fold(train_meta, val_meta, fold_num, device):
    """Train a single fold and return results + best model state"""
    
    print(f"\n{'='*20} FOLD {fold_num} {'='*20}")
    print(f"📊 Training: {len(train_meta)}, Validation: {len(val_meta)}")
    
    # Create datasets
    train_dataset = SpectralImageDataset(train_meta, is_training=True)
    val_dataset = SpectralImageDataset(val_meta, is_training=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Initialize fresh model for this fold
    model = ImprovedSpectralCNN(dropout_rate=0.3).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Training parameters
    num_epochs = 100
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 20
    best_model_state = None
    
    print(f"🏃 Training fold {fold_num}...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gentle gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate average losses
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping and model saving
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f'  Epoch {epoch:3d}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, Patience={patience_counter}/{patience_limit}')
        
        # Early stopping
        if patience_counter >= patience_limit:
            print(f"  🛑 Early stopping at epoch {epoch}")
            break
    
    # Load best model for evaluation
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Denormalize for proper evaluation
    pred_denorm, target_denorm = denormalize_predictions(all_predictions, all_targets)
    
    # Calculate metrics
    fold_results = {}
    properties = ['PLQY', 'Emission_Peak', 'FWHM']
    
    for i, prop in enumerate(properties):
        mae = mean_absolute_error(target_denorm[:, i], pred_denorm[:, i])
        r2 = r2_score(target_denorm[:, i], pred_denorm[:, i])
        fold_results[prop] = {'mae': mae, 'r2': r2}
    
    # Overall metrics
    overall_mae = np.mean([fold_results[prop]['mae'] for prop in properties])
    overall_r2 = np.mean([fold_results[prop]['r2'] for prop in properties])
    
    fold_results['overall'] = {'mae': overall_mae, 'r2': overall_r2}
    
    print(f"📈 Fold {fold_num} Results:")
    for prop in properties:
        print(f"  {prop:<15} MAE: {fold_results[prop]['mae']:.4f}, R²: {fold_results[prop]['r2']:.4f}")
    print(f"  {'Overall':<15} MAE: {overall_mae:.4f}, R²: {overall_r2:.4f}")
    
    return fold_results, epoch + 1, best_model_state, overall_r2

def train_final_model_on_all_data(metadata, device, optimal_epochs):
    """Train final model on all data for deployment"""
    
    print(f"\n{'='*50}")
    print("🏁 TRAINING FINAL MODEL ON ALL DATA")
    print(f"{'='*50}")
    print(f"📊 Using all {len(metadata)} samples")
    print(f"⏰ Training for {optimal_epochs} epochs")
    
    # Create dataset with all data
    full_dataset = SpectralImageDataset(metadata, is_training=True)
    full_loader = DataLoader(full_dataset, batch_size=16, shuffle=True, num_workers=2)
    
    # Initialize model
    final_model = ImprovedSpectralCNN(dropout_rate=0.3).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(final_model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimal_epochs)
    
    print(f"🏃 Training final model...")
    
    for epoch in range(optimal_epochs):
        final_model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for images, targets in full_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = final_model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gentle gradient clipping
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == optimal_epochs - 1:
            print(f'  Epoch {epoch:3d}: Loss={avg_loss:.4f}')
    
    print(f"✅ Final model training completed!")
    return final_model

def train_stratified_kfold_cnn():
    """Train CNN with stratified 5-fold cross-validation and save final model"""
    
    print("🔀 Training with STRATIFIED 5-FOLD Cross-Validation")
    print("🎯 CsPbBr3 Perovskite Synthesis Optimization")
    print("=" * 70)
    
    # Load metadata
    with open('spectral_images/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"📊 Loaded {len(metadata)} spectral images")
    
    # Extract quality classes for stratification
    quality_classes = [item['quality_class'] for item in metadata]
    quality_distribution = {}
    for qc in quality_classes:
        quality_distribution[qc] = quality_distribution.get(qc, 0) + 1
    
    print(f"📈 Quality distribution:")
    for quality, count in quality_distribution.items():
        print(f"  {quality}: {count} samples ({count/len(metadata)*100:.1f}%)")
    
    # Setup stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🖥️  Training on device: {device}")
    print(f"🔀 5-fold stratified cross-validation")
    print("=" * 70)
    
    # Store results from all folds
    all_fold_results = []
    total_epochs = 0
    best_fold_r2 = 0
    best_fold_model_state = None
    
    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(metadata)), quality_classes), 1):
        # Create train and validation metadata for this fold
        train_meta = [metadata[i] for i in train_idx]
        val_meta = [metadata[i] for i in val_idx]
        
        # Verify stratification worked
        train_quality = [item['quality_class'] for item in train_meta]
        val_quality = [item['quality_class'] for item in val_meta]
        
        print(f"\n🔍 Fold {fold} quality distribution:")
        for quality in quality_distribution.keys():
            train_count = train_quality.count(quality)
            val_count = val_quality.count(quality)
            print(f"  {quality}: Train={train_count}, Val={val_count}")
        
        # Train this fold
        fold_results, epochs_trained, fold_model_state, fold_r2 = train_single_fold(train_meta, val_meta, fold, device)
        all_fold_results.append(fold_results)
        total_epochs += epochs_trained
        
        # Track best fold model
        if fold_r2 > best_fold_r2:
            best_fold_r2 = fold_r2
            best_fold_model_state = fold_model_state
            print(f"  🏆 New best fold! R² = {fold_r2:.4f}")
    
    # Calculate cross-validation statistics
    print(f"\n{'='*70}")
    print("🏆 STRATIFIED 5-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    
    properties = ['PLQY', 'Emission_Peak', 'FWHM', 'overall']
    cv_results = {}
    
    for prop in properties:
        # Collect all fold results for this property
        mae_scores = [fold[prop]['mae'] for fold in all_fold_results]
        r2_scores = [fold[prop]['r2'] for fold in all_fold_results]
        
        # Calculate statistics
        cv_results[prop] = {
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'mae_scores': mae_scores,
            'r2_scores': r2_scores
        }
    
    # Print detailed results
    print(f"{'Property':<15} {'MAE Mean±Std':<20} {'R² Mean±Std':<20} {'R² Range':<20}")
    print("-" * 75)
    
    for prop in properties:
        mae_mean = cv_results[prop]['mae_mean']
        mae_std = cv_results[prop]['mae_std']
        r2_mean = cv_results[prop]['r2_mean']
        r2_std = cv_results[prop]['r2_std']
        r2_min = min(cv_results[prop]['r2_scores'])
        r2_max = max(cv_results[prop]['r2_scores'])
        
        print(f"{prop:<15} {mae_mean:.4f}±{mae_std:.4f}      {r2_mean:.4f}±{r2_std:.4f}      [{r2_min:.4f}, {r2_max:.4f}]")
    
    # Performance assessment
    overall_r2_mean = cv_results['overall']['r2_mean']
    overall_r2_std = cv_results['overall']['r2_std']
    
    print(f"\n📊 CROSS-VALIDATION SUMMARY:")
    print(f"Overall R²: {overall_r2_mean:.4f} ± {overall_r2_std:.4f}")
    print(f"Overall MAE: {cv_results['overall']['mae_mean']:.4f} ± {cv_results['overall']['mae_std']:.4f}")
    print(f"Average epochs per fold: {total_epochs / 5:.1f}")
    
    if overall_r2_mean > 0.8:
        performance = "🌟 EXCELLENT"
    elif overall_r2_mean > 0.6:
        performance = "✅ GOOD"
    elif overall_r2_mean > 0.4:
        performance = "⚠️  FAIR"
    else:
        performance = "❌ NEEDS IMPROVEMENT"
    
    print(f"\n{performance}")
    
    # Model consistency check
    r2_cv = overall_r2_std / overall_r2_mean  # Coefficient of variation
    if r2_cv < 0.1:
        consistency = "🎯 Very consistent across folds"
    elif r2_cv < 0.2:
        consistency = "👍 Reasonably consistent"
    else:
        consistency = "⚠️  High variance between folds"
    
    print(f"{consistency} (CV = {r2_cv:.3f})")
    
    # Compare with single train/test split
    print(f"\n📈 COMPARISON:")
    print(f"Single train/test R²:     0.8526")
    print(f"5-fold CV R²:            {overall_r2_mean:.4f} ± {overall_r2_std:.4f}")
    
    if abs(overall_r2_mean - 0.8526) < 0.05:
        print("✅ Consistent with single split - model is robust!")
    elif overall_r2_mean > 0.8526:
        print("🚀 Even better with CV - excellent generalization!")
    else:
        print("⚠️  Lower than single split - check for overfitting")
    
    # NOW TRAIN FINAL MODEL AND SAVE IT!
    optimal_epochs = int(total_epochs / 5)  # Average epochs from CV
    final_model = train_final_model_on_all_data(metadata, device, optimal_epochs)
    
    # Save the final model for deployment
    model_filename = 'cspbbr3_final_model.pth'
    torch.save(final_model.state_dict(), model_filename)
    print(f"\n💾 FINAL MODEL SAVED: {model_filename}")
    
    # Also save the best fold model as backup
    if best_fold_model_state is not None:
        best_fold_filename = 'cspbbr3_best_fold_model.pth'
        torch.save(best_fold_model_state, best_fold_filename)
        print(f"💾 BEST FOLD MODEL SAVED: {best_fold_filename} (R² = {best_fold_r2:.4f})")
    
    # Save comprehensive results
    final_results = {
        'cv_method': 'stratified_5fold',
        'cv_results': cv_results,
        'quality_distribution': quality_distribution,
        'total_samples': len(metadata),
        'average_epochs_per_fold': total_epochs / 5,
        'device_used': str(device),
        'individual_fold_results': all_fold_results,
        'final_model_file': model_filename,
        'best_fold_model_file': best_fold_filename,
        'best_fold_r2': best_fold_r2,
        'final_model_epochs': optimal_epochs
    }
    
    with open('stratified_kfold_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: stratified_kfold_results.json")
    print(f"🎉 Cross-validation completed!")
    print(f"\n🚀 READY FOR DEPLOYMENT!")
    print(f"   Main model: {model_filename}")
    print(f"   Backup model: {best_fold_filename}")
    print(f"   Expected R²: {overall_r2_mean:.4f}")
    
    return final_results

if __name__ == "__main__":
    # Train with stratified 5-fold cross-validation and save final model
    results = train_stratified_kfold_cnn()
    
    print(f"\n🏁 FINAL RESULTS:")
    print(f"📈 Cross-validated R²: {results['cv_results']['overall']['r2_mean']:.4f} ± {results['cv_results']['overall']['r2_std']:.4f}")
    print(f"📉 Cross-validated MAE: {results['cv_results']['overall']['mae_mean']:.4f} ± {results['cv_results']['overall']['mae_std']:.4f}")
    print(f"💾 Model saved as: {results['final_model_file']}")
    
    if results['cv_results']['overall']['r2_mean'] > 0.7:
        print("🚀 EXCELLENT: Your model shows strong, consistent performance!")
        print("✅ Ready for CsPbBr3 synthesis optimization!")
        print(f"🔗 Use '{results['final_model_file']}' in your FastAPI app!")
    else:
        print("🔧 More work needed for reliable synthesis optimization")