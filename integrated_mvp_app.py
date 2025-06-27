#!/usr/bin/env python3
"""
Integrated MVP App - CsPbBr3 Synthesis Optimization
===================================================
Combines Isaiah's CNN (94.4% R²) + Ryan's RL (PPO) + File-based Integration
Single deployable app for autonomous perovskite quantum dot synthesis

Performance Requirements:
- CNN predictions: Every 30 seconds
- RL responses: Within 10 seconds
- Web monitoring interface included
"""

import os
import sys
import time
import csv
import json
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import asyncio

# ML/DL imports
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

# RL imports
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Web interface imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# ISAIAH'S CNN COMPONENTS (94.4% R² ACCURACY)
# =============================================================================

class ImprovedSpectralCNN(nn.Module):
    """Isaiah's proven CNN architecture for spectral image analysis"""
    
    def __init__(self, dropout_rate=0.3):
        super(ImprovedSpectralCNN, self).__init__()
        
        # Exact architecture from spectral_image_cnn_v3.py
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
        
        # Regression head
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
        """Initialize weights properly"""
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

class SpectralPredictor:
    """Isaiah's CNN predictor with proper model loading and preprocessing"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Normalization ranges from training data
        self.norm_ranges = {
            'plqy': (0.108, 0.920),
            'emission_peak': (500.3, 523.8),
            'fwhm': (17.2, 60.0)
        }
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load trained CNN model"""
        try:
            self.model = ImprovedSpectralCNN(dropout_rate=0.3)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ CNN model loaded: {model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load CNN model: {e}")
            return False
    
    def denormalize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Convert normalized predictions back to original scales"""
        denorm = predictions.copy()
        
        # Denormalize PLQY: (0-1) -> (0.108-0.920)
        denorm[:, 0] = denorm[:, 0] * (0.920 - 0.108) + 0.108
        
        # Denormalize emission peak: (0-1) -> (500.3-523.8)
        denorm[:, 1] = denorm[:, 1] * (523.8 - 500.3) + 500.3
        
        # Denormalize FWHM: (0-1) -> (17.2-60.0)
        denorm[:, 2] = denorm[:, 2] * (60.0 - 17.2) + 17.2
        
        return denorm
    
    def predict_from_image(self, image_path: str) -> Dict[str, Any]:
        """Predict material properties from spectral image"""
        if self.model is None:
            return self._create_mock_prediction()
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                normalized_pred = self.model(image_tensor)
                denormalized = self.denormalize_predictions(normalized_pred.cpu().numpy())
            
            # Extract values
            plqy = float(denormalized[0, 0])
            emission_peak = float(denormalized[0, 1])
            fwhm = float(denormalized[0, 2])
            
            return {
                'predicted_plqy': round(plqy, 3),
                'predicted_emission_peak': round(emission_peak, 1),
                'predicted_fwhm': round(fwhm, 1),
                'confidence': 0.944,  # CNN's actual performance
                'model_version': 'spectral_cnn_v3',
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return self._create_mock_prediction()
    
    def _create_mock_prediction(self) -> Dict[str, Any]:
        """Fallback mock prediction when model unavailable"""
        plqy = 0.65 + 0.25 * np.random.random()
        emission_peak = 515 + 10 * (np.random.random() - 0.5)
        fwhm = 20 + 15 * np.random.random()
        
        return {
            'predicted_plqy': round(plqy, 3),
            'predicted_emission_peak': round(emission_peak, 1),
            'predicted_fwhm': round(fwhm, 1),
            'confidence': 0.500,
            'model_version': 'mock_predictor',
            'timestamp': datetime.now().isoformat(),
            'image_path': 'mock'
        }

# =============================================================================
# RYAN'S RL COMPONENTS (PPO OPTIMIZATION)
# =============================================================================

class FlowTempFwhmEnv(gym.Env):
    """Ryan's RL environment for CsPbBr3 synthesis optimization"""
    
    def __init__(self):
        super().__init__()
        
        # Action space: [cs_flow, pb_flow, temperature]
        self.ACT_LOW = np.array([0.10, 0.10, 60.], dtype=np.float32)
        self.ACT_HIGH = np.array([0.35, 0.35, 120.])
        
        # Observation space: [current_PLQY, current_lambda, current_FWHM, current_temp, target_PLQY, target_lambda, target_FWHM]
        self.OBS_LOW = np.array([0., 510., 10., 60., 0., 510., 10.], dtype=np.float32)
        self.OBS_HIGH = np.array([100., 530., 40., 120., 100., 530., 40.])
        
        self.action_space = gym.spaces.Box(self.ACT_LOW, self.ACT_HIGH)
        self.observation_space = gym.spaces.Box(self.OBS_LOW, self.OBS_HIGH)
        
        self.max_steps = 40
        self.state = None
        self.last_action = None
        self.t = 0
    
    @staticmethod
    def _react(action):
        """Simulate chemistry response to synthesis parameters"""
        cs, pb, T = action
        plqy = np.clip(25 + 450*(cs-pb) - 0.4*abs(T-90) + np.random.randn()*2, 0, 100)
        lam = np.clip(540 - 65*(cs+pb) + 0.25*(T-90) + np.random.randn()*0.6, 510, 530)
        fwhm = np.clip(35 - 40*abs(cs-pb) + 0.1*abs(T-90) + np.random.randn()*0.4, 10, 40)
        return float(plqy), float(lam), float(fwhm)
    
    def reset(self, *, seed=None, options=None, tgt_plqy=None, tgt_lambda=None, tgt_fwhm=None):
        super().reset(seed=seed)
        
        # Set targets (or use defaults)
        tgt_plqy = float(self.np_random.uniform(70, 90)) if tgt_plqy is None else tgt_plqy
        tgt_lambda = float(self.np_random.uniform(515, 525)) if tgt_lambda is None else tgt_lambda
        tgt_fwhm = float(self.np_random.uniform(15, 25)) if tgt_fwhm is None else tgt_fwhm
        
        # Initialize current state
        cur_plqy = float(self.np_random.uniform(25, 45))
        cur_lambda = float(self.np_random.uniform(515, 525))
        cur_fwhm = float(self.np_random.uniform(20, 30))
        cur_T = float(self.np_random.uniform(80, 100))
        
        self.state = np.array([cur_plqy, cur_lambda, cur_fwhm, cur_T, tgt_plqy, tgt_lambda, tgt_fwhm], dtype=np.float32)
        self.last_action = None
        self.t = 0
        
        return self.state, {}
    
    def step(self, action):
        self.t += 1
        action = np.clip(action, self.ACT_LOW, self.ACT_HIGH)
        
        # Simulate reaction
        cur_plqy, cur_lam, cur_fwhm = self._react(action)
        cur_T = action[2]
        tgt_plqy, tgt_lambda, tgt_fwhm = self.state[4:]
        
        # Update observation
        obs_next = np.array([cur_plqy, cur_lam, cur_fwhm, cur_T, tgt_plqy, tgt_lambda, tgt_fwhm], dtype=np.float32)
        
        # Calculate reward
        w1, w2, w3, w4 = 1.0, 0.2, 0.2, 0.002
        reward = -(w1*(cur_plqy-tgt_plqy)**2 + w2*(cur_lam - tgt_lambda)**2 + w3*(cur_fwhm-tgt_fwhm)**2 + w4*abs(cur_T-90))
        
        # Stability penalty
        if self.last_action is not None:
            reward -= 0.01*np.linalg.norm(action-self.last_action)
        
        self.state = obs_next
        self.last_action = action
        
        terminated = False
        truncated = self.t >= self.max_steps
        
        return obs_next, reward, terminated, truncated, {}

class RLOptimizer:
    """Ryan's RL optimizer using PPO"""
    
    def __init__(self, model_path: str = None):
        self.env = FlowTempFwhmEnv()
        self.model = None
        self.current_state = None
        
        # Check environment
        check_env(self.env, warn=True)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.train_new_model()
    
    def load_model(self, model_path: str) -> bool:
        """Load trained PPO model"""
        try:
            self.model = PPO.load(model_path, env=self.env)
            logger.info(f"✅ RL model loaded: {model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load RL model: {e}")
            return False
    
    def train_new_model(self):
        """Train new PPO model"""
        logger.info("🏃 Training new PPO model...")
        self.model = PPO("MlpPolicy", self.env, learning_rate=3e-4, n_steps=2048, batch_size=256, verbose=1)
        self.model.learn(50_000)
        
        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/ppo_integrated_mvp.zip"
        self.model.save(model_path)
        logger.info(f"💾 New RL model saved: {model_path}")
    
    def set_targets(self, target_plqy: float, target_lambda: float, target_fwhm: float):
        """Set optimization targets"""
        self.current_state, _ = self.env.reset(tgt_plqy=target_plqy, tgt_lambda=target_lambda, tgt_fwhm=target_fwhm)
        logger.info(f"🎯 RL targets set: PLQY={target_plqy:.1f}%, λ={target_lambda:.1f}nm, FWHM={target_fwhm:.1f}nm")
    
    def get_action_with_reasoning(self, current_plqy: float, current_lambda: float, current_fwhm: float, current_temp: float, previous_params: Dict[str, float] = None) -> Dict[str, Any]:
        """Get RL action with detailed reasoning"""
        if self.model is None:
            return self._mock_action_with_reasoning(current_plqy, current_lambda, current_fwhm, previous_params)
        
        try:
            # Update current state
            if self.current_state is not None:
                self.current_state[:4] = [current_plqy, current_lambda, current_fwhm, current_temp]
            else:
                # Initialize with default targets
                self.set_targets(80.0, 520.0, 20.0)
                self.current_state[:4] = [current_plqy, current_lambda, current_fwhm, current_temp]
            
            # Get action from PPO
            action, _ = self.model.predict(self.current_state, deterministic=False)
            
            # Calculate reward for current state
            targets = self.current_state[4:]
            current = self.current_state[:3]
            reward = -np.sum((current - targets)**2)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(current_plqy, current_lambda, current_fwhm, action, previous_params)
            
            return {
                'cs_flow_rate': float(action[0]),
                'pb_flow_rate': float(action[1]),
                'temperature': float(action[2]),
                'reward': float(reward),
                'action_type': 'ppo_optimized',
                'timestamp': datetime.now().isoformat(),
                'reasoning': reasoning,
                'previous_params': previous_params or {'cs_flow_rate': 0.125, 'pb_flow_rate': 0.125, 'temperature': 85.0}
            }
            
        except Exception as e:
            logger.error(f"❌ RL action failed: {e}")
            return self._mock_action_with_reasoning(current_plqy, current_lambda, current_fwhm, previous_params)
    
    def get_action(self, current_plqy: float, current_lambda: float, current_fwhm: float, current_temp: float) -> Dict[str, Any]:
        """Legacy method for backward compatibility"""
        result = self.get_action_with_reasoning(current_plqy, current_lambda, current_fwhm, current_temp)
        # Remove reasoning for legacy compatibility
        legacy_result = {k: v for k, v in result.items() if k not in ['reasoning', 'previous_params']}
        return legacy_result
    
    def _generate_reasoning(self, plqy: float, emission: float, fwhm: float, action: np.ndarray, previous_params: Dict[str, float] = None) -> str:
        """Generate human-readable reasoning for RL decisions"""
        reasons = []
        
        # Target values
        target_plqy = 80.0
        target_emission = 520.0
        target_fwhm = 20.0
        
        # Analyze PLQY
        if plqy > target_plqy + 5:
            reasons.append(f"PLQY ({plqy:.1f}%) exceeds target ({target_plqy:.1f}%) - maintaining high quality")
        elif plqy < target_plqy - 5:
            reasons.append(f"PLQY ({plqy:.1f}%) below target ({target_plqy:.1f}%) - need to improve quantum yield")
        else:
            reasons.append(f"PLQY ({plqy:.1f}%) near target ({target_plqy:.1f}%) - good performance")
        
        # Analyze emission peak
        if abs(emission - target_emission) > 3:
            if emission < target_emission:
                reasons.append(f"Emission peak ({emission:.1f}nm) blue-shifted from target ({target_emission:.1f}nm)")
            else:
                reasons.append(f"Emission peak ({emission:.1f}nm) red-shifted from target ({target_emission:.1f}nm)")
        else:
            reasons.append(f"Emission peak ({emission:.1f}nm) well-centered at target ({target_emission:.1f}nm)")
        
        # Analyze FWHM
        if fwhm > target_fwhm + 5:
            reasons.append(f"FWHM ({fwhm:.1f}nm) broader than target ({target_fwhm:.1f}nm) - emission too wide")
        elif fwhm < target_fwhm - 2:
            reasons.append(f"FWHM ({fwhm:.1f}nm) very narrow - excellent emission quality")
        else:
            reasons.append(f"FWHM ({fwhm:.1f}nm) close to target ({target_fwhm:.1f}nm)")
        
        # Parameter change reasoning
        if previous_params:
            cs_change = float(action[0]) - previous_params.get('cs_flow_rate', 0.125)
            pb_change = float(action[1]) - previous_params.get('pb_flow_rate', 0.125)
            temp_change = float(action[2]) - previous_params.get('temperature', 85.0)
            
            if abs(cs_change) > 0.01:
                direction = "increase" if cs_change > 0 else "decrease"
                reasons.append(f"Adjusting Cs flow ({direction}) to optimize Cs:Pb stoichiometry")
            
            if abs(pb_change) > 0.01:
                direction = "increase" if pb_change > 0 else "decrease"
                reasons.append(f"Adjusting Pb flow ({direction}) for better nucleation control")
            
            if abs(temp_change) > 2:
                direction = "increase" if temp_change > 0 else "decrease"
                if temp_change > 0:
                    reasons.append(f"Increasing temperature to improve crystallinity and narrow FWHM")
                else:
                    reasons.append(f"Decreasing temperature to prevent over-nucleation")
        
        return " • ".join(reasons)
    
    def _mock_action_with_reasoning(self, plqy: float, emission: float, fwhm: float, previous_params: Dict[str, float] = None) -> Dict[str, Any]:
        """Fallback mock action with reasoning"""
        cs_flow = 0.15 + 0.1 * np.random.random()
        pb_flow = 0.15 + 0.1 * np.random.random()
        temp = 80 + 20 * np.random.random()
        
        reasoning = f"Mock RL decision: PLQY={plqy:.1f}%, Peak={emission:.1f}nm, FWHM={fwhm:.1f}nm • Using random parameter adjustments (model not available)"
        
        return {
            'cs_flow_rate': cs_flow,
            'pb_flow_rate': pb_flow,
            'temperature': temp,
            'reward': -10.0,
            'action_type': 'mock',
            'timestamp': datetime.now().isoformat(),
            'reasoning': reasoning,
            'previous_params': previous_params or {'cs_flow_rate': 0.125, 'pb_flow_rate': 0.125, 'temperature': 85.0}
        }
    
    def _mock_action(self) -> Dict[str, Any]:
        """Fallback mock action"""
        return {
            'cs_flow_rate': 0.15 + 0.1 * np.random.random(),
            'pb_flow_rate': 0.15 + 0.1 * np.random.random(),
            'temperature': 80 + 20 * np.random.random(),
            'reward': -10.0,
            'action_type': 'mock',
            'timestamp': datetime.now().isoformat()
        }

# =============================================================================
# AROYSTON'S DIGITAL TWIN COMPONENTS
# =============================================================================

class DigitalTwinSimulator:
    """Aroyston's digital twin for real-time reaction condition simulation"""
    
    def __init__(self):
        self.current_conditions = {
            'cs_br_concentration': 1.5,
            'pb_br2_concentration': 1.0,
            'temperature': 160.0,
            'oa_concentration': 0.4,
            'oam_concentration': 0.3,
            'reaction_time': 30.0,
            'solvent_type': 1
        }
        
        # Digital twin prediction model (simplified version of working pipeline)
        self.prediction_model = None
        self.scaler = None
        self.feature_names = [
            'cs_br_concentration', 'pb_br2_concentration', 'temperature',
            'oa_concentration', 'oam_concentration', 'reaction_time', 'solvent_type',
            'cs_pb_ratio', 'temp_normalized', 'ligand_ratio', 'supersaturation',
            'nucleation_rate', 'growth_rate', 'solvent_effect',
            'cs_pb_temp_interaction', 'ligand_temp_interaction', 'concentration_product'
        ]
        
        # Synthesis outcome mapping
        self.synthesis_outcomes = {
            0: "Mixed Phase",
            1: "0D Perovskite", 
            2: "2D Perovskite",
            3: "3D Perovskite",
            4: "Failed Synthesis"
        }
        
        # Hardware simulation state
        self.hardware_state = {
            'pump_cs_status': 'running',
            'pump_pb_status': 'running',
            'heater_status': 'active',
            'mixer_rpm': 500,
            'pressure': 1.02,
            'flow_stability': 0.95
        }
        
        # History tracking
        self.condition_history = []
        self.prediction_history = []
        
        logger.info("🔧 Digital Twin Simulator initialized")
    
    def calculate_derived_features(self, conditions):
        """Calculate derived features from base conditions (Aroyston's physics)"""
        cs_conc = conditions['cs_br_concentration']
        pb_conc = conditions['pb_br2_concentration']
        temp = conditions['temperature']
        oa_conc = conditions['oa_concentration']
        oam_conc = conditions['oam_concentration']
        
        # Derived features
        cs_pb_ratio = cs_conc / (pb_conc + 1e-8)
        temp_normalized = (temp - 100) / (220 - 100)
        ligand_ratio = (oa_conc + oam_conc) / (cs_conc + pb_conc + 1e-8)
        
        # Simplified physics calculations (based on Aroyston's work)
        supersaturation = cs_conc * pb_conc * np.exp(-2000 / (8.314 * (temp + 273.15)))
        nucleation_rate = 0.1 + 0.3 * np.random.random()  # Simplified
        growth_rate = 50 + 150 * temp_normalized
        solvent_effect = 1.0 + 0.2 * np.random.normal(0, 0.1)
        
        # Interaction terms
        cs_pb_temp_interaction = cs_pb_ratio * temp_normalized
        ligand_temp_interaction = ligand_ratio * temp_normalized
        concentration_product = cs_conc * pb_conc
        
        derived = {
            'cs_pb_ratio': cs_pb_ratio,
            'temp_normalized': temp_normalized,
            'ligand_ratio': ligand_ratio,
            'supersaturation': supersaturation,
            'nucleation_rate': nucleation_rate,
            'growth_rate': growth_rate,
            'solvent_effect': solvent_effect,
            'cs_pb_temp_interaction': cs_pb_temp_interaction,
            'ligand_temp_interaction': ligand_temp_interaction,
            'concentration_product': concentration_product
        }
        
        return derived
    
    def predict_synthesis_outcome(self, conditions):
        """Predict synthesis outcome using digital twin model"""
        # Combine base and derived features
        derived = self.calculate_derived_features(conditions)
        full_features = {**conditions, **derived}
        
        # Mock prediction (in real implementation, this would use Aroyston's trained model)
        # Simulate realistic predictions based on conditions
        temp = conditions['temperature']
        cs_pb_ratio = derived['cs_pb_ratio']
        ligand_ratio = derived['ligand_ratio']
        
        # Physics-based prediction logic
        if 0.9 <= cs_pb_ratio <= 2.0 and 140 <= temp <= 180 and ligand_ratio < 0.5:
            # Favorable for 3D perovskite
            predicted_class = 3 if np.random.random() > 0.2 else np.random.choice([0, 2])
            confidence = 0.85 + 0.1 * np.random.random()
        elif cs_pb_ratio > 2.5 or ligand_ratio > 0.6:
            # Favors 2D or 0D
            predicted_class = np.random.choice([1, 2], p=[0.3, 0.7])
            confidence = 0.75 + 0.15 * np.random.random()
        elif temp < 130 or temp > 200:
            # Poor conditions
            predicted_class = np.random.choice([0, 4], p=[0.4, 0.6])
            confidence = 0.65 + 0.2 * np.random.random()
        else:
            # Mixed conditions
            predicted_class = np.random.choice([0, 1, 2, 3], p=[0.3, 0.2, 0.25, 0.25])
            confidence = 0.70 + 0.2 * np.random.random()
        
        # Generate class probabilities
        probs = np.random.dirichlet([1] * 5)
        probs[predicted_class] = max(probs[predicted_class], confidence)
        probs = probs / probs.sum()
        
        prediction = {
            'predicted_class': int(predicted_class),
            'predicted_outcome': self.synthesis_outcomes[predicted_class],
            'confidence': float(confidence),
            'class_probabilities': {i: float(probs[i]) for i in range(5)},
            'conditions': conditions.copy(),
            'derived_features': derived,
            'timestamp': datetime.now().isoformat()
        }
        
        return prediction
    
    def simulate_hardware_response(self, new_parameters):
        """Simulate hardware response to RL parameter changes"""
        # Simulate gradual parameter changes (hardware can't change instantly)
        cs_target = new_parameters['cs_flow_rate'] * 2.0  # Convert flow to concentration
        pb_target = new_parameters['pb_flow_rate'] * 1.8
        temp_target = new_parameters['temperature']
        
        # Current vs target
        cs_current = self.current_conditions['cs_br_concentration']
        pb_current = self.current_conditions['pb_br2_concentration']
        temp_current = self.current_conditions['temperature']
        
        # Simulate realistic hardware response rates
        cs_rate = 0.1  # 10% change per time step
        pb_rate = 0.15  # 15% change per time step
        temp_rate = 0.05  # 5% change per time step (slower thermal response)
        
        # Update conditions gradually
        self.current_conditions['cs_br_concentration'] += cs_rate * (cs_target - cs_current)
        self.current_conditions['pb_br2_concentration'] += pb_rate * (pb_target - pb_current)
        self.current_conditions['temperature'] += temp_rate * (temp_target - temp_current)
        
        # Add realistic noise
        self.current_conditions['cs_br_concentration'] += 0.02 * np.random.normal()
        self.current_conditions['pb_br2_concentration'] += 0.015 * np.random.normal()
        self.current_conditions['temperature'] += 0.5 * np.random.normal()
        
        # Update hardware state
        self.hardware_state['flow_stability'] = 0.90 + 0.1 * np.random.random()
        self.hardware_state['mixer_rpm'] = 480 + 40 * np.random.random()
        self.hardware_state['pressure'] = 1.0 + 0.05 * np.random.normal()
        
        # Simulate occasional hardware issues
        if np.random.random() < 0.05:  # 5% chance
            self.hardware_state['pump_cs_status'] = 'warning'
            logger.warning("🔧 Digital Twin: Cs pump showing warning status")
        else:
            self.hardware_state['pump_cs_status'] = 'running'
        
        return self.current_conditions.copy()
    
    def validate_rl_recommendations(self, rl_action, current_prediction):
        """Validate RL recommendations before hardware deployment"""
        # Simulate the proposed parameters
        test_conditions = self.current_conditions.copy()
        test_conditions['cs_br_concentration'] = rl_action['cs_flow_rate'] * 2.0
        test_conditions['pb_br2_concentration'] = rl_action['pb_flow_rate'] * 1.8
        test_conditions['temperature'] = rl_action['temperature']
        
        # Predict outcome with new parameters
        predicted_outcome = self.predict_synthesis_outcome(test_conditions)
        
        # Safety checks
        safety_checks = {
            'temperature_safe': 40 <= test_conditions['temperature'] <= 250,
            'concentration_safe': test_conditions['cs_br_concentration'] < 3.0 and test_conditions['pb_br2_concentration'] < 2.5,
            'ratio_reasonable': 0.3 <= (test_conditions['cs_br_concentration'] / test_conditions['pb_br2_concentration']) <= 5.0,
            'predicted_not_failed': predicted_outcome['predicted_class'] != 4
        }
        
        validation_result = {
            'is_safe': all(safety_checks.values()),
            'safety_checks': safety_checks,
            'predicted_outcome': predicted_outcome,
            'recommendation': 'approve' if all(safety_checks.values()) else 'reject',
            'confidence_score': predicted_outcome['confidence'],
            'expected_improvement': predicted_outcome['confidence'] > 0.8 and predicted_outcome['predicted_class'] == 3,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Add validation reasoning
        if validation_result['is_safe']:
            if predicted_outcome['predicted_class'] == 3:
                validation_result['reasoning'] = f"RL parameters validated: High probability ({predicted_outcome['confidence']:.1%}) of 3D perovskite formation"
            else:
                validation_result['reasoning'] = f"RL parameters safe but may produce {predicted_outcome['predicted_outcome']} (confidence: {predicted_outcome['confidence']:.1%})"
        else:
            failed_checks = [check for check, passed in safety_checks.items() if not passed]
            validation_result['reasoning'] = f"RL parameters rejected: Safety violations in {', '.join(failed_checks)}"
        
        return validation_result
    
    def get_current_status(self):
        """Get current digital twin status"""
        global simulation_active
        
        # Only generate new predictions when simulation is active
        if simulation_active:
            current_prediction = self.predict_synthesis_outcome(self.current_conditions)
        else:
            # Use the last prediction from history if available
            current_prediction = self.prediction_history[-1] if self.prediction_history else self.predict_synthesis_outcome(self.current_conditions)
        
        status = {
            'current_conditions': self.current_conditions.copy(),
            'hardware_state': self.hardware_state.copy(),
            'current_prediction': current_prediction,
            'condition_history': self.condition_history[-10:],  # Last 10 entries
            'prediction_history': self.prediction_history[-10:],
            'system_status': 'operational' if simulation_active else 'idle',
            'last_update': datetime.now().isoformat()
        }
        
        return status
    
    def update_from_rl_action(self, rl_action, validation_result=None):
        """Update digital twin based on RL action"""
        if validation_result and validation_result['recommendation'] == 'approve':
            # Apply the RL recommendations through hardware simulation
            new_conditions = self.simulate_hardware_response(rl_action)
            
            # Record history
            self.condition_history.append({
                'timestamp': datetime.now().isoformat(),
                'conditions': new_conditions.copy(),
                'rl_action': rl_action,
                'validation': validation_result
            })
            
            # Generate new prediction
            new_prediction = self.predict_synthesis_outcome(new_conditions)
            self.prediction_history.append(new_prediction)
            
            # Keep history manageable
            if len(self.condition_history) > 50:
                self.condition_history = self.condition_history[-50:]
            if len(self.prediction_history) > 50:
                self.prediction_history = self.prediction_history[-50:]
            
            logger.info(f"🔧 Digital Twin updated: {new_prediction['predicted_outcome']} "
                       f"(confidence: {new_prediction['confidence']:.1%})")
            
            return new_prediction
        else:
            logger.warning("🔧 Digital Twin: RL action rejected, maintaining current conditions")
            return None

# =============================================================================
# FILE-BASED INTEGRATION SYSTEM
# =============================================================================

class FileBasedIntegration:
    """Manages file-based communication between CNN and RL components"""
    
    def __init__(self, predictions_folder: str = "real_time_predictions"):
        self.predictions_folder = predictions_folder
        self.spectral_folder = "spectral_images_realtime"
        self.is_monitoring = False
        self.processed_files = set()
        
        # Components
        self.predictor = None
        self.optimizer = None
        self.digital_twin = None  # Aroyston's digital twin
        self.synthesis_history = []
        
        # Current synthesis parameters for tracking changes
        self.current_parameters = {
            'cs_flow_rate': 0.125,  # Default starting values
            'pb_flow_rate': 0.125,
            'temperature': 85.0
        }
        
        # Target parameters (commercial goals)
        self.target_parameters = {
            'plqy': 80.0,
            'emission_peak': 520.0,
            'fwhm': 20.0
        }
        
        # Create folders
        os.makedirs(self.predictions_folder, exist_ok=True)
        os.makedirs(self.spectral_folder, exist_ok=True)
        
        logger.info(f"🔧 Integration system initialized")
        logger.info(f"📁 Predictions: {self.predictions_folder}")
        logger.info(f"📁 Spectral: {self.spectral_folder}")
    
    def initialize_components(self):
        """Initialize CNN and RL components"""
        logger.info("🔧 Initializing AI components...")
        
        # Initialize CNN predictor
        model_paths = [
            "ml-model-files/ml-models/cspbbr3_final_model.pth",
            "ml-model-files/ml-models/cspbbr3_best_fold_model.pth",
            "cspbbr3_final_model.pth",
            "cspbbr3_best_fold_model.pth"
        ]
        
        self.predictor = SpectralPredictor()
        for path in model_paths:
            if os.path.exists(path):
                if self.predictor.load_model(path):
                    break
        
        # Initialize RL optimizer
        rl_model_paths = [
            "models/ppo_integrated_mvp.zip",
            "ai-agent/RL-agent/models/ppo_flow_temp_fwhm.zip"
        ]
        
        rl_model_path = None
        for path in rl_model_paths:
            if os.path.exists(path):
                rl_model_path = path
                break
        
        self.optimizer = RLOptimizer(rl_model_path)
        
        # Set default targets (commercial goals)
        self.optimizer.set_targets(target_plqy=80.0, target_lambda=520.0, target_fwhm=20.0)
        
        # Initialize digital twin
        self.digital_twin = DigitalTwinSimulator()
        
        logger.info("✅ AI components initialized (CNN + RL + Digital Twin)")
    
    def start_monitoring(self):
        """Start monitoring for new spectral images and predictions"""
        if self.is_monitoring:
            logger.warning("⚠️ Already monitoring!")
            return
        
        if not self.predictor or not self.optimizer:
            self.initialize_components()
        
        self.is_monitoring = True
        
        # Start monitoring threads
        spectral_thread = threading.Thread(target=self._monitor_spectral_images, daemon=True)
        prediction_thread = threading.Thread(target=self._monitor_predictions, daemon=True)
        
        spectral_thread.start()
        prediction_thread.start()
        
        logger.info("🔍 File monitoring started")
        logger.info(f"⏱️ CNN predictions: Every 30s")
        logger.info(f"⏱️ RL responses: Within 10s")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        logger.info("🛑 File monitoring stopped")
    
    def _monitor_spectral_images(self):
        """Monitor for new spectral images (CNN processing)"""
        logger.info("👀 Monitoring spectral images...")
        processed_images = set()
        
        while self.is_monitoring:
            try:
                # Check for new PNG files
                if os.path.exists(self.spectral_folder):
                    png_files = {f for f in os.listdir(self.spectral_folder) if f.endswith('.png')}
                    new_images = png_files - processed_images
                    
                    for image_file in sorted(new_images):
                        if not self.is_monitoring:
                            break
                        
                        image_path = os.path.join(self.spectral_folder, image_file)
                        self._process_spectral_image(image_path)
                        processed_images.add(image_file)
                
                time.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"❌ Error monitoring spectral images: {e}")
                time.sleep(5)
    
    def _monitor_predictions(self):
        """Monitor for new prediction files (RL processing)"""
        logger.info("👀 Monitoring prediction files...")
        
        while self.is_monitoring:
            try:
                # Check for new CSV files
                if os.path.exists(self.predictions_folder):
                    csv_files = {f for f in os.listdir(self.predictions_folder) if f.endswith('.csv')}
                    new_predictions = csv_files - self.processed_files
                    
                    for csv_file in sorted(new_predictions):
                        if not self.is_monitoring:
                            break
                        
                        csv_path = os.path.join(self.predictions_folder, csv_file)
                        self._process_prediction_file(csv_path)
                        self.processed_files.add(csv_file)
                
                time.sleep(1)  # Check every 1 second for fast RL response
                
            except Exception as e:
                logger.error(f"❌ Error monitoring predictions: {e}")
                time.sleep(2)
    
    def _process_spectral_image(self, image_path: str):
        """Process new spectral image with CNN (30s target)"""
        start_time = time.time()
        
        try:
            logger.info(f"📸 Processing spectral image: {os.path.basename(image_path)}")
            
            # CNN prediction
            prediction = self.predictor.predict_from_image(image_path)
            
            # Save prediction to CSV file for RL
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"prediction_{timestamp}.csv"
            csv_path = os.path.join(self.predictions_folder, csv_filename)
            
            # Write CSV with key-value pairs
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['property', 'value'])  # Header
                writer.writerow(['predicted_plqy', prediction['predicted_plqy']])
                writer.writerow(['predicted_emission_peak', prediction['predicted_emission_peak']])
                writer.writerow(['predicted_fwhm', prediction['predicted_fwhm']])
                writer.writerow(['confidence', prediction['confidence']])
                writer.writerow(['model_version', prediction['model_version']])
                writer.writerow(['timestamp', prediction['timestamp']])
                writer.writerow(['source_image', os.path.basename(image_path)])
            
            processing_time = time.time() - start_time
            
            logger.info(f"🔮 CNN Prediction Complete ({processing_time:.2f}s):")
            logger.info(f"   PLQY: {prediction['predicted_plqy']:.3f} ({prediction['predicted_plqy']*100:.1f}%)")
            logger.info(f"   Peak: {prediction['predicted_emission_peak']:.1f} nm")
            logger.info(f"   FWHM: {prediction['predicted_fwhm']:.1f} nm")
            logger.info(f"💾 Saved: {csv_filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {image_path}: {e}")
    
    def _process_prediction_file(self, csv_path: str):
        """Process new prediction file with RL (10s target)"""
        start_time = time.time()
        
        try:
            # Read prediction CSV
            prediction_data = {}
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        key, value = row[0].strip(), row[1].strip()
                        if key in ['predicted_plqy', 'predicted_emission_peak', 'predicted_fwhm', 'confidence']:
                            try:
                                prediction_data[key] = float(value)
                            except ValueError:
                                prediction_data[key] = value
                        else:
                            prediction_data[key] = value
            
            if not prediction_data:
                return
            
            logger.info(f"📥 RL Processing: {os.path.basename(csv_path)}")
            
            # Get RL action with reasoning
            current_temp = self.current_parameters.get('temperature', 85.0)
            
            action = self.optimizer.get_action_with_reasoning(
                current_plqy=prediction_data.get('predicted_plqy', 70.0),
                current_lambda=prediction_data.get('predicted_emission_peak', 520.0),
                current_fwhm=prediction_data.get('predicted_fwhm', 25.0),
                current_temp=current_temp,
                previous_params=self.current_parameters.copy()
            )
            
            # Digital Twin Validation (Aroyston's component)
            digital_twin_validation = self.digital_twin.validate_rl_recommendations(action, prediction_data)
            logger.info(f"🔧 Digital Twin Validation: {digital_twin_validation['recommendation'].upper()}")
            logger.info(f"   Reasoning: {digital_twin_validation['reasoning']}")
            
            # Update digital twin if RL action is approved
            digital_twin_prediction = None
            if digital_twin_validation['recommendation'] == 'approve':
                digital_twin_prediction = self.digital_twin.update_from_rl_action(action, digital_twin_validation)
            else:
                logger.warning(f"⚠️ RL action blocked by Digital Twin safety validation")
            
            # Update current parameters
            self.current_parameters.update({
                'cs_flow_rate': action['cs_flow_rate'],
                'pb_flow_rate': action['pb_flow_rate'],
                'temperature': action['temperature']
            })
            
            # Create synthesis record
            synthesis_record = {
                'timestamp': datetime.now().isoformat(),
                'prediction_file': os.path.basename(csv_path),
                'source_image': prediction_data.get('source_image', 'unknown'),
                'predictions': prediction_data,
                'rl_action': action,
                'digital_twin_validation': digital_twin_validation,
                'digital_twin_prediction': digital_twin_prediction,
                'processing_time_seconds': time.time() - start_time
            }
            
            # Store in history
            self.synthesis_history.append(synthesis_record)
            
            # Keep only last 10 records for performance
            if len(self.synthesis_history) > 10:
                self.synthesis_history = self.synthesis_history[-10:]
            
            # Log RL decision
            processing_time = time.time() - start_time
            logger.info(f"🤖 RL Decision Complete ({processing_time:.2f}s):")
            logger.info(f"   Cs Flow: {action['cs_flow_rate']:.3f} mL/min")
            logger.info(f"   Pb Flow: {action['pb_flow_rate']:.3f} mL/min")
            logger.info(f"   Temperature: {action['temperature']:.1f} °C")
            logger.info(f"   Reward: {action['reward']:.2f}")
            
            # Save synthesis record
            record_filename = f"synthesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            record_path = os.path.join(self.predictions_folder, record_filename)
            with open(record_path, 'w') as f:
                json.dump(synthesis_record, f, indent=2)
            
        except Exception as e:
            logger.error(f"❌ Failed to process prediction {csv_path}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        # Get digital twin status
        digital_twin_status = self.digital_twin.get_current_status() if self.digital_twin else None
        
        return {
            'monitoring': self.is_monitoring,
            'cnn_model_loaded': self.predictor is not None and self.predictor.model is not None,
            'rl_model_loaded': self.optimizer is not None and self.optimizer.model is not None,
            'digital_twin_loaded': self.digital_twin is not None,
            'synthesis_cycles': len(self.synthesis_history),
            'last_synthesis': self.synthesis_history[-1] if self.synthesis_history else None,
            'predictions_processed': len(self.processed_files),
            'current_parameters': self.current_parameters,
            'target_parameters': self.target_parameters,
            'digital_twin_status': digital_twin_status,
            'parameter_history': [
                {
                    'timestamp': record['timestamp'],
                    'current_params': record['rl_action'].get('previous_params', {}),
                    'new_params': {
                        'cs_flow_rate': record['rl_action']['cs_flow_rate'],
                        'pb_flow_rate': record['rl_action']['pb_flow_rate'],
                        'temperature': record['rl_action']['temperature']
                    },
                    'reasoning': record['rl_action'].get('reasoning', 'No reasoning available'),
                    'cnn_analysis': {
                        'plqy': record['predictions']['predicted_plqy'],
                        'emission_peak': record['predictions']['predicted_emission_peak'],
                        'fwhm': record['predictions']['predicted_fwhm']
                    }
                } for record in self.synthesis_history[-10:]  # Last 10 decisions
            ],
            'folders': {
                'predictions': self.predictions_folder,
                'spectral': self.spectral_folder
            }
        }

# =============================================================================
# DEMO SIMULATOR FOR TESTING
# =============================================================================

class DemoSimulator:
    """Demo simulator for testing the integrated system"""
    
    def __init__(self, integration_system: FileBasedIntegration):
        self.integration = integration_system
        self.demo_images = [
            f"demo_spectrum_{i}.png" for i in range(1, 7)
        ]
        self.is_running = False
        self.current_image_index = 0
    
    def create_demo_images(self):
        """Create demo spectral images"""
        for i, filename in enumerate(self.demo_images):
            self._create_demo_spectral_image(filename, i + 1)
        logger.info(f"✅ Created {len(self.demo_images)} demo spectral images")
    
    def _create_demo_spectral_image(self, filename: str, image_num: int):
        """Create realistic demo spectral image"""
        width, height = 640, 480
        img = Image.new('RGB', (width, height), color='black')
        
        # Simulate CsPbBr3 emission around 520nm with variations
        base_wavelength = 520
        peak_variations = [0, -2, 1, -1, 2, 0]
        intensity_variations = [1.0, 0.95, 1.1, 0.9, 1.05, 0.98]
        
        peak_wl = base_wavelength + peak_variations[image_num - 1]
        intensity_factor = intensity_variations[image_num - 1]
        
        # Create spectral data
        wavelengths = np.linspace(400, 700, width)
        sigma = 15
        spectrum = intensity_factor * np.exp(-((wavelengths - peak_wl) / sigma) ** 2)
        spectrum += 0.05 * np.random.random(len(wavelengths))
        spectrum = np.clip(spectrum, 0, 1)
        
        # Convert to image
        pixels = img.load()
        for x in range(width):
            intensity = spectrum[x]
            green_value = int(255 * intensity)
            for y in range(int(height * (1 - intensity)), height):
                if y < height:
                    pixels[x, y] = (0, green_value, 0)
        
        img.save(filename)
    
    def start_demo(self, interval_seconds: int = 30):
        """Start demo simulation"""
        global simulation_active
        self.is_running = True
        simulation_active = True
        self.current_image_index = 0
        
        def demo_loop():
            global simulation_active
            for i, demo_image in enumerate(self.demo_images):
                if not self.is_running:
                    break
                
                self.current_image_index = i
                
                # Copy demo image to spectral folder
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_filename = f"spectrum_{i+1:03d}_{timestamp}.png"
                target_path = os.path.join(self.integration.spectral_folder, target_filename)
                
                if os.path.exists(demo_image):
                    import shutil
                    shutil.copy2(demo_image, target_path)
                    logger.info(f"📸 Demo: Released {target_filename}")
                
                if i < len(self.demo_images) - 1:
                    # Interruptible sleep - check every second if we should stop
                    for sleep_count in range(interval_seconds):
                        if not self.is_running:
                            break
                        time.sleep(1)
            
            logger.info("✅ Demo simulation completed")
            self.is_running = False
            simulation_active = False
        
        # Create demo images if they don't exist
        self.create_demo_images()
        
        # Start demo in background
        demo_thread = threading.Thread(target=demo_loop, daemon=True)
        demo_thread.start()
        
        logger.info(f"🚀 Demo started - releasing images every {interval_seconds}s")
        return demo_thread
    
    def stop_demo(self):
        """Stop demo simulation"""
        global simulation_active
        self.is_running = False
        simulation_active = False
        logger.info("🛑 Demo stopped")

# =============================================================================
# WEB MONITORING INTERFACE
# =============================================================================

# Global integration system
integration_system = None
demo_simulator = None
simulation_active = False

# FastAPI app
app = FastAPI(title="CsPbBr3 Synthesis MVP", description="Integrated ML + RL optimization")

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global integration_system, demo_simulator
    integration_system = FileBasedIntegration()
    integration_system.initialize_components()
    integration_system.start_monitoring()
    demo_simulator = DemoSimulator(integration_system)
    logger.info("🚀 MVP system started")

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    global integration_system
    if integration_system:
        integration_system.stop_monitoring()
    logger.info("🛑 MVP system shutdown")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main monitoring dashboard"""
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>CsPbBr3 Synthesis MVP - Live Monitor</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            /* Modern YouTube-inspired design */
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
                background: linear-gradient(135deg, #0f0f23 0%, #1a1a3a 100%);
                color: #ffffff;
                min-height: 100vh;
                line-height: 1.6;
            }
            
            /* Header */
            .header { 
                background: linear-gradient(135deg, #1e1e3f 0%, #3b3b7d 100%);
                padding: 30px 40px;
                margin-bottom: 30px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                border-bottom: 2px solid #ff6b6b;
                position: relative;
                overflow: hidden;
            }
            .header::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 20"><defs><radialGradient id="a"><stop offset="20%" stop-color="%23FF6B6B" stop-opacity="0.1"/><stop offset="100%" stop-color="%23ffffff" stop-opacity="0"/></radialGradient></defs><rect width="100" height="20" fill="url(%23a)"/></svg>');
            }
            .header h1 { 
                font-size: 2.5rem; 
                font-weight: 700;
                background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 10px;
                position: relative;
                z-index: 1;
            }
            .header p { 
                font-size: 1.1rem; 
                opacity: 0.9; 
                position: relative;
                z-index: 1;
            }
            
            /* Control Panel */
            .control-panel {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                padding: 25px;
                margin: 0 40px 30px 40px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }
            
            .start-simulation-btn {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 50px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
                position: relative;
                overflow: hidden;
            }
            .start-simulation-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(255, 107, 107, 0.6);
            }
            .start-simulation-btn:active {
                transform: translateY(0);
            }
            .start-simulation-btn.running {
                background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
                box-shadow: 0 4px 15px rgba(78, 205, 196, 0.4);
            }
            .start-simulation-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .simulation-status {
                display: flex;
                align-items: center;
                gap: 15px;
                font-size: 1rem;
                color: #4ecdc4;
            }
            .status-dot {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #4ecdc4;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            /* Container */
            .container { 
                display: grid; 
                grid-template-columns: 1fr 1fr;
                gap: 25px; 
                padding: 0 40px;
                max-width: 1400px;
                margin: 0 auto;
            }
            
            /* Top row for system status panels */
            .top-row {
                display: grid;
                grid-template-columns: 1fr 0.7fr 1fr;
                gap: 20px;
                margin-bottom: 25px;
                padding: 0 40px;
                max-width: 1400px;
                margin-left: auto;
                margin-right: auto;
            }
            
            /* Panels */
            .panel { 
                background: rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(15px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                padding: 25px;
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            .panel:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
                border-color: rgba(255, 255, 255, 0.2);
            }
            .wide-panel { grid-column: span 2; }
            
            .panel h2 {
                font-size: 1.4rem;
                font-weight: 600;
                margin-bottom: 20px;
                color: #4ecdc4;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            /* Status indicators */
            .status { 
                display: flex; 
                align-items: center; 
                gap: 12px; 
                margin: 12px 0;
                padding: 12px;
                background: rgba(255, 255, 255, 0.03);
                border-radius: 8px;
                transition: all 0.2s ease;
            }
            .status:hover {
                background: rgba(255, 255, 255, 0.08);
            }
            .status-indicator { 
                width: 14px; 
                height: 14px; 
                border-radius: 50%; 
                box-shadow: 0 0 10px currentColor;
            }
            .status-green { background: #4ecdc4; color: #4ecdc4; }
            .status-red { background: #ff6b6b; color: #ff6b6b; }
            
            /* Metrics */
            .metric { 
                display: flex; 
                justify-content: space-between; 
                padding: 12px 0; 
                border-bottom: 1px solid rgba(255, 255, 255, 0.1); 
                transition: all 0.2s ease;
            }
            .metric:hover {
                background: rgba(255, 255, 255, 0.03);
                margin: 0 -12px;
                padding: 12px;
                border-radius: 8px;
            }
            .metric:last-child { border-bottom: none; }
            .value { 
                font-weight: 600; 
                color: #4ecdc4;
                font-size: 1.1rem;
            }
            
            /* Buttons */
            .refresh-btn { 
                background: linear-gradient(135deg, #45b7d1 0%, #2c5aa0 100%);
                color: white; 
                border: none; 
                padding: 12px 24px; 
                border-radius: 25px; 
                cursor: pointer;
                font-weight: 500;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(69, 183, 209, 0.3);
            }
            .refresh-btn:hover { 
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(69, 183, 209, 0.5);
            }
            
            /* Digital Twin Sections */
            .digital-twin-section { 
                background: rgba(78, 205, 196, 0.1);
                border: 1px solid rgba(78, 205, 196, 0.2);
                padding: 15px; 
                border-radius: 12px; 
                margin: 15px 0;
            }
            .hardware-status { 
                display: grid; 
                grid-template-columns: 1fr 1fr; 
                gap: 12px; 
                margin: 15px 0; 
            }
            .hardware-item { 
                background: rgba(255, 255, 255, 0.05);
                padding: 12px; 
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: all 0.2s ease;
            }
            .hardware-item:hover {
                background: rgba(255, 255, 255, 0.1);
            }
            
            /* Validation Status */
            .validation-status { 
                padding: 15px; 
                border-radius: 12px; 
                margin: 15px 0;
                border-left: 4px solid;
            }
            .validation-approved { 
                background: rgba(40, 167, 69, 0.1);
                border-color: #28a745;
                color: #4ecdc4;
            }
            .validation-rejected { 
                background: rgba(220, 53, 69, 0.1);
                border-color: #dc3545;
                color: #ff6b6b;
            }
            .twin-prediction { 
                background: rgba(0, 123, 255, 0.1);
                border: 1px solid rgba(0, 123, 255, 0.2);
                padding: 15px; 
                border-radius: 12px; 
                margin: 15px 0;
            }
            
            /* Reasoning and Analysis */
            .reasoning { 
                background: rgba(78, 205, 196, 0.1);
                border: 1px solid rgba(78, 205, 196, 0.2);
                padding: 15px; 
                border-radius: 12px; 
                margin: 15px 0; 
                font-size: 0.95rem; 
                line-height: 1.5;
            }
            .cnn-analysis { 
                background: rgba(255, 193, 7, 0.1);
                border: 1px solid rgba(255, 193, 7, 0.3);
                padding: 12px; 
                border-radius: 8px; 
                margin: 10px 0;
            }
            
            /* Decision History */
            .decision-log { 
                max-height: 500px; 
                overflow-y: auto;
                scrollbar-width: thin;
                scrollbar-color: rgba(78, 205, 196, 0.5) transparent;
            }
            .decision-log::-webkit-scrollbar {
                width: 6px;
            }
            .decision-log::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
            }
            .decision-log::-webkit-scrollbar-thumb {
                background: rgba(78, 205, 196, 0.5);
                border-radius: 3px;
            }
            .decision-item { 
                background: rgba(255, 255, 255, 0.05);
                margin: 12px 0; 
                padding: 15px; 
                border-radius: 12px; 
                border-left: 4px solid #4ecdc4;
                transition: all 0.2s ease;
            }
            .decision-item:hover {
                background: rgba(255, 255, 255, 0.08);
                transform: translateX(5px);
            }
            .decision-timestamp { 
                font-size: 0.85rem; 
                color: rgba(255, 255, 255, 0.7); 
                margin-bottom: 8px;
            }
            
            /* Parameter Comparison */
            .param-comparison { 
                display: grid; 
                grid-template-columns: 1fr auto 1fr; 
                gap: 15px; 
                align-items: center; 
                margin: 12px 0; 
                padding: 12px; 
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            .param-current { text-align: right; color: rgba(255, 255, 255, 0.7); }
            .param-arrow { font-weight: bold; color: #4ecdc4; font-size: 1.2rem; }
            .param-new { color: #4ecdc4; font-weight: 600; }
            .param-arrow.up { color: #4ecdc4; }
            .param-arrow.down { color: #ff6b6b; }
            
            /* Loading animations */
            .loading {
                opacity: 0.7;
                animation: breathe 2s ease-in-out infinite;
            }
            @keyframes breathe {
                0%, 100% { opacity: 0.7; }
                50% { opacity: 1; }
            }
            
            /* Liquid Loading Bar */
            .ai-update-loader {
                margin: 15px 0;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                height: 40px;
                overflow: hidden;
                position: relative;
                backdrop-filter: blur(5px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .loader-label {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.9rem;
                font-weight: 600;
                color: #ffffff;
                z-index: 3;
                text-shadow: 0 1px 2px rgba(0,0,0,0.5);
            }
            
            .liquid-fill {
                position: absolute;
                top: 0;
                left: 0;
                width: 0%;
                height: 100%;
                background: linear-gradient(90deg, #4ecdc4 0%, #44a08d 50%, #45b7d1 100%);
                transition: width 0.3s ease;
                border-radius: 20px 0 0 20px;
                overflow: hidden;
            }
            
            .liquid-fill::before {
                content: '';
                position: absolute;
                top: 0;
                right: -20px;
                width: 40px;
                height: 100%;
                background: linear-gradient(90deg, rgba(78, 205, 196, 0.8) 0%, rgba(68, 160, 141, 0.8) 50%, rgba(69, 183, 209, 0.8) 100%);
                border-radius: 50%;
                animation: wave 2s ease-in-out infinite;
            }
            
            .liquid-fill::after {
                content: '';
                position: absolute;
                top: 10%;
                right: -25px;
                width: 30px;
                height: 80%;
                background: linear-gradient(90deg, rgba(78, 205, 196, 0.6) 0%, rgba(68, 160, 141, 0.6) 50%, rgba(69, 183, 209, 0.6) 100%);
                border-radius: 50%;
                animation: wave 2s ease-in-out infinite 0.5s;
            }
            
            @keyframes wave {
                0%, 100% {
                    transform: translateY(0) scale(1);
                    opacity: 0.8;
                }
                50% {
                    transform: translateY(-5px) scale(1.1);
                    opacity: 1;
                }
            }
            
            .loader-time {
                position: absolute;
                top: -25px;
                right: 0;
                font-size: 0.8rem;
                color: #4ecdc4;
                font-weight: 500;
            }
            
            .liquid-fill.complete {
                border-radius: 20px;
            }
            
            /* Dynamic Bar Graph Styles */
            .bar-graph-container {
                margin: 15px 0;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                padding: 15px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .bar-graph-title {
                font-size: 1rem;
                font-weight: 600;
                margin-bottom: 15px;
                color: #4ecdc4;
                text-align: center;
            }
            
            .parameter-bar {
                margin-bottom: 20px;
            }
            
            .parameter-label {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
                font-size: 0.9rem;
                font-weight: 500;
            }
            
            .parameter-name {
                color: #ffffff;
            }
            
            .parameter-values {
                font-size: 0.8rem;
                color: #4ecdc4;
            }
            
            .bar-container {
                position: relative;
                height: 30px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                overflow: hidden;
                margin-bottom: 5px;
            }
            
            .bar-current {
                position: absolute;
                top: 0;
                left: 0;
                height: 100%;
                background: linear-gradient(90deg, #45b7d1 0%, #4ecdc4 100%);
                border-radius: 15px;
                transition: width 0.8s ease-in-out;
                opacity: 0.8;
            }
            
            .bar-proposed {
                position: absolute;
                top: 0;
                left: 0;
                height: 100%;
                background: linear-gradient(90deg, #ff6b6b 0%, #ee5a52 100%);
                border-radius: 15px;
                transition: width 0.8s ease-in-out;
                opacity: 0.9;
                border: 2px solid rgba(255, 255, 255, 0.3);
            }
            
            .bar-legend {
                display: flex;
                justify-content: space-between;
                font-size: 0.7rem;
                margin-top: 5px;
            }
            
            .legend-item {
                display: flex;
                align-items: center;
                gap: 5px;
            }
            
            .legend-color {
                width: 12px;
                height: 12px;
                border-radius: 3px;
            }
            
            .legend-current {
                background: linear-gradient(90deg, #45b7d1 0%, #4ecdc4 100%);
            }
            
            .legend-proposed {
                background: linear-gradient(90deg, #ff6b6b 0%, #ee5a52 100%);
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
            
            .bar-graph-summary {
                margin-top: 15px;
                padding: 10px;
                background: rgba(78, 205, 196, 0.1);
                border-radius: 8px;
                border-left: 3px solid #4ecdc4;
                font-size: 0.85rem;
                line-height: 1.4;
            }
            
            /* Notification animations */
            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            @keyframes slideOut {
                from {
                    transform: translateX(0);
                    opacity: 1;
                }
                to {
                    transform: translateX(100%);
                    opacity: 0;
                }
            }
            
            /* Mobile responsiveness */
            @media (max-width: 768px) {
                .container { 
                    grid-template-columns: 1fr;
                    padding: 0 20px;
                }
                .wide-panel { grid-column: span 1; }
                .header { padding: 20px; }
                .header h1 { font-size: 2rem; }
                .control-panel { 
                    margin: 0 20px 20px 20px;
                    flex-direction: column;
                    gap: 15px;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧪 CsPbBr3 Synthesis MVP - Live Monitor</h1>
            <p>Autonomous perovskite quantum dot synthesis with AI optimization</p>
        </div>
        
        <div class="control-panel">
            <div class="simulation-status">
                <div class="status-dot" id="simulation-dot"></div>
                <span id="simulation-text">Ready to start simulation</span>
            </div>
            <button class="start-simulation-btn" id="start-simulation-btn" onclick="toggleSimulation()">
                🚀 Start Simulation
            </button>
        </div>
        
        <div class="top-row">
            <div class="panel">
                <h2>🔧 System Status</h2>
                <div id="system-status">
                    <div class="status">
                        <div class="status-indicator status-green"></div>
                        <span>Loading system status...</span>
                    </div>
                </div>
                <button class="refresh-btn" onclick="refreshData()">🔄 Refresh</button>
            </div>
            
            <div class="panel">
                <h2>📊 Metrics</h2>
                <div id="current-metrics">
                    <div class="metric">
                        <span>Loading...</span>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <h2>🤖 AI Components</h2>
                <div class="ai-update-loader">
                    <div class="loader-time" id="loader-time">30s</div>
                    <div class="liquid-fill" id="liquid-fill"></div>
                    <div class="loader-label" id="loader-label">Next AI Update</div>
                </div>
                <div id="ai-status">
                    <div class="metric">
                        <span>Loading AI status...</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="container">
            
            <div class="panel wide-panel">
                <h2>🔧 Digital Twin Status (Aroyston)</h2>
                <div id="digital-twin-status">
                    <div class="metric">
                        <span>Loading digital twin status...</span>
                    </div>
                </div>
            </div>
            
            <div class="panel wide-panel">
                <h2>🎯 RL Parameter Recommendations</h2>
                <div id="rl-recommendations">
                    <div class="reasoning">
                        Loading RL recommendations...
                    </div>
                    <div id="parameter-comparison">
                        <p>No parameter changes yet...</p>
                    </div>
                </div>
            </div>
            
            <div class="panel wide-panel">
                <h2>📈 Decision History (Last 10)</h2>
                <div id="decision-history" class="decision-log">
                    <p>Loading decision history...</p>
                </div>
            </div>
        </div>
        
        <script>
            async function fetchStatus() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    updateDashboard(data);
                } catch (error) {
                    console.error('Error fetching status:', error);
                }
            }
            
            function updateDashboard(data) {
                // Check if this is a new update by comparing timestamp
                const currentTimestamp = data.last_synthesis ? data.last_synthesis.timestamp : null;
                if (currentTimestamp && currentTimestamp !== lastUpdateTimestamp) {
                    lastUpdateTimestamp = currentTimestamp;
                    // Reset timer when real AI update occurs (only during simulation)
                    if (simulationRunning) {
                        resetAIUpdateTimer();
                    }
                }
                
                // System status
                const systemStatus = document.getElementById('system-status');
                systemStatus.innerHTML = `
                    <div class="status">
                        <div class="status-indicator ${data.monitoring ? 'status-green' : 'status-red'}"></div>
                        <span>Monitoring: ${data.monitoring ? 'Active' : 'Inactive'}</span>
                    </div>
                    <div class="status">
                        <div class="status-indicator ${data.cnn_model_loaded ? 'status-green' : 'status-red'}"></div>
                        <span>CNN Model: ${data.cnn_model_loaded ? 'Loaded' : 'Not Loaded'}</span>
                    </div>
                    <div class="status">
                        <div class="status-indicator ${data.rl_model_loaded ? 'status-green' : 'status-red'}"></div>
                        <span>RL Model: ${data.rl_model_loaded ? 'Loaded' : 'Not Loaded'}</span>
                    </div>
                    <div class="status">
                        <div class="status-indicator ${data.digital_twin_loaded ? 'status-green' : 'status-red'}"></div>
                        <span>Digital Twin: ${data.digital_twin_loaded ? 'Active' : 'Inactive'}</span>
                    </div>
                `;
                
                // Current metrics
                const metrics = document.getElementById('current-metrics');
                metrics.innerHTML = `
                    <div class="metric">
                        <span>Synthesis Cycles</span>
                        <span class="value">${data.synthesis_cycles}</span>
                    </div>
                    <div class="metric">
                        <span>Predictions Processed</span>
                        <span class="value">${data.predictions_processed}</span>
                    </div>
                `;
                
                // AI status
                const aiStatus = document.getElementById('ai-status');
                const lastSynthesis = data.last_synthesis;
                if (lastSynthesis && lastSynthesis.predictions) {
                    const pred = lastSynthesis.predictions;
                    aiStatus.innerHTML = `
                        <div class="metric">
                            <span>Last PLQY</span>
                            <span class="value">${(pred.predicted_plqy * 100).toFixed(1)}%</span>
                        </div>
                        <div class="metric">
                            <span>Last Emission Peak</span>
                            <span class="value">${pred.predicted_emission_peak} nm</span>
                        </div>
                        <div class="metric">
                            <span>Last FWHM</span>
                            <span class="value">${pred.predicted_fwhm} nm</span>
                        </div>
                    `;
                } else {
                    aiStatus.innerHTML = '<div class="metric"><span>No predictions yet</span></div>';
                }
                
                // Digital Twin Status
                updateDigitalTwinStatus(data);
                
                // RL Parameter Recommendations
                updateRLRecommendations(data);
                
                // Decision History
                updateDecisionHistory(data);
            }
            
            function updateDigitalTwinStatus(data) {
                const digitalTwinStatus = document.getElementById('digital-twin-status');
                
                if (data.digital_twin_status && data.digital_twin_loaded) {
                    const twinData = data.digital_twin_status;
                    const conditions = twinData.current_conditions;
                    const hardware = twinData.hardware_state;
                    const prediction = twinData.current_prediction;
                    
                    const hardwareStatusHTML = `
                        <div class="hardware-status">
                            <div class="hardware-item">
                                <strong>Cs Pump:</strong> ${hardware.pump_cs_status}
                            </div>
                            <div class="hardware-item">
                                <strong>Pb Pump:</strong> ${hardware.pump_pb_status}
                            </div>
                            <div class="hardware-item">
                                <strong>Heater:</strong> ${hardware.heater_status}
                            </div>
                            <div class="hardware-item">
                                <strong>Mixer:</strong> ${hardware.mixer_rpm} RPM
                            </div>
                            <div class="hardware-item">
                                <strong>Pressure:</strong> ${hardware.pressure.toFixed(2)} atm
                            </div>
                            <div class="hardware-item">
                                <strong>Flow Stability:</strong> ${(hardware.flow_stability * 100).toFixed(1)}%
                            </div>
                        </div>
                    `;
                    
                    const conditionsHTML = `
                        <div class="digital-twin-section">
                            <h4>🧪 Current Reaction Conditions</h4>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                                <div><strong>Cs Concentration:</strong> ${conditions.cs_br_concentration.toFixed(3)} M</div>
                                <div><strong>Pb Concentration:</strong> ${conditions.pb_br2_concentration.toFixed(3)} M</div>
                                <div><strong>Temperature:</strong> ${conditions.temperature.toFixed(1)} °C</div>
                                <div><strong>OA Concentration:</strong> ${conditions.oa_concentration.toFixed(3)} M</div>
                                <div><strong>OAM Concentration:</strong> ${conditions.oam_concentration.toFixed(3)} M</div>
                                <div><strong>Reaction Time:</strong> ${conditions.reaction_time.toFixed(1)} min</div>
                            </div>
                        </div>
                    `;
                    
                    const predictionHTML = `
                        <div class="twin-prediction">
                            <h4>🔮 Digital Twin Prediction</h4>
                            <div><strong>Predicted Outcome:</strong> ${prediction.predicted_outcome}</div>
                            <div><strong>Confidence:</strong> ${(prediction.confidence * 100).toFixed(1)}%</div>
                            <div style="font-size: 0.9em; margin-top: 5px;">
                                <strong>Class Probabilities:</strong><br>
                                Mixed Phase: ${(prediction.class_probabilities[0] * 100).toFixed(1)}% | 
                                0D: ${(prediction.class_probabilities[1] * 100).toFixed(1)}% | 
                                2D: ${(prediction.class_probabilities[2] * 100).toFixed(1)}% | 
                                3D: ${(prediction.class_probabilities[3] * 100).toFixed(1)}% | 
                                Failed: ${(prediction.class_probabilities[4] * 100).toFixed(1)}%
                            </div>
                        </div>
                    `;
                    
                    // Show last validation if available
                    let validationHTML = '';
                    if (data.last_synthesis && data.last_synthesis.digital_twin_validation) {
                        const validation = data.last_synthesis.digital_twin_validation;
                        const validationClass = validation.recommendation === 'approve' ? 'validation-approved' : 'validation-rejected';
                        validationHTML = `
                            <div class="validation-status ${validationClass}">
                                <h4>🛡️ Last RL Validation</h4>
                                <div><strong>Status:</strong> ${validation.recommendation.toUpperCase()}</div>
                                <div><strong>Reasoning:</strong> ${validation.reasoning}</div>
                                <div style="font-size: 0.8em; margin-top: 5px; color: #666;">
                                    Safety Checks: Temperature ${validation.safety_checks.temperature_safe ? '✅' : '❌'} | 
                                    Concentration ${validation.safety_checks.concentration_safe ? '✅' : '❌'} | 
                                    Ratio ${validation.safety_checks.ratio_reasonable ? '✅' : '❌'} | 
                                    Outcome ${validation.safety_checks.predicted_not_failed ? '✅' : '❌'}
                                </div>
                            </div>
                        `;
                    }
                    
                    // Get RL action data for comparison
                    const rlAction = data.last_synthesis ? data.last_synthesis.rl_action : null;
                    const validation = data.last_synthesis ? data.last_synthesis.digital_twin_validation : null;
                    
                    // Create Digital Twin bar graph
                    const digitalTwinBarGraph = createDigitalTwinBarGraph({
                        current: {
                            cs_concentration: conditions.cs_br_concentration,
                            pb_concentration: conditions.pb_br2_concentration,
                            temperature: conditions.temperature
                        },
                        proposed: rlAction ? {
                            cs_concentration: rlAction.cs_flow_rate * 2.0, // Convert flow to concentration
                            pb_concentration: rlAction.pb_flow_rate * 1.8,
                            temperature: rlAction.temperature
                        } : null,
                        prediction: prediction,
                        validation: validation,
                        hardware: hardware
                    });
                    
                    digitalTwinStatus.innerHTML = digitalTwinBarGraph;
                } else {
                    digitalTwinStatus.innerHTML = `
                        <div class="bar-graph-container">
                            <div class="bar-graph-title">🔮 Digital Twin Simulator</div>
                            <div style="text-align: center; color: #4ecdc4; padding: 20px;">
                                Digital Twin not available
                            </div>
                        </div>
                    `;
                }
            }
            
            function updateRLRecommendations(data) {
                const rlRecommendations = document.getElementById('rl-recommendations');
                
                if (data.last_synthesis && data.last_synthesis.rl_action) {
                    const action = data.last_synthesis.rl_action;
                    const pred = data.last_synthesis.predictions;
                    const currentParams = data.current_parameters;
                    const targetParams = data.target_parameters;
                    const previousParams = action.previous_params || {};
                    
                    // Create RL bar graph
                    const rlBarGraph = createRLBarGraph({
                        current: {
                            cs_flow: previousParams.cs_flow_rate || 0,
                            pb_flow: previousParams.pb_flow_rate || 0,
                            temperature: previousParams.temperature || 0
                        },
                        proposed: {
                            cs_flow: currentParams.cs_flow_rate || 0,
                            pb_flow: currentParams.pb_flow_rate || 0,
                            temperature: currentParams.temperature || 0
                        },
                        targets: targetParams,
                        reasoning: action.reasoning || 'No reasoning available',
                        cnnAnalysis: {
                            plqy: (pred.predicted_plqy * 100).toFixed(1),
                            peak: pred.predicted_emission_peak,
                            fwhm: pred.predicted_fwhm
                        }
                    });
                    
                    rlRecommendations.innerHTML = rlBarGraph;
                } else {
                    rlRecommendations.innerHTML = `
                        <div class="bar-graph-container">
                            <div class="bar-graph-title">🤖 RL Parameter Optimizer</div>
                            <div style="text-align: center; color: #4ecdc4; padding: 20px;">
                                Waiting for first CNN prediction and RL decision...
                            </div>
                        </div>
                    `;
                }
            }
            
            function generateParamComparison(paramName, oldValue, newValue) {
                if (oldValue === undefined || newValue === undefined) {
                    return `
                        <div class="param-comparison">
                            <div class="param-current">${paramName}: N/A</div>
                            <div class="param-arrow">→</div>
                            <div class="param-new">${newValue !== undefined ? newValue.toFixed(3) : 'N/A'}</div>
                        </div>
                    `;
                }
                
                const change = newValue - oldValue;
                let arrow = '→';
                let arrowClass = '';
                
                if (Math.abs(change) > 0.001) {
                    if (change > 0) {
                        arrow = '↑';
                        arrowClass = 'up';
                    } else {
                        arrow = '↓';
                        arrowClass = 'down';
                    }
                }
                
                return `
                    <div class="param-comparison">
                        <div class="param-current">${oldValue.toFixed(3)}</div>
                        <div class="param-arrow ${arrowClass}">${arrow}</div>
                        <div class="param-new">${newValue.toFixed(3)}</div>
                    </div>
                    <div style="text-align: center; font-size: 0.9em; color: #666; margin-bottom: 10px;">${paramName}</div>
                `;
            }
            
            function updateDecisionHistory(data) {
                const decisionHistory = document.getElementById('decision-history');
                
                if (data.parameter_history && data.parameter_history.length > 0) {
                    const historyHTML = data.parameter_history.reverse().map((decision, index) => {
                        const timestamp = new Date(decision.timestamp).toLocaleString();
                        const cnn = decision.cnn_analysis;
                        
                        return `
                            <div class="decision-item">
                                <div class="decision-timestamp">${timestamp}</div>
                                <div class="cnn-analysis">
                                    <strong>CNN:</strong> PLQY: ${(cnn.plqy * 100).toFixed(1)}% | 
                                    Peak: ${cnn.emission_peak}nm | FWHM: ${cnn.fwhm}nm
                                </div>
                                <div style="margin: 5px 0; font-size: 0.9em;">
                                    <strong>RL Action:</strong><br>
                                    Cs: ${decision.current_params.cs_flow_rate?.toFixed(3) || 'N/A'} → ${decision.new_params.cs_flow_rate.toFixed(3)} | 
                                    Pb: ${decision.current_params.pb_flow_rate?.toFixed(3) || 'N/A'} → ${decision.new_params.pb_flow_rate.toFixed(3)} | 
                                    T: ${decision.current_params.temperature?.toFixed(1) || 'N/A'} → ${decision.new_params.temperature.toFixed(1)}°C
                                </div>
                                <div style="font-size: 0.85em; color: #555; margin-top: 5px;">
                                    ${decision.reasoning}
                                </div>
                            </div>
                        `;
                    }).join('');
                    
                    decisionHistory.innerHTML = historyHTML;
                } else {
                    decisionHistory.innerHTML = '<p>No decisions recorded yet</p>';
                }
            }
            
            function refreshData() {
                fetchStatus();
            }
            
            // Simulation control
            let simulationRunning = false;
            
            // AI Update Timer
            let aiUpdateTimer = null;
            let aiUpdateStartTime = null;
            const AI_UPDATE_INTERVAL = 30000; // 30 seconds to match actual demo cycle
            let lastUpdateTimestamp = null;
            
            function startAIUpdateTimer() {
                const liquidFill = document.getElementById('liquid-fill');
                const loaderLabel = document.getElementById('loader-label');
                const loaderTime = document.getElementById('loader-time');
                
                if (!liquidFill || !loaderLabel || !loaderTime) return;
                
                aiUpdateStartTime = Date.now();
                
                // Reset the liquid fill
                liquidFill.style.width = '0%';
                liquidFill.classList.remove('complete');
                loaderLabel.textContent = 'AI Processing...';
                
                // Clear any existing timer
                if (aiUpdateTimer) {
                    clearInterval(aiUpdateTimer);
                }
                
                // Update every 50ms for smooth animation
                aiUpdateTimer = setInterval(() => {
                    const elapsed = Date.now() - aiUpdateStartTime;
                    const progress = Math.min(elapsed / AI_UPDATE_INTERVAL, 1);
                    const remainingTime = Math.max(0, Math.ceil((AI_UPDATE_INTERVAL - elapsed) / 1000));
                    
                    // Update liquid fill width
                    liquidFill.style.width = `${progress * 100}%`;
                    
                    // Update border radius when complete
                    if (progress >= 1) {
                        liquidFill.classList.add('complete');
                    }
                    
                    // Update time display
                    loaderTime.textContent = `${remainingTime}s`;
                    
                    // Update label based on progress
                    if (progress < 0.4) {
                        loaderLabel.textContent = 'CNN Analyzing...';
                    } else if (progress < 0.8) {
                        loaderLabel.textContent = 'RL Optimizing...';
                    } else if (progress < 0.95) {
                        loaderLabel.textContent = 'Digital Twin Validating...';
                    } else {
                        loaderLabel.textContent = 'Update Complete!';
                    }
                    
                    // When complete, reset for next cycle
                    if (progress >= 1) {
                        clearInterval(aiUpdateTimer);
                        setTimeout(() => {
                            if (simulationRunning) {
                                startAIUpdateTimer();
                            } else {
                                liquidFill.style.width = '0%';
                                liquidFill.classList.remove('complete');
                                loaderLabel.textContent = 'Simulation Stopped';
                                loaderTime.textContent = '--';
                            }
                        }, 200);
                    }
                }, 100);
            }
            
            function stopAIUpdateTimer() {
                if (aiUpdateTimer) {
                    clearInterval(aiUpdateTimer);
                    aiUpdateTimer = null;
                }
                
                const liquidFill = document.getElementById('liquid-fill');
                const loaderLabel = document.getElementById('loader-label');
                const loaderTime = document.getElementById('loader-time');
                
                if (liquidFill && loaderLabel && loaderTime) {
                    liquidFill.style.width = '0%';
                    liquidFill.classList.remove('complete');
                    loaderLabel.textContent = 'Next AI Update';
                    loaderTime.textContent = '--';
                }
            }
            
            // Reset timer when AI data actually updates
            function resetAIUpdateTimer() {
                if (simulationRunning && aiUpdateTimer) {
                    // Let current cycle complete, then restart
                    const liquidFill = document.getElementById('liquid-fill');
                    const loaderLabel = document.getElementById('loader-label');
                    
                    if (liquidFill) {
                        // Complete the current fill immediately
                        liquidFill.style.width = '100%';
                        liquidFill.classList.add('complete');
                        if (loaderLabel) {
                            loaderLabel.textContent = 'Update Complete!';
                        }
                        
                        // Then restart after a brief moment
                        setTimeout(() => {
                            if (simulationRunning) {
                                clearInterval(aiUpdateTimer);
                                startAIUpdateTimer();
                            }
                        }, 300);
                    }
                }
            }
            
            async function toggleSimulation() {
                const btn = document.getElementById('start-simulation-btn');
                const statusText = document.getElementById('simulation-text');
                const statusDot = document.getElementById('simulation-dot');
                
                if (!simulationRunning) {
                    // Start simulation
                    btn.disabled = true;
                    btn.textContent = '🔄 Starting...';
                    statusText.textContent = 'Starting simulation...';
                    
                    try {
                        const response = await fetch('/api/demo/start', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            }
                        });
                        
                        if (response.ok) {
                            const result = await response.json();
                            simulationRunning = true;
                            btn.classList.add('running');
                            btn.textContent = '⏹️ Stop Simulation';
                            statusText.textContent = `Simulation running (${result.interval_seconds}s intervals)`;
                            statusDot.style.background = '#4ecdc4';
                            
                            // Start AI update timer
                            startAIUpdateTimer();
                            
                            // Show success message
                            showNotification('✅ Simulation started successfully!', 'success');
                            
                            // Start monitoring simulation status
                            monitorSimulationStatus();
                        } else {
                            throw new Error('Failed to start simulation');
                        }
                    } catch (error) {
                        console.error('Error starting simulation:', error);
                        statusText.textContent = 'Error starting simulation';
                        showNotification('❌ Failed to start simulation', 'error');
                    } finally {
                        btn.disabled = false;
                    }
                } else {
                    // Stop simulation
                    btn.disabled = true;
                    btn.textContent = '🔄 Stopping...';
                    statusText.textContent = 'Stopping simulation...';
                    
                    try {
                        const response = await fetch('/api/demo/stop', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            }
                        });
                        
                        if (response.ok) {
                            simulationRunning = false;
                            btn.classList.remove('running');
                            btn.textContent = '🚀 Start Simulation';
                            statusText.textContent = 'Simulation stopped';
                            statusDot.style.background = '#ff6b6b';
                            
                            // Stop AI update timer
                            stopAIUpdateTimer();
                            
                            showNotification('⏹️ Simulation stopped', 'info');
                        } else {
                            throw new Error('Failed to stop simulation');
                        }
                    } catch (error) {
                        console.error('Error stopping simulation:', error);
                        statusText.textContent = 'Error stopping simulation';
                        showNotification('❌ Failed to stop simulation', 'error');
                    } finally {
                        btn.disabled = false;
                    }
                }
            }
            
            async function monitorSimulationStatus() {
                const checkStatus = async () => {
                    if (!simulationRunning) return;
                    
                    try {
                        const response = await fetch('/api/demo/status');
                        const status = await response.json();
                        
                        // If simulation completed naturally, update UI
                        if (!status.simulation_active && simulationRunning) {
                            const btn = document.getElementById('start-simulation-btn');
                            const statusText = document.getElementById('simulation-text');
                            const statusDot = document.getElementById('simulation-dot');
                            
                            simulationRunning = false;
                            btn.classList.remove('running');
                            btn.textContent = '🚀 Start Simulation';
                            statusText.textContent = 'Simulation completed';
                            statusDot.style.background = '#ff6b6b';
                            
                            // Stop AI update timer
                            stopAIUpdateTimer();
                            
                            showNotification('✅ Simulation completed successfully!', 'success');
                            return;
                        }
                        
                        // Continue monitoring if still running
                        if (simulationRunning) {
                            setTimeout(checkStatus, 2000); // Check every 2 seconds
                        }
                    } catch (error) {
                        console.error('Error checking simulation status:', error);
                        if (simulationRunning) {
                            setTimeout(checkStatus, 5000); // Retry in 5 seconds on error
                        }
                    }
                };
                
                setTimeout(checkStatus, 2000); // Start checking after 2 seconds
            }
            
            function showNotification(message, type) {
                // Create notification element
                const notification = document.createElement('div');
                notification.className = `notification notification-${type}`;
                notification.innerHTML = `
                    <div style="
                        position: fixed;
                        top: 20px;
                        right: 20px;
                        background: ${type === 'success' ? '#4ecdc4' : type === 'error' ? '#ff6b6b' : '#45b7d1'};
                        color: white;
                        padding: 15px 25px;
                        border-radius: 8px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                        z-index: 1000;
                        font-weight: 500;
                        animation: slideIn 0.3s ease;
                    ">
                        ${message}
                    </div>
                `;
                
                document.body.appendChild(notification);
                
                // Remove after 3 seconds
                setTimeout(() => {
                    notification.style.animation = 'slideOut 0.3s ease';
                    setTimeout(() => {
                        if (document.body.contains(notification)) {
                            document.body.removeChild(notification);
                        }
                    }, 300);
                }, 3000);
            }
            
            // Auto-refresh every 5 seconds
            setInterval(fetchStatus, 5000);
            
            // Initial load
            fetchStatus();
            
            // Initialize loading bar on page load
            document.addEventListener('DOMContentLoaded', function() {
                const liquidFill = document.getElementById('liquid-fill');
                const loaderLabel = document.getElementById('loader-label');
                const loaderTime = document.getElementById('loader-time');
                
                if (liquidFill && loaderLabel && loaderTime) {
                    liquidFill.style.width = '0%';
                    liquidFill.classList.remove('complete');
                    loaderLabel.textContent = 'Next AI Update';
                    loaderTime.textContent = '--';
                }
            });
            
            function createRLBarGraph(data) {
                const { current, proposed, targets, reasoning, cnnAnalysis } = data;
                
                // Calculate dynamic ranges that utilize full bar length
                function calculateRange(currentVal, proposedVal, absoluteMin, absoluteMax) {
                    const minVal = Math.min(currentVal, proposedVal);
                    const maxVal = Math.max(currentVal, proposedVal);
                    const range = maxVal - minVal;
                    
                    // Add padding to make bars more visible (20% on each side)
                    const padding = Math.max(range * 0.4, (absoluteMax - absoluteMin) * 0.1);
                    
                    const rangeMin = Math.max(absoluteMin, minVal - padding);
                    const rangeMax = Math.min(absoluteMax, maxVal + padding);
                    
                    return { min: rangeMin, max: rangeMax };
                }
                
                const ranges = {
                    cs_flow: calculateRange(current.cs_flow, proposed.cs_flow, 0, 0.4),
                    pb_flow: calculateRange(current.pb_flow, proposed.pb_flow, 0, 0.4),
                    temperature: calculateRange(current.temperature, proposed.temperature, 40, 250)
                };
                
                const cnnSection = `
                    <div class="bar-graph-summary">
                        <strong>🔮 CNN Analysis:</strong> PLQY: ${cnnAnalysis.plqy}% | Peak: ${cnnAnalysis.peak}nm | FWHM: ${cnnAnalysis.fwhm}nm<br>
                        <strong>🎯 Targets:</strong> PLQY: ${targets.plqy}% | Peak: ${targets.emission_peak}nm | FWHM: ${targets.fwhm}nm
                    </div>
                `;
                
                const parametersSection = `
                    <div class="parameter-bar">
                        <div class="parameter-label">
                            <span class="parameter-name">Cs Flow Rate</span>
                            <span class="parameter-values">Current: ${current.cs_flow.toFixed(3)} → Proposed: ${proposed.cs_flow.toFixed(3)} mL/min</span>
                        </div>
                        <div class="bar-container">
                            <div class="bar-current" style="width: ${((current.cs_flow - ranges.cs_flow.min) / (ranges.cs_flow.max - ranges.cs_flow.min) * 100)}%"></div>
                            <div class="bar-proposed" style="width: ${((proposed.cs_flow - ranges.cs_flow.min) / (ranges.cs_flow.max - ranges.cs_flow.min) * 100)}%"></div>
                        </div>
                        <div class="bar-legend">
                            <span style="font-size: 0.7rem; color: #888;">${ranges.cs_flow.min.toFixed(3)}</span>
                            <span style="font-size: 0.7rem; color: #888;">${ranges.cs_flow.max.toFixed(3)} mL/min</span>
                        </div>
                    </div>
                    
                    <div class="parameter-bar">
                        <div class="parameter-label">
                            <span class="parameter-name">Pb Flow Rate</span>
                            <span class="parameter-values">Current: ${current.pb_flow.toFixed(3)} → Proposed: ${proposed.pb_flow.toFixed(3)} mL/min</span>
                        </div>
                        <div class="bar-container">
                            <div class="bar-current" style="width: ${((current.pb_flow - ranges.pb_flow.min) / (ranges.pb_flow.max - ranges.pb_flow.min) * 100)}%"></div>
                            <div class="bar-proposed" style="width: ${((proposed.pb_flow - ranges.pb_flow.min) / (ranges.pb_flow.max - ranges.pb_flow.min) * 100)}%"></div>
                        </div>
                        <div class="bar-legend">
                            <span style="font-size: 0.7rem; color: #888;">${ranges.pb_flow.min.toFixed(3)}</span>
                            <span style="font-size: 0.7rem; color: #888;">${ranges.pb_flow.max.toFixed(3)} mL/min</span>
                        </div>
                    </div>
                    
                    <div class="parameter-bar">
                        <div class="parameter-label">
                            <span class="parameter-name">Temperature</span>
                            <span class="parameter-values">Current: ${current.temperature.toFixed(1)} → Proposed: ${proposed.temperature.toFixed(1)} °C</span>
                        </div>
                        <div class="bar-container">
                            <div class="bar-current" style="width: ${((current.temperature - ranges.temperature.min) / (ranges.temperature.max - ranges.temperature.min) * 100)}%"></div>
                            <div class="bar-proposed" style="width: ${((proposed.temperature - ranges.temperature.min) / (ranges.temperature.max - ranges.temperature.min) * 100)}%"></div>
                        </div>
                        <div class="bar-legend">
                            <span style="font-size: 0.7rem; color: #888;">${ranges.temperature.min.toFixed(1)}</span>
                            <span style="font-size: 0.7rem; color: #888;">${ranges.temperature.max.toFixed(1)} °C</span>
                        </div>
                    </div>
                    
                    <div class="bar-legend">
                        <div class="legend-item">
                            <div class="legend-color legend-current"></div>
                            <span>Current Parameters</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color legend-proposed"></div>
                            <span>RL Recommendations</span>
                        </div>
                    </div>
                `;
                
                const reasoningSection = `
                    <div class="bar-graph-summary">
                        <strong>🧠 RL Reasoning:</strong><br>
                        ${reasoning}
                    </div>
                `;
                
                return `
                    <div class="bar-graph-container">
                        <div class="bar-graph-title">🤖 RL Parameter Optimizer</div>
                        ${cnnSection}
                        ${parametersSection}
                        ${reasoningSection}
                    </div>
                `;
            }
            
            function createDigitalTwinBarGraph(data) {
                const { current, proposed, prediction, validation, hardware } = data;
                
                // Calculate dynamic ranges that utilize full bar length
                function calculateRange(currentVal, proposedVal, absoluteMin, absoluteMax) {
                    if (!proposedVal) {
                        // If no proposed value, create a range around current value
                        const padding = (absoluteMax - absoluteMin) * 0.2;
                        const rangeMin = Math.max(absoluteMin, currentVal - padding);
                        const rangeMax = Math.min(absoluteMax, currentVal + padding);
                        return { min: rangeMin, max: rangeMax };
                    }
                    
                    const minVal = Math.min(currentVal, proposedVal);
                    const maxVal = Math.max(currentVal, proposedVal);
                    const range = maxVal - minVal;
                    
                    // Add padding to make bars more visible (20% on each side)
                    const padding = Math.max(range * 0.4, (absoluteMax - absoluteMin) * 0.1);
                    
                    const rangeMin = Math.max(absoluteMin, minVal - padding);
                    const rangeMax = Math.min(absoluteMax, maxVal + padding);
                    
                    return { min: rangeMin, max: rangeMax };
                }
                
                const ranges = proposed ? {
                    cs_concentration: calculateRange(current.cs_concentration, proposed.cs_concentration, 0, 3.0),
                    pb_concentration: calculateRange(current.pb_concentration, proposed.pb_concentration, 0, 2.5),
                    temperature: calculateRange(current.temperature, proposed.temperature, 40, 250)
                } : {
                    cs_concentration: calculateRange(current.cs_concentration, null, 0, 3.0),
                    pb_concentration: calculateRange(current.pb_concentration, null, 0, 2.5),
                    temperature: calculateRange(current.temperature, null, 40, 250)
                };
                
                const predictionSection = `
                    <div class="bar-graph-summary">
                        <strong>🔮 Prediction:</strong> ${prediction.predicted_outcome} (${(prediction.confidence * 100).toFixed(1)}% confidence)<br>
                        <strong>🎯 Class Probabilities:</strong> 
                        Mixed: ${(prediction.class_probabilities[0] * 100).toFixed(1)}% | 
                        0D: ${(prediction.class_probabilities[1] * 100).toFixed(1)}% | 
                        2D: ${(prediction.class_probabilities[2] * 100).toFixed(1)}% | 
                        3D: ${(prediction.class_probabilities[3] * 100).toFixed(1)}% | 
                        Failed: ${(prediction.class_probabilities[4] * 100).toFixed(1)}%
                    </div>
                `;
                
                const parametersSection = proposed ? `
                    <div class="parameter-bar">
                        <div class="parameter-label">
                            <span class="parameter-name">Cs Concentration</span>
                            <span class="parameter-values">Current: ${current.cs_concentration.toFixed(3)} → Proposed: ${proposed.cs_concentration.toFixed(3)} M</span>
                        </div>
                        <div class="bar-container">
                            <div class="bar-current" style="width: ${((current.cs_concentration - ranges.cs_concentration.min) / (ranges.cs_concentration.max - ranges.cs_concentration.min) * 100)}%"></div>
                            <div class="bar-proposed" style="width: ${((proposed.cs_concentration - ranges.cs_concentration.min) / (ranges.cs_concentration.max - ranges.cs_concentration.min) * 100)}%"></div>
                        </div>
                        <div class="bar-legend">
                            <span style="font-size: 0.7rem; color: #888;">${ranges.cs_concentration.min.toFixed(3)}</span>
                            <span style="font-size: 0.7rem; color: #888;">${ranges.cs_concentration.max.toFixed(3)} M</span>
                        </div>
                    </div>
                    
                    <div class="parameter-bar">
                        <div class="parameter-label">
                            <span class="parameter-name">Pb Concentration</span>
                            <span class="parameter-values">Current: ${current.pb_concentration.toFixed(3)} → Proposed: ${proposed.pb_concentration.toFixed(3)} M</span>
                        </div>
                        <div class="bar-container">
                            <div class="bar-current" style="width: ${((current.pb_concentration - ranges.pb_concentration.min) / (ranges.pb_concentration.max - ranges.pb_concentration.min) * 100)}%"></div>
                            <div class="bar-proposed" style="width: ${((proposed.pb_concentration - ranges.pb_concentration.min) / (ranges.pb_concentration.max - ranges.pb_concentration.min) * 100)}%"></div>
                        </div>
                        <div class="bar-legend">
                            <span style="font-size: 0.7rem; color: #888;">${ranges.pb_concentration.min.toFixed(3)}</span>
                            <span style="font-size: 0.7rem; color: #888;">${ranges.pb_concentration.max.toFixed(3)} M</span>
                        </div>
                    </div>
                    
                    <div class="parameter-bar">
                        <div class="parameter-label">
                            <span class="parameter-name">Temperature</span>
                            <span class="parameter-values">Current: ${current.temperature.toFixed(1)} → Proposed: ${proposed.temperature.toFixed(1)} °C</span>
                        </div>
                        <div class="bar-container">
                            <div class="bar-current" style="width: ${((current.temperature - ranges.temperature.min) / (ranges.temperature.max - ranges.temperature.min) * 100)}%"></div>
                            <div class="bar-proposed" style="width: ${((proposed.temperature - ranges.temperature.min) / (ranges.temperature.max - ranges.temperature.min) * 100)}%"></div>
                        </div>
                        <div class="bar-legend">
                            <span style="font-size: 0.7rem; color: #888;">${ranges.temperature.min.toFixed(1)}</span>
                            <span style="font-size: 0.7rem; color: #888;">${ranges.temperature.max.toFixed(1)} °C</span>
                        </div>
                    </div>
                    
                    <div class="bar-legend">
                        <div class="legend-item">
                            <div class="legend-color legend-current"></div>
                            <span>Current Conditions</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color legend-proposed"></div>
                            <span>RL Proposed Changes</span>
                        </div>
                    </div>
                ` : `
                    <div style="text-align: center; padding: 15px; color: #4ecdc4;">
                        No RL recommendations yet
                    </div>
                `;
                
                const validationSection = validation ? `
                    <div class="bar-graph-summary">
                        <strong>🛡️ Validation:</strong> ${validation.recommendation.toUpperCase()} 
                        (${validation.is_safe ? '✅ Safe' : '⚠️ Unsafe'})<br>
                        <strong>Reasoning:</strong> ${validation.reasoning}
                    </div>
                ` : '';
                
                const hardwareSection = `
                    <div class="bar-graph-summary">
                        <strong>⚙️ Hardware:</strong> 
                        Cs Pump: ${hardware.pump_cs_status} | 
                        Pb Pump: ${hardware.pump_pb_status} | 
                        Heater: ${hardware.heater_status} | 
                        Mixer: ${hardware.mixer_rpm.toFixed(0)} RPM | 
                        Pressure: ${hardware.pressure.toFixed(2)} atm | 
                        Flow Stability: ${(hardware.flow_stability * 100).toFixed(1)}%
                    </div>
                `;
                
                return `
                    <div class="bar-graph-container">
                        <div class="bar-graph-title">🔮 Digital Twin Simulator</div>
                        ${predictionSection}
                        ${parametersSection}
                        ${validationSection}
                        ${hardwareSection}
                    </div>
                `;
            }
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=html_content)

@app.get("/api/status")
async def api_status():
    """API endpoint for system status"""
    global integration_system
    if integration_system:
        return integration_system.get_status()
    else:
        return {"error": "System not initialized"}

@app.post("/api/demo/start")
async def start_demo():
    """Start demo simulation"""
    global integration_system, demo_simulator
    if integration_system:
        demo_simulator = DemoSimulator(integration_system)
        demo_simulator.start_demo(interval_seconds=30)
        return {"message": "Demo started", "interval_seconds": 30}
    else:
        raise HTTPException(status_code=500, detail="System not initialized")

@app.post("/api/demo/stop")
async def stop_demo():
    """Stop demo simulation"""
    global demo_simulator
    if demo_simulator:
        demo_simulator.stop_demo()
        return {"message": "Demo stopped"}
    else:
        raise HTTPException(status_code=400, detail="No demo running")

@app.get("/api/demo/status")
async def get_demo_status():
    """Get demo simulation status"""
    global simulation_active, demo_simulator
    return {
        "simulation_active": simulation_active,
        "is_running": demo_simulator.is_running if demo_simulator else False
    }

@app.get("/api/history")
async def get_history():
    """Get synthesis history"""
    global integration_system
    if integration_system:
        return {"history": integration_system.synthesis_history}
    else:
        raise HTTPException(status_code=500, detail="System not initialized")

@app.get("/api/digital-twin")
async def get_digital_twin_status():
    """Get digital twin status"""
    global integration_system
    if integration_system and integration_system.digital_twin:
        return integration_system.digital_twin.get_current_status()
    else:
        raise HTTPException(status_code=500, detail="Digital Twin not available")

@app.post("/api/digital-twin/validate")
async def validate_rl_action(rl_action: dict):
    """Validate RL action through digital twin"""
    global integration_system
    if integration_system and integration_system.digital_twin:
        validation = integration_system.digital_twin.validate_rl_recommendations(rl_action, {})
        return validation
    else:
        raise HTTPException(status_code=500, detail="Digital Twin not available")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""
    print("🧪 CsPbBr3 Synthesis MVP - Integrated AI System")
    print("=" * 60)
    print("🤖 Isaiah's CNN (94.4% R²) + Ryan's RL (PPO) + File Integration")
    print("📊 Web monitoring interface included")
    print("⏱️  Performance: CNN=30s, RL=10s")
    print("=" * 60)
    
    # Check for required files
    logger.info("🔍 Checking system requirements...")
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("real_time_predictions", exist_ok=True)
    os.makedirs("spectral_images_realtime", exist_ok=True)
    
    # Start web server
    logger.info("🌐 Starting web interface on http://localhost:8000")
    logger.info("📊 Monitor at: http://localhost:8000")
    logger.info("🚀 Demo API: POST http://localhost:8000/api/demo/start")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Application failed: {e}")

if __name__ == "__main__":
    main()