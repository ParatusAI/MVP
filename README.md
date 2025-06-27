# AI Spectral Agent - CsPbBr3 Synthesis Optimization

**Autonomous perovskite quantum dot synthesis optimization using AI-driven spectral analysis and reinforcement learning**

## 🔬 Project Overview

This project implements an AI-powered system for optimizing CsPbBr3 (cesium lead bromide) perovskite quantum dot synthesis in real-time. The system combines:

- **Deep Learning**: CNN-based spectral image analysis (94.4% R² accuracy) - Isaiah's Component
- **Reinforcement Learning**: Parameter optimization using PPO algorithm - Ryan's Component
- **Digital Twin**: Real-time synthesis simulation and RL validation (91.55% accuracy) - Aroyston's Component
- **API Integration**: RESTful interfaces for seamless component communication

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ML Predictor  │    │   RL Agent      │    │  Digital Twin   │
│   (Isaiah)      │    │   (Ryan)        │    │  (Aroyston)     │
│                 │    │                 │    │                 │
│ • CNN Analysis  │◄──►│ • PPO Training  │◄──►│ • Hardware      │
│ • 94.4% R²      │    │ • Parameter     │    │   Simulation    │
│ • Real-time     │    │   Optimization  │    │ • RL Validation │
│ • Spectral Data │    │ • Action Gen.   │    │ • 91.55% Accuracy│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Integration   │
                    │     Layer       │
                    │                 │
                    │ • File Watcher  │
                    │ • CSV Pipeline  │
                    │ • Web Dashboard │
                    │ • Safety Checks │
                    └─────────────────┘
```

## 🔄 Enhanced Workflow

1. **Spectral Capture**: Real-time spectral images generated/captured
2. **CNN Analysis**: Isaiah's model predicts PLQY, emission peak, FWHM
3. **RL Decision**: Ryan's agent optimizes synthesis parameters  
4. **Digital Twin Validation**: Aroyston's system validates RL recommendations for safety
5. **Hardware Simulation**: Digital twin simulates hardware response to parameter changes
6. **Feedback Loop**: Continuous monitoring and optimization with safety validation

## 📁 Directory Structure

### `/ml-model-files/ml-models/`
**Machine Learning Core** - Isaiah's CNN implementation
- `app.py` - FastAPI server with CNN model (94.4% R² accuracy)
- `spectral_image_cnn_v3.py` - CNN architecture for spectral analysis
- `model_predictor.py` - Prediction engine and utilities
- `cspbbr3_final_model.pth` - Trained CNN model weights
- `spectral_image_simulator.py` - Real-time spectral simulation
- `mvp_demo_complete.py` - Complete MVP demonstration
- `for_ryan_file_watcher.py` - RL integration via file monitoring

### `/ai-agent/RL-agent/`
**Reinforcement Learning** - Ryan's optimization agent
- `rl_agent_api.py` - PPO-based parameter optimization API
- `rl_integration_example.py` - Integration examples
- `sample_targets.csv` - Target property definitions

### `/ai-agent/Digital Twin/digital-twin-V2/`
**Digital Twin System** - Aroyston's simulation and validation
- `working_prediction_pipeline.py` - Production-ready prediction system (91.55% accuracy)
- `fixed_ultimate_generator.py` - Balanced dataset generator with proper physics
- `system_status.py` - Real-time system monitoring
- `HONEST_RECOMMENDATIONS_SUCCESS_REPORT.md` - Comprehensive validation results
- `robust_final_results/` - Validated model files and training results

### `/integration/`
**System Integration** - Communication layer with digital twin validation
- API bridges and data flow management
- Digital twin safety validation
- Real-time hardware simulation

## 🎯 Target Properties

The system optimizes CsPbBr3 quantum dots for:

| Property | Target Range | Commercial Goal |
|----------|--------------|-----------------|
| **PLQY** | 65-92% | >80% (Excellent) |
| **Emission Peak** | 505-530 nm | 515-520 nm (Green) |
| **FWHM** | 12-60 nm | <25 nm (Narrow) |

## 🧪 Synthesis Parameters

The RL agent controls:

| Parameter | Range | Impact |
|-----------|-------|---------|
| **Cs Flow Rate** | 0.5-2.2 mL/min | Cs:Pb stoichiometry |
| **Pb Flow Rate** | 0.6-1.8 mL/min | Lead availability |
| **Temperature** | 100-200°C | Nucleation kinetics |
| **Residence Time** | 60-250 s | Crystal formation |

## 🚀 Quick Start

### 1. Start Database Services
```bash
docker-compose up -d
```

### 2. Run ML Predictor (Isaiah's CNN)
```bash
cd ml-model-files/ml-models/
pip install -r requirements.txt
python app.py
```

### 3. Start RL Agent (Ryan's Optimizer)
```bash
cd ai-agent/RL-agent/
python rl_agent_api.py
```

### 4. Run Integrated System with Digital Twin
```bash
python integrated_mvp_app.py
```

### 5. Access Web Dashboard
Navigate to `http://localhost:8000` to view the integrated dashboard with:
- Isaiah's CNN predictions
- Ryan's RL recommendations 
- Aroyston's Digital Twin status and validation

## 🔄 Enhanced Workflow with Digital Twin

1. **Spectral Capture**: Real-time spectral images generated/captured
2. **CNN Analysis**: Isaiah's model predicts PLQY, emission peak, FWHM (94.4% R²)
3. **RL Decision**: Ryan's agent optimizes synthesis parameters using PPO
4. **Digital Twin Validation**: Aroyston's system validates RL recommendations for safety
   - Temperature safety checks (80-250°C)
   - Concentration limits (Cs<3.0M, Pb<2.5M)
   - Stoichiometric ratio validation (0.3-5.0)
   - Outcome prediction (avoid failed synthesis)
5. **Hardware Simulation**: Digital twin simulates gradual hardware response
6. **Feedback Loop**: Continuous monitoring and optimization with safety validation

## 📊 Integration Methods

### Method 1: File-Based (CSV)
```python
# Isaiah creates prediction files
prediction_T030s.csv  # PLQY, emission_peak, FWHM, confidence

# Ryan's file watcher responds immediately
from for_ryan_file_watcher import RyanRLEnvironment
rl_env = RyanRLEnvironment()
rl_env.start_monitoring()
```

### Method 2: REST API
```python
# Direct API communication
import requests

# Get prediction
response = requests.post('http://localhost:8000/predict/', 
                        json={'cs_flow_rate': 1.2, 'pb_flow_rate': 1.0, 
                              'temperature': 85, 'residence_time': 120})

# Get RL action
response = requests.post('http://localhost:8001/step/',
                        json={'plqy': 0.75, 'emission': 518, 
                              'fwhm': 22, 'temperature': 85})
```

## 🧠 ML Model Performance

**Isaiah's CNN Architecture**:
- 4-layer CNN + 3-layer regressor
- Input: RGB spectral images (224x224)
- Output: Normalized PLQY, emission peak, FWHM
- **Performance**: 94.4% R² on test set
- Training: 5-fold stratified cross-validation

**Feature Engineering**:
- Spectral wavelength → RGB color mapping
- Intensity → 2D heatmap conversion
- Gaussian peak profiles for realistic visualization

## 🤖 RL Agent Details

**Ryan's PPO Implementation**:
- Algorithm: Proximal Policy Optimization
- Action Space: [cs_flow, pb_flow, temperature] (continuous)
- Observation Space: [current_PLQY, current_lambda, current_FWHM, current_temp, target_PLQY, target_lambda, target_FWHM]
- Reward Function: Weighted MSE with stability penalties

## 💾 Data Pipeline

### Training Data
- `cspbbr3_ml_dataset.csv` - 2000+ synthesis experiments
- Quality classes: Poor (0) → Fair (1) → Good (2) → Excellent (3)
- Stratified sampling ensures balanced representation

### Real-time Data
- Spectral images: `spectral_images_realtime/`
- Predictions: `real_time_predictions/`
- Results: `mvp_results_YYYYMMDD_HHMMSS.json`

## 🔧 API Endpoints

### ML Predictor API (Port 8000)
- `GET /` - System status
- `POST /predict/` - Parameter-based prediction
- `POST /predict_from_spectrum/` - Spectral analysis
- `GET /health/` - Health check

### RL Agent API (Port 8001)
- `GET /status` - Agent status
- `POST /reset` - Environment reset
- `POST /step` - Action prediction
- `POST /train` - Model training

### Digital Twin API (Port 8000)
- `GET /api/digital-twin` - Current digital twin status
- `POST /api/digital-twin/validate` - Validate RL actions
- `GET /` - Integrated web dashboard with all components

## 📈 Performance Metrics

- **CNN Accuracy**: 94.4% R² score (Isaiah's model)
- **Digital Twin Accuracy**: 91.55% classification accuracy (Aroyston's model)
- **Prediction Time**: <100ms per spectrum (CNN), <50ms per validation (Digital Twin)
- **RL Convergence**: ~50k training steps
- **System Latency**: <2s end-to-end with digital twin validation
- **Safety Validation**: 100% of RL actions validated before hardware deployment

## 🔬 Commercial Readiness

**Current Status**: MVP Complete with Digital Twin Integration ✅
- Proven 94.4% CNN prediction accuracy (Isaiah)
- Proven 91.55% digital twin accuracy (Aroyston)
- Real-time synthesis optimization with safety validation
- Autonomous parameter adjustment with RL validation
- File-based and API integration with web dashboard

**Next Steps**:
- Real hardware integration with digital twin validation
- Production deployment optimization
- Extended parameter space exploration
- Multi-objective optimization (cost, yield, quality)
- Enhanced digital twin physics modeling

## 👥 Team

- **Isaiah**: ML/CNN Development
- **Ryan**: RL/Optimization
- **Aroyston**: Digital Twin/Hardware

## 📚 Key Files

- `integrated_mvp_app.py` - **Main integrated system with all three components**
- `app.py` - Isaiah's ML API server
- `rl_agent_api.py` - Ryan's RL optimization engine  
- `working_prediction_pipeline.py` - Aroyston's digital twin system
- `for_ryan_file_watcher.py` - Real-time integration
- `cspbbr3_data_dictionary.txt` - Dataset documentation

---

**🎉 Status**: Ready for autonomous lab integration and commercial deployment with integrated digital twin validation!