#!/usr/bin/env python3
"""
Test Script for Integrated MVP System
====================================
Validates the integrated CsPbBr3 synthesis optimization system
"""

import os
import sys
import time
import requests
import subprocess
from pathlib import Path

def test_dependencies():
    """Test if all required dependencies are available"""
    print("🔍 Testing Dependencies...")
    
    required_packages = [
        'torch', 'torchvision', 'numpy', 'PIL', 
        'stable_baselines3', 'gymnasium', 
        'fastapi', 'uvicorn', 'pandas', 'sklearn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        print("Run: pip install -r requirements_integrated.txt")
        return False
    
    print("✅ All dependencies available")
    return True

def test_file_structure():
    """Test if required files and folders exist"""
    print("\n🔍 Testing File Structure...")
    
    required_files = [
        'integrated_mvp_app.py',
        'requirements_integrated.txt'
    ]
    
    optional_files = [
        'ml-model-files/ml-models/cspbbr3_final_model.pth',
        'ml-model-files/ml-models/cspbbr3_best_fold_model.pth',
        'models/ppo_integrated_mvp.zip'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            return False
    
    print("  📁 Optional model files:")
    for file in optional_files:
        if os.path.exists(file):
            print(f"    ✅ {file}")
        else:
            print(f"    ⚠️  {file} - Will train/create on startup")
    
    print("✅ File structure OK")
    return True

def test_import_system():
    """Test if the integrated system can be imported"""
    print("\n🔍 Testing System Import...")
    
    try:
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import main components
        from integrated_mvp_app import (
            ImprovedSpectralCNN, SpectralPredictor,
            FlowTempFwhmEnv, RLOptimizer,
            FileBasedIntegration, DemoSimulator
        )
        
        print("  ✅ ImprovedSpectralCNN")
        print("  ✅ SpectralPredictor") 
        print("  ✅ FlowTempFwhmEnv")
        print("  ✅ RLOptimizer")
        print("  ✅ FileBasedIntegration")
        print("  ✅ DemoSimulator")
        
        print("✅ System imports successful")
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_cnn_component():
    """Test CNN component initialization"""
    print("\n🔍 Testing CNN Component...")
    
    try:
        from integrated_mvp_app import ImprovedSpectralCNN, SpectralPredictor
        
        # Test CNN model creation
        model = ImprovedSpectralCNN()
        print("  ✅ CNN model created")
        
        # Test predictor
        predictor = SpectralPredictor()
        print("  ✅ SpectralPredictor created")
        
        # Test mock prediction
        mock_pred = predictor._create_mock_prediction()
        assert 'predicted_plqy' in mock_pred
        assert 'predicted_emission_peak' in mock_pred
        assert 'predicted_fwhm' in mock_pred
        print("  ✅ Mock predictions working")
        
        print("✅ CNN component OK")
        return True
        
    except Exception as e:
        print(f"  ❌ CNN test failed: {e}")
        return False

def test_rl_component():
    """Test RL component initialization"""
    print("\n🔍 Testing RL Component...")
    
    try:
        from integrated_mvp_app import FlowTempFwhmEnv, RLOptimizer
        
        # Test environment
        env = FlowTempFwhmEnv()
        print("  ✅ RL environment created")
        
        # Test environment functionality
        obs, info = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print("  ✅ Environment step working")
        
        # Test optimizer (will train new model if needed)
        print("  🏃 Creating RL optimizer (may train new model)...")
        optimizer = RLOptimizer()
        print("  ✅ RL optimizer created")
        
        # Test mock action
        mock_action = optimizer._mock_action()
        assert 'cs_flow_rate' in mock_action
        assert 'pb_flow_rate' in mock_action
        assert 'temperature' in mock_action
        print("  ✅ Mock actions working")
        
        print("✅ RL component OK")
        return True
        
    except Exception as e:
        print(f"  ❌ RL test failed: {e}")
        return False

def test_integration_system():
    """Test file-based integration system"""
    print("\n🔍 Testing Integration System...")
    
    try:
        from integrated_mvp_app import FileBasedIntegration
        
        # Create integration system
        integration = FileBasedIntegration()
        print("  ✅ Integration system created")
        
        # Test status
        status = integration.get_status()
        assert 'monitoring' in status
        assert 'folders' in status
        print("  ✅ Status reporting working")
        
        print("✅ Integration system OK")
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False

def test_web_interface():
    """Test if web interface can start"""
    print("\n🔍 Testing Web Interface...")
    
    try:
        from integrated_mvp_app import app
        print("  ✅ FastAPI app created")
        
        # Test if we can create the app without errors
        assert app is not None
        print("  ✅ App initialization successful")
        
        print("✅ Web interface OK")
        return True
        
    except Exception as e:
        print(f"  ❌ Web interface test failed: {e}")
        return False

def run_full_system_test():
    """Run a complete system test"""
    print("\n" + "="*60)
    print("🧪 INTEGRATED MVP SYSTEM TEST")
    print("="*60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("File Structure", test_file_structure),
        ("System Import", test_import_system),
        ("CNN Component", test_cnn_component),
        ("RL Component", test_rl_component),
        ("Integration System", test_integration_system),
        ("Web Interface", test_web_interface)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
    
    print("\n" + "="*60)
    print(f"🏁 TEST RESULTS: {passed}/{total} PASSED")
    print("="*60)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ System ready for deployment")
        print("\n🚀 To start the system:")
        print("   python integrated_mvp_app.py")
        print("   📊 Monitor at: http://localhost:8000")
        return True
    else:
        print("⚠️  SOME TESTS FAILED")
        print("🔧 Please fix issues before deployment")
        return False

if __name__ == "__main__":
    success = run_full_system_test()
    sys.exit(0 if success else 1)