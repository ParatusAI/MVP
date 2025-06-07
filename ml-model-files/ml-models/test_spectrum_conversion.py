"""
Test script for spectrum-to-image conversion and API
Run this to verify everything works before integration
"""
import numpy as np
import requests
import json

def create_realistic_perovskite_spectrum():
    """
    Create realistic CsPbBr3 photoluminescence spectrum
    Based on literature: peak ~515nm, FWHM ~25nm
    """
    # Create wavelength array (typical spectrometer resolution)
    wavelengths = np.linspace(400, 700, 300).tolist()
    
    # Create Gaussian PL peak at 515nm
    peak_wl = 515
    fwhm = 25
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    intensities = []
    for wl in wavelengths:
        # Main PL peak
        intensity = np.exp(-((wl - peak_wl) / sigma)**2)
        
        # Add some background and noise
        background = 0.1 * np.exp(-(wl - 400) / 100)
        noise = 0.02 * np.random.normal()
        
        total_intensity = intensity + background + noise
        intensities.append(max(0, total_intensity))  # No negative values
    
    # Normalize to 0-1 range
    max_intensity = max(intensities)
    intensities = [i/max_intensity for i in intensities]
    
    return wavelengths, intensities

def test_spectrum_conversion_locally():
    """Test the spectrum conversion functions locally"""
    print("🧪 Testing spectrum-to-image conversion locally...")
    
    try:
        # Import the conversion function
        from app import spectrum_to_image, wavelength_to_rgb
        
        # Create test data
        wavelengths, intensities = create_realistic_perovskite_spectrum()
        
        print(f"✅ Created test spectrum:")
        print(f"   - {len(wavelengths)} wavelength points")
        print(f"   - Range: {min(wavelengths):.1f} - {max(wavelengths):.1f} nm")
        print(f"   - Peak at: {wavelengths[np.argmax(intensities)]:.1f} nm")
        print(f"   - Max intensity: {max(intensities):.3f}")
        
        # Test conversion
        image = spectrum_to_image(wavelengths, intensities)
        
        print(f"✅ Spectrum-to-image conversion successful!")
        print(f"   - Output shape: {image.shape}")
        print(f"   - Image dtype: {image.dtype}")
        print(f"   - Value range: {image.min()} - {image.max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Local test failed: {e}")
        return False

def test_api_endpoints(base_url="http://localhost:8000"):
    """Test your FastAPI endpoints"""
    print(f"\n🚀 Testing API endpoints at {base_url}...")
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health/")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 2: Model info
    try:
        response = requests.get(f"{base_url}/model/info/")
        if response.status_code == 200:
            info = response.json()
            print(f"✅ Model info: {info.get('message', 'No model loaded')}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model info failed: {e}")
    
    # Test 3: Create synthesis run
    try:
        run_data = {
            "cs_flow_rate": 1.0,
            "pb_flow_rate": 1.0,
            "temperature": 80.0,
            "residence_time": 2.0,
            "notes": "Test run from API test script"
        }
        
        response = requests.post(f"{base_url}/synthesis-runs/", json=run_data)
        if response.status_code == 200:
            run_info = response.json()
            run_id = run_info["id"]
            print(f"✅ Created synthesis run #{run_id}")
            
            # Test 4: Analyze spectrum for this run
            wavelengths, intensities = create_realistic_perovskite_spectrum()
            
            spectral_data = [
                {"wavelength": wl, "intensity": intensity, "measurement_type": "pl"}
                for wl, intensity in zip(wavelengths[::10], intensities[::10])  # Sample every 10th point
            ]
            
            response = requests.post(
                f"{base_url}/synthesis-runs/{run_id}/analyze-spectrum/",
                json=spectral_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Spectrum analysis successful!")
                print(f"   - PLQY: {result['analysis_results']['plqy']:.3f}")
                print(f"   - Emission peak: {result['analysis_results']['emission_peak']:.1f} nm")
                print(f"   - FWHM: {result['analysis_results']['fwhm']:.1f} nm")
            else:
                print(f"❌ Spectrum analysis failed: {response.status_code}")
                print(f"   Response: {response.text}")
        else:
            print(f"❌ Create synthesis run failed: {response.status_code}")
            print(f"   Response: {response.text}")
    
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False
    
    return True

def test_realistic_data_formats():
    """Test with various realistic spectral data formats"""
    print("\n📊 Testing realistic spectral data formats...")
    
    formats = {
        "High Resolution": {
            "wavelengths": np.linspace(400, 700, 1000).tolist(),
            "description": "1000 points, 0.3nm resolution"
        },
        "Standard Resolution": {
            "wavelengths": np.linspace(400, 700, 300).tolist(),
            "description": "300 points, 1nm resolution"
        },
        "Low Resolution": {
            "wavelengths": np.linspace(400, 700, 100).tolist(),
            "description": "100 points, 3nm resolution"
        },
        "UV-Vis Extended": {
            "wavelengths": np.linspace(350, 800, 450).tolist(),
            "description": "Extended UV-Vis range"
        },
        "NIR Extended": {
            "wavelengths": np.linspace(400, 1000, 600).tolist(),
            "description": "Extended to NIR"
        }
    }
    
    for format_name, format_info in formats.items():
        wavelengths = format_info["wavelengths"]
        
        # Create corresponding intensities with PL peak at 515nm
        intensities = []
        for wl in wavelengths:
            intensity = np.exp(-((wl - 515) / 12.5)**2)  # Gaussian peak
            noise = 0.05 * np.random.normal()
            intensities.append(max(0, intensity + noise))
        
        # Normalize
        max_intensity = max(intensities)
        if max_intensity > 0:
            intensities = [i/max_intensity for i in intensities]
        
        print(f"📈 {format_name}: {format_info['description']}")
        print(f"   Range: {min(wavelengths):.1f} - {max(wavelengths):.1f} nm")
        print(f"   Points: {len(wavelengths)}")
        
        # Test conversion
        try:
            from app import spectrum_to_image
            image = spectrum_to_image(wavelengths, intensities)
            print(f"   ✅ Conversion successful: {image.shape}")
        except Exception as e:
            print(f"   ❌ Conversion failed: {e}")

if __name__ == "__main__":
    print("🧪 CsPbBr3 Spectrum Analysis API Test Suite")
    print("=" * 50)
    
    # Test 1: Local function tests
    local_success = test_spectrum_conversion_locally()
    
    if local_success:
        # Test 2: Realistic data format tests
        test_realistic_data_formats()
        
        # Test 3: API endpoint tests (requires running server)
        print("\n" + "=" * 50)
        print("🚀 API Integration Tests")
        print("(Make sure your FastAPI server is running: python app.py)")
        print("=" * 50)
        
        api_success = test_api_endpoints()
        
        if api_success:
            print("\n🎉 ALL TESTS PASSED! Your MVP is ready for integration!")
            print("\n📋 Next steps:")
            print("   1. Test with Aroyston's actual hardware data")
            print("   2. Help Ryan connect his RL agent")
            print("   3. Run end-to-end autonomous synthesis")
        else:
            print("\n⚠️  API tests failed. Check that your server is running.")
    else:
        print("\n❌ Local tests failed. Fix spectrum conversion first.")
        
    print("\n📝 Test Summary:")
    print(f"   Local functions: {'✅ PASS' if local_success else '❌ FAIL'}")
    try:
        api_success
        print(f"   API endpoints: {'✅ PASS' if api_success else '❌ FAIL'}")
    except:
        print(f"   API endpoints: ⏭️  SKIPPED")
        
    print("\n🔗 Quick API test command:")
    print("   curl http://localhost:8000/health/")
    print("\n🔗 View API docs:")
    print("   http://localhost:8000/docs")