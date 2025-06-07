"""
Spectrum to Image Conversion for CsPbBr3 Perovskite CNN Analysis
Based on real spectral data format research

Key findings from research:
- Spectrometers output wavelength/intensity arrays
- Typical range: 400-700nm for visible, up to 1000nm for PL
- Photoluminescence peaks typically 500-600nm for perovskites
- Data format: arrays of [wavelength, intensity] pairs
- Resolution: 0.1-1nm typically
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from PIL import Image
import cv2

def spectrum_to_image(wavelengths, intensities, image_size=(224, 224), method='heatmap'):
    """
    Convert spectral data to RGB image for CNN analysis
    
    Based on research of real spectrometer data formats:
    - Wavelengths: typically 400-700nm array (300+ points)
    - Intensities: normalized 0-1 values
    - Output: RGB image that CNN can analyze
    
    Args:
        wavelengths (array): Wavelength values in nm (e.g., [400, 401, ..., 700])
        intensities (array): Intensity values (normalized 0-1)
        image_size (tuple): Output image dimensions for CNN
        method (str): 'heatmap', 'spectrogram', or 'colormap'
    
    Returns:
        numpy.ndarray: RGB image (H, W, 3) suitable for CNN
    """
    
    # Validate inputs
    wavelengths = np.array(wavelengths)
    intensities = np.array(intensities)
    
    if len(wavelengths) != len(intensities):
        raise ValueError("Wavelengths and intensities must have same length")
    
    # Normalize intensities to 0-1 range
    intensities = np.array(intensities)
    if intensities.max() > 0:
        intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
    
    if method == 'heatmap':
        return _create_spectral_heatmap(wavelengths, intensities, image_size)
    elif method == 'spectrogram':
        return _create_spectrogram_image(wavelengths, intensities, image_size)
    elif method == 'colormap':
        return _create_colormap_image(wavelengths, intensities, image_size)
    else:
        raise ValueError("Method must be 'heatmap', 'spectrogram', or 'colormap'")

def _create_spectral_heatmap(wavelengths, intensities, image_size):
    """
    Create 2D heatmap representation of spectrum
    
    This method creates a 2D visualization where:
    - X-axis represents wavelength
    - Y-axis represents intensity levels
    - Color intensity shows spectral strength
    """
    height, width = image_size
    
    # Create wavelength range (400-700nm typical for visible PL)
    wl_min, wl_max = 400, 700
    wl_range = np.linspace(wl_min, wl_max, width)
    
    # Interpolate spectrum to match image width
    if len(wavelengths) > 1:
        f = interpolate.interp1d(wavelengths, intensities, 
                               bounds_error=False, fill_value=0)
        spectrum_interp = f(wl_range)
    else:
        spectrum_interp = np.zeros(width)
    
    # Create 2D heatmap
    image = np.zeros((height, width, 3))
    
    for i, intensity in enumerate(spectrum_interp):
        if intensity > 0:
            # Create vertical intensity profile
            peak_height = int(intensity * height * 0.8)  # Use 80% of height
            center_y = height // 2
            
            # Gaussian-like profile around the peak
            for y in range(height):
                distance = abs(y - center_y)
                if distance < peak_height:
                    alpha = np.exp(-(distance / (peak_height/3))**2)
                    
                    # Convert wavelength to color
                    rgb = wavelength_to_rgb(wl_range[i])
                    image[y, i] = rgb * alpha * intensity
    
    return (image * 255).astype(np.uint8)

def _create_spectrogram_image(wavelengths, intensities, image_size):
    """
    Create spectrogram-style image with time/position axis
    
    This simulates how spectra might vary across a sample
    """
    height, width = image_size
    
    # Create base spectrum
    wl_min, wl_max = 400, 700
    wl_range = np.linspace(wl_min, wl_max, width)
    
    # Interpolate to match width
    if len(wavelengths) > 1:
        f = interpolate.interp1d(wavelengths, intensities, 
                               bounds_error=False, fill_value=0)
        base_spectrum = f(wl_range)
    else:
        base_spectrum = np.zeros(width)
    
    # Create 2D spectrogram with variations
    image = np.zeros((height, width, 3))
    
    for y in range(height):
        # Add slight variations to simulate sample heterogeneity
        noise_factor = 0.1 * np.random.normal(0, 0.1, width)
        spectrum_variant = base_spectrum * (1 + noise_factor)
        spectrum_variant = np.clip(spectrum_variant, 0, 1)
        
        for x, intensity in enumerate(spectrum_variant):
            if intensity > 0:
                rgb = wavelength_to_rgb(wl_range[x])
                image[y, x] = rgb * intensity
    
    return (image * 255).astype(np.uint8)

def _create_colormap_image(wavelengths, intensities, image_size):
    """
    Create false-color image using matplotlib colormaps
    """
    height, width = image_size
    
    # Create 2D intensity map
    wl_min, wl_max = 400, 700
    wl_range = np.linspace(wl_min, wl_max, width)
    
    # Interpolate spectrum
    if len(wavelengths) > 1:
        f = interpolate.interp1d(wavelengths, intensities, 
                               bounds_error=False, fill_value=0)
        spectrum_interp = f(wl_range)
    else:
        spectrum_interp = np.zeros(width)
    
    # Create 2D data
    data_2d = np.zeros((height, width))
    
    for i, intensity in enumerate(spectrum_interp):
        # Create column with intensity profile
        profile = np.exp(-np.linspace(-2, 2, height)**2) * intensity
        data_2d[:, i] = profile
    
    # Apply colormap
    from matplotlib import cm
    colormap = cm.get_cmap('viridis')
    image_colored = colormap(data_2d)
    
    # Convert to RGB (remove alpha channel)
    image_rgb = image_colored[:, :, :3]
    
    return (image_rgb * 255).astype(np.uint8)

def wavelength_to_rgb(wavelength):
    """
    Convert wavelength (nm) to RGB color
    Based on visible light spectrum
    """
    wavelength = float(wavelength)
    
    if wavelength < 380 or wavelength > 750:
        return np.array([0.0, 0.0, 0.0])  # Outside visible range
    
    if 380 <= wavelength < 440:
        # Violet to Blue
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= wavelength < 490:
        # Blue to Cyan
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength < 510:
        # Cyan to Green
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength < 580:
        # Green to Yellow
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= wavelength < 645:
        # Yellow to Red
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif 645 <= wavelength <= 750:
        # Red
        R = 1.0
        G = 0.0
        B = 0.0
    
    # Apply intensity factor for near-UV and near-IR
    factor = 1.0
    if 380 <= wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 700 < wavelength <= 750:
        factor = 0.3 + 0.7 * (750 - wavelength) / (750 - 700)
    
    return np.array([R, G, B]) * factor

def create_test_perovskite_spectrum():
    """
    Create realistic test spectrum for CsPbBr3 perovskites
    Based on literature values
    """
    # Typical CsPbBr3 PL: peak around 515nm, FWHM ~20-30nm
    wavelengths = np.linspace(400, 700, 300)
    
    # Create Gaussian peak at 515nm
    peak_wl = 515
    fwhm = 25
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    intensities = np.exp(-((wavelengths - peak_wl) / sigma)**2)
    
    # Add some noise and background
    noise = 0.05 * np.random.normal(0, 1, len(wavelengths))
    background = 0.1 * np.exp(-(wavelengths - 400) / 100)
    
    intensities = intensities + noise + background
    intensities = np.clip(intensities, 0, None)
    
    return wavelengths, intensities

def test_conversion_methods():
    """
    Test all conversion methods with realistic perovskite data
    """
    # Create test data
    wavelengths, intensities = create_test_perovskite_spectrum()
    
    methods = ['heatmap', 'spectrogram', 'colormap']
    images = {}
    
    for method in methods:
        try:
            image = spectrum_to_image(wavelengths, intensities, method=method)
            images[method] = image
            print(f"✅ {method}: Created {image.shape} image")
        except Exception as e:
            print(f"❌ {method}: Failed - {e}")
    
    return images

# Example usage for your FastAPI app
def preprocess_spectrum_for_cnn(wavelengths, intensities, model_input_size=(224, 224)):
    """
    Complete preprocessing pipeline for your CNN
    
    Args:
        wavelengths: Array of wavelength values from spectrometer
        intensities: Array of intensity values from spectrometer
        model_input_size: Expected input size for your CNN
    
    Returns:
        torch.Tensor: Preprocessed tensor ready for CNN
    """
    import torch
    from torchvision import transforms
    
    # Convert spectrum to image
    image = spectrum_to_image(wavelengths, intensities, 
                            image_size=model_input_size, 
                            method='heatmap')  # Choose best method for your model
    
    # Convert to PIL Image for torchvision transforms
    pil_image = Image.fromarray(image)
    
    # Apply same preprocessing as your training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension
    
    return tensor

if __name__ == "__main__":
    # Test the conversion
    print("🧪 Testing Spectrum-to-Image Conversion...")
    test_images = test_conversion_methods()
    
    # Test with realistic perovskite data
    wavelengths, intensities = create_test_perovskite_spectrum()
    
    print(f"\n📊 Test data: {len(wavelengths)} wavelength points")
    print(f"   Range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
    print(f"   Peak at: {wavelengths[np.argmax(intensities)]:.1f} nm")
    print(f"   Max intensity: {intensities.max():.3f}")
    
    # Test CNN preprocessing
    try:
        tensor = preprocess_spectrum_for_cnn(wavelengths, intensities)
        print(f"\n✅ CNN preprocessing successful!")
        print(f"   Output tensor shape: {tensor.shape}")
        print(f"   Tensor range: {tensor.min():.3f} to {tensor.max():.3f}")
    except Exception as e:
        print(f"\n❌ CNN preprocessing failed: {e}")