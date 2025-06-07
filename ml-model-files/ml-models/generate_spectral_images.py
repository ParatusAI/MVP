# ml-models/generate_spectral_images.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from PIL import Image
import cv2

class SpectralImageGenerator:
    """Convert synthesis data to realistic spectral images for CNN training"""
    
    def __init__(self):
        self.wavelength_range = np.linspace(400, 700, 512)  # 400-700nm, 512 points
        self.image_size = (224, 224)  # Standard CNN input size
        self.output_dir = Path("spectral_images")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different qualities
        for quality in ['excellent', 'good', 'fair', 'poor']:
            (self.output_dir / quality).mkdir(exist_ok=True)
    
    def generate_realistic_spectrum(self, plqy, emission_peak, fwhm, cs_flow, pb_flow, temperature):
        """Generate realistic absorption and PL spectra based on synthesis parameters"""
        
        wavelengths = self.wavelength_range
        
        # 1. Generate Photoluminescence (PL) Spectrum
        pl_intensity = self.generate_pl_spectrum(wavelengths, emission_peak, fwhm, plqy)
        
        # 2. Generate UV-Vis Absorption Spectrum  
        absorption = self.generate_absorption_spectrum(wavelengths, emission_peak, plqy, cs_flow, pb_flow, temperature)
        
        # 3. Add realistic experimental artifacts
        pl_intensity = self.add_experimental_artifacts(pl_intensity, plqy)
        absorption = self.add_experimental_artifacts(absorption, plqy)
        
        return wavelengths, pl_intensity, absorption
    
    def generate_pl_spectrum(self, wavelengths, emission_peak, fwhm, plqy):
        """Generate photoluminescence spectrum"""
        
        # Main emission peak (Gaussian)
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
        main_peak = plqy * np.exp(-0.5 * ((wavelengths - emission_peak) / sigma) ** 2)
        
        # Add some spectral features based on quality
        if plqy > 0.7:  # High quality - narrow, symmetric peak
            intensity = main_peak
        elif plqy > 0.5:  # Medium quality - slight asymmetry
            # Add small shoulder peak (defect states)
            shoulder_peak = 0.1 * plqy * np.exp(-0.5 * ((wavelengths - (emission_peak + 8)) / (sigma * 1.5)) ** 2)
            intensity = main_peak + shoulder_peak
        else:  # Low quality - broad, multiple peaks
            # Add defect emission
            defect_peak1 = 0.15 * plqy * np.exp(-0.5 * ((wavelengths - (emission_peak + 12)) / (sigma * 2)) ** 2)
            defect_peak2 = 0.08 * plqy * np.exp(-0.5 * ((wavelengths - (emission_peak - 15)) / (sigma * 1.8)) ** 2)
            intensity = main_peak + defect_peak1 + defect_peak2
        
        # Add baseline
        baseline = 0.01 + 0.02 * (1 - plqy)  # Poor quality = higher baseline
        intensity += baseline
        
        return intensity
    
    def generate_absorption_spectrum(self, wavelengths, emission_peak, plqy, cs_flow, pb_flow, temperature):
        """Generate UV-Vis absorption spectrum"""
        
        # Main absorption edge (around 510nm for CsPbBr3)
        absorption_edge = emission_peak - 5  # Stokes shift
        
        # Absorption coefficient (exponential tail)
        absorption = np.zeros_like(wavelengths)
        
        # Band edge absorption
        for i, wl in enumerate(wavelengths):
            if wl < absorption_edge:
                # Strong absorption below band gap
                absorption[i] = 0.8 + 0.2 * plqy
            else:
                # Exponential tail above band gap
                absorption[i] = (0.8 + 0.2 * plqy) * np.exp(-(wl - absorption_edge) / 10)
        
        # Add scattering (particle size effects)
        cs_pb_ratio = cs_flow / pb_flow
        if cs_pb_ratio < 0.8 or cs_pb_ratio > 1.8:
            # Poor stoichiometry = more scattering
            scattering = 0.1 * (wavelengths / 400) ** (-4)  # Rayleigh scattering
            absorption += scattering
        
        # Add contamination peaks for poor synthesis
        if plqy < 0.4:
            # PbBr2 contamination peak around 480nm
            contamination = 0.2 * np.exp(-0.5 * ((wavelengths - 480) / 15) ** 2)
            absorption += contamination
        
        return absorption
    
    def add_experimental_artifacts(self, spectrum, plqy):
        """Add realistic experimental noise and artifacts"""
        
        # Noise level depends on quality
        noise_level = 0.01 + 0.03 * (1 - plqy)
        noise = np.random.normal(0, noise_level, len(spectrum))
        
        # Add systematic baseline drift
        baseline_drift = np.linspace(0, 0.01 * (1 - plqy), len(spectrum))
        
        # Add occasional cosmic ray spikes (rare)
        if np.random.random() < 0.05:  # 5% chance
            spike_position = np.random.randint(0, len(spectrum))
            spectrum[spike_position] += np.random.uniform(0.1, 0.3)
        
        return spectrum + noise + baseline_drift
    
    def create_spectral_image(self, wavelengths, pl_intensity, absorption, plqy, emission_peak, fwhm):
        """Create a spectral image suitable for CNN input"""
        
        # Create figure with dark background (like real spectrometer software)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), facecolor='black')
        
        # Plot PL spectrum (top)
        ax1.plot(wavelengths, pl_intensity, color='lime', linewidth=2, alpha=0.8)
        ax1.fill_between(wavelengths, 0, pl_intensity, color='lime', alpha=0.3)
        ax1.set_xlim(450, 650)
        ax1.set_ylim(0, max(pl_intensity) * 1.1)
        ax1.set_ylabel('PL Intensity', color='white', fontsize=10)
        ax1.tick_params(colors='white', labelsize=8)
        ax1.set_facecolor('black')
        ax1.grid(True, alpha=0.3, color='gray')
        
        # Add peak marker
        peak_idx = np.argmax(pl_intensity)
        ax1.axvline(wavelengths[peak_idx], color='red', linestyle='--', alpha=0.7)
        ax1.text(wavelengths[peak_idx] + 10, max(pl_intensity) * 0.8, 
                f'{emission_peak:.1f}nm\nPLQY: {plqy:.3f}', 
                color='white', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        # Plot Absorption spectrum (bottom)
        ax2.plot(wavelengths, absorption, color='cyan', linewidth=2, alpha=0.8)
        ax2.fill_between(wavelengths, 0, absorption, color='cyan', alpha=0.3)
        ax2.set_xlim(450, 650)
        ax2.set_ylim(0, max(absorption) * 1.1)
        ax2.set_xlabel('Wavelength (nm)', color='white', fontsize=10)
        ax2.set_ylabel('Absorbance', color='white', fontsize=10)
        ax2.tick_params(colors='white', labelsize=8)
        ax2.set_facecolor('black')
        ax2.grid(True, alpha=0.3, color='gray')
        
        plt.tight_layout()
        
        # Save to temporary file
        temp_path = "temp_spectrum.png"
        plt.savefig(temp_path, facecolor='black', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Load image and resize for CNN
        img = Image.open(temp_path)
        img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Clean up
        Path(temp_path).unlink()
        
        return img_array
    
    def process_csv_to_images(self, csv_file):
        """Convert entire CSV dataset to spectral images"""
        
        print(f"Loading data from {csv_file}")
        df = pd.read_csv(csv_file)
        
        print(f"Processing {len(df)} samples into spectral images...")
        
        # Prepare metadata for training
        metadata = []
        
        for idx, row in df.iterrows():
            # Generate realistic spectra
            wavelengths, pl_intensity, absorption = self.generate_realistic_spectrum(
                plqy=row['plqy'],
                emission_peak=row['emission_peak'], 
                fwhm=row['fwhm'],
                cs_flow=row['cs_flow_rate'],
                pb_flow=row['pb_flow_rate'],
                temperature=row['temperature']
            )
            
            # Create spectral image
            img_array = self.create_spectral_image(
                wavelengths, pl_intensity, absorption,
                row['plqy'], row['emission_peak'], row['fwhm']
            )
            
            # Save image in appropriate quality folder
            quality = row['quality_class']
            filename = f"spectrum_{idx:05d}.png"
            filepath = self.output_dir / quality / filename
            
            # Save as PNG
            Image.fromarray(img_array).save(filepath)
            
            # Store metadata
            metadata.append({
                'filename': str(filepath),
                'cs_flow_rate': row['cs_flow_rate'],
                'pb_flow_rate': row['pb_flow_rate'],
                'temperature': row['temperature'],
                'residence_time': row['residence_time'],
                'plqy': row['plqy'],
                'emission_peak': row['emission_peak'],
                'fwhm': row['fwhm'],
                'quality_class': row['quality_class']
            })
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)} images...")
        
        # Save metadata
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Successfully created {len(metadata)} spectral images!")
        print(f"📁 Images saved in: {self.output_dir}")
        print(f"📊 Metadata saved in: {metadata_file}")
        
        # Print distribution
        quality_counts = df['quality_class'].value_counts()
        print(f"\nImage distribution:")
        for quality, count in quality_counts.items():
            print(f"  {quality}: {count} images")
        
        return metadata

# Main execution
if __name__ == "__main__":
    generator = SpectralImageGenerator()
    
    # Convert your CSV data to spectral images
    metadata = generator.process_csv_to_images("my_training_data_2000.csv")
    
    print(f"\n🎉 Ready for CNN training on spectral images!")
    print(f"📸 {len(metadata)} spectral images created")
    print(f"🧠 Each image contains both PL and absorption spectra")
    print(f"🎯 Images organized by quality class for easy training")