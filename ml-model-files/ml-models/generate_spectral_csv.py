# ml-models/generate_spectral_csv.py
import pandas as pd
import numpy as np
from pathlib import Path
import json

class SpectralCSVGenerator:
    """Convert synthesis data to realistic spectral CSV data for ML training"""
    
    def __init__(self, wavelength_points=256):
        # Spectral parameters
        self.wavelength_points = wavelength_points
        self.wavelength_range = np.linspace(400, 700, wavelength_points)  # 400-700nm
        self.output_dir = Path("spectral_csv_data")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create wavelength column headers
        self.pl_columns = [f"PL_{wl:.1f}nm" for wl in self.wavelength_range]
        self.abs_columns = [f"ABS_{wl:.1f}nm" for wl in self.wavelength_range]
        self.combined_columns = [f"SPEC_{wl:.1f}nm" for wl in self.wavelength_range]
    
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
        
        # 4. Normalize spectra for ML training
        pl_intensity = self.normalize_spectrum(pl_intensity)
        absorption = self.normalize_spectrum(absorption)
        
        return wavelengths, pl_intensity, absorption
    
    def generate_pl_spectrum(self, wavelengths, emission_peak, fwhm, plqy):
        """Generate photoluminescence spectrum"""
        
        # Main emission peak (Gaussian)
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
        main_peak = plqy * np.exp(-0.5 * ((wavelengths - emission_peak) / sigma) ** 2)
        
        # Add spectral features based on quality
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
    
    def normalize_spectrum(self, spectrum):
        """Normalize spectrum for ML training"""
        # Min-max normalization to [0, 1]
        spectrum_min = np.min(spectrum)
        spectrum_max = np.max(spectrum)
        
        if spectrum_max > spectrum_min:
            normalized = (spectrum - spectrum_min) / (spectrum_max - spectrum_min)
        else:
            normalized = spectrum
            
        return normalized
    
    def create_combined_spectral_features(self, pl_intensity, absorption):
        """Create combined spectral features for ML"""
        
        # Method 1: Concatenate PL and Absorption
        combined_simple = np.concatenate([pl_intensity, absorption])
        
        # Method 2: Ratio features (PL/Absorption where meaningful)
        ratio_features = np.divide(pl_intensity, absorption + 1e-8)  # Avoid division by zero
        
        # Method 3: Difference features
        # Pad to same length if needed
        min_len = min(len(pl_intensity), len(absorption))
        diff_features = pl_intensity[:min_len] - absorption[:min_len]
        
        return combined_simple, ratio_features, diff_features
    
    def extract_spectral_features(self, wavelengths, pl_intensity, absorption):
        """Extract key spectral features as additional columns"""
        
        features = {}
        
        # PL Features
        pl_peak_idx = np.argmax(pl_intensity)
        features['pl_peak_wavelength'] = wavelengths[pl_peak_idx]
        features['pl_peak_intensity'] = pl_intensity[pl_peak_idx]
        features['pl_integrated_intensity'] = np.trapz(pl_intensity, wavelengths)
        
        # Calculate FWHM
        half_max = features['pl_peak_intensity'] / 2
        indices = np.where(pl_intensity >= half_max)[0]
        if len(indices) > 1:
            features['pl_fwhm_calculated'] = wavelengths[indices[-1]] - wavelengths[indices[0]]
        else:
            features['pl_fwhm_calculated'] = 0
        
        # Absorption Features
        abs_edge_idx = np.where(absorption > 0.5 * np.max(absorption))[0]
        if len(abs_edge_idx) > 0:
            features['abs_edge_wavelength'] = wavelengths[abs_edge_idx[0]]
        else:
            features['abs_edge_wavelength'] = 0
            
        features['abs_peak_intensity'] = np.max(absorption)
        features['abs_integrated'] = np.trapz(absorption, wavelengths)
        
        # Stokes shift
        features['stokes_shift'] = features['pl_peak_wavelength'] - features['abs_edge_wavelength']
        
        # Spectral quality indicators
        features['pl_symmetry'] = self.calculate_peak_symmetry(wavelengths, pl_intensity, pl_peak_idx)
        features['baseline_level'] = np.mean(pl_intensity[:10])  # First 10 points as baseline
        
        return features
    
    def calculate_peak_symmetry(self, wavelengths, intensity, peak_idx):
        """Calculate peak symmetry as a quality indicator"""
        
        if peak_idx == 0 or peak_idx == len(intensity) - 1:
            return 0
        
        # Get left and right halves
        left_half = intensity[:peak_idx]
        right_half = intensity[peak_idx+1:]
        
        # Calculate asymmetry
        if len(left_half) > 0 and len(right_half) > 0:
            left_area = np.trapz(left_half, wavelengths[:peak_idx])
            right_area = np.trapz(right_half, wavelengths[peak_idx+1:])
            symmetry = 1 - abs(left_area - right_area) / (left_area + right_area + 1e-8)
        else:
            symmetry = 0
            
        return symmetry
    
    def process_csv_to_spectral_data(self, csv_file):
        """Convert entire CSV dataset to spectral CSV data"""
        
        print(f"Loading synthesis data from {csv_file}")
        df = pd.read_csv(csv_file)
        
        print(f"Generating spectral data for {len(df)} samples...")
        print(f"Wavelength range: {self.wavelength_range[0]:.1f} - {self.wavelength_range[-1]:.1f} nm")
        print(f"Spectral resolution: {self.wavelength_points} points")
        
        # Initialize result dataframes
        all_spectral_data = []
        
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
            
            # Extract spectral features
            spectral_features = self.extract_spectral_features(wavelengths, pl_intensity, absorption)
            
            # Create row data
            row_data = {
                # Original synthesis parameters
                'sample_id': idx,
                'cs_flow_rate': row['cs_flow_rate'],
                'pb_flow_rate': row['pb_flow_rate'],
                'temperature': row['temperature'],
                'residence_time': row['residence_time'],
                'plqy': row['plqy'],
                'emission_peak': row['emission_peak'],
                'fwhm': row['fwhm'],
                'quality_class': row['quality_class'],
            }
            
            # Add extracted spectral features
            row_data.update(spectral_features)
            
            # Add PL spectrum data
            for i, intensity in enumerate(pl_intensity):
                row_data[self.pl_columns[i]] = intensity
            
            # Add Absorption spectrum data
            for i, absorbance in enumerate(absorption):
                row_data[self.abs_columns[i]] = absorbance
            
            all_spectral_data.append(row_data)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)} spectra...")
        
        # Create final dataframe
        spectral_df = pd.DataFrame(all_spectral_data)
        
        # Save complete spectral dataset
        complete_file = self.output_dir / "complete_spectral_data.csv"
        spectral_df.to_csv(complete_file, index=False)
        
        # Create separate files for different use cases
        self.create_specialized_datasets(spectral_df)
        
        print(f"\n✅ Successfully created spectral CSV datasets!")
        print(f"📁 Data saved in: {self.output_dir}")
        print(f"📊 Total features per sample: {len(spectral_df.columns)}")
        print(f"📈 Spectral data points: {len(self.pl_columns) + len(self.abs_columns)}")
        
        return spectral_df
    
    def create_specialized_datasets(self, spectral_df):
        """Create specialized datasets for different ML approaches"""
        
        # 1. PL-only dataset (for PL analysis)
        pl_features = ['sample_id', 'cs_flow_rate', 'pb_flow_rate', 'temperature', 'residence_time', 
                      'plqy', 'emission_peak', 'fwhm', 'quality_class'] + self.pl_columns
        pl_df = spectral_df[pl_features]
        pl_df.to_csv(self.output_dir / "pl_spectral_data.csv", index=False)
        
        # 2. Absorption-only dataset
        abs_features = ['sample_id', 'cs_flow_rate', 'pb_flow_rate', 'temperature', 'residence_time',
                       'plqy', 'emission_peak', 'fwhm', 'quality_class'] + self.abs_columns
        abs_df = spectral_df[abs_features]
        abs_df.to_csv(self.output_dir / "absorption_spectral_data.csv", index=False)
        
        # 3. Feature-engineering dataset (extracted features only)
        feature_cols = [col for col in spectral_df.columns if not any(
            col.startswith(prefix) for prefix in ['PL_', 'ABS_'])]
        features_df = spectral_df[feature_cols]
        features_df.to_csv(self.output_dir / "extracted_features.csv", index=False)
        
        # 4. X/Y formatted datasets
        self.create_xy_datasets(spectral_df)
        
        # 5. Quality-separated datasets
        self.create_quality_datasets(spectral_df)
        
        print(f"📂 Created specialized datasets:")
        print(f"   • complete_spectral_data.csv    ({spectral_df.shape[0]} × {spectral_df.shape[1]})")
        print(f"   • pl_spectral_data.csv          ({pl_df.shape[0]} × {pl_df.shape[1]})")
        print(f"   • absorption_spectral_data.csv  ({abs_df.shape[0]} × {abs_df.shape[1]})")
        print(f"   • extracted_features.csv        ({features_df.shape[0]} × {features_df.shape[1]})")
    
    def create_xy_datasets(self, spectral_df):
        """Create X/Y formatted datasets for ML training"""
        
        # X features: synthesis parameters + spectral data
        x_synthesis = ['cs_flow_rate', 'pb_flow_rate', 'temperature', 'residence_time']
        x_spectral = self.pl_columns + self.abs_columns
        x_features = x_synthesis + x_spectral
        
        # Y targets
        y_targets = ['plqy', 'emission_peak', 'fwhm', 'quality_class']
        
        # Create X dataset
        X_df = spectral_df[x_features].copy()
        X_df.to_csv(self.output_dir / "X_spectral_features.csv", index=False)
        
        # Create Y dataset  
        Y_df = spectral_df[y_targets].copy()
        # Encode quality class
        quality_map = {'poor': 0, 'fair': 1, 'good': 2, 'excellent': 3}
        Y_df['quality_class_encoded'] = Y_df['quality_class'].map(quality_map)
        Y_df.to_csv(self.output_dir / "Y_spectral_targets.csv", index=False)
        
        print(f"   • X_spectral_features.csv       ({X_df.shape[0]} × {X_df.shape[1]})")
        print(f"   • Y_spectral_targets.csv        ({Y_df.shape[0]} × {Y_df.shape[1]})")
    
    def create_quality_datasets(self, spectral_df):
        """Create separate datasets by quality class"""
        
        quality_dir = self.output_dir / "by_quality"
        quality_dir.mkdir(exist_ok=True)
        
        for quality in spectral_df['quality_class'].unique():
            quality_subset = spectral_df[spectral_df['quality_class'] == quality]
            quality_subset.to_csv(quality_dir / f"{quality}_spectral_data.csv", index=False)
            print(f"   • by_quality/{quality}_spectral_data.csv ({len(quality_subset)} samples)")
    
    def create_wavelength_reference(self):
        """Create wavelength reference file"""
        
        wavelength_ref = pd.DataFrame({
            'wavelength_nm': self.wavelength_range,
            'pl_column': self.pl_columns,
            'abs_column': self.abs_columns,
            'energy_ev': 1240 / self.wavelength_range,  # Convert to eV
            'color_region': ['UV' if w < 400 else 'Violet' if w < 450 else 'Blue' if w < 495 else 
                           'Green' if w < 570 else 'Yellow' if w < 590 else 'Orange' if w < 620 else 'Red' 
                           for w in self.wavelength_range]
        })
        
        wavelength_ref.to_csv(self.output_dir / "wavelength_reference.csv", index=False)
        print(f"   • wavelength_reference.csv      (wavelength mapping)")

# Main execution
if __name__ == "__main__":
    # Create generator with desired spectral resolution
    generator = SpectralCSVGenerator(wavelength_points=256)  # Adjust as needed
    
    # Convert your synthesis CSV to spectral CSV data
    spectral_df = generator.process_csv_to_spectral_data("my_training_data_2000.csv")
    
    # Create wavelength reference
    generator.create_wavelength_reference()
    
    print(f"\n🎉 Spectral CSV datasets ready for ML training!")
    print(f"📊 {len(spectral_df)} samples with full spectral data")
    print(f"🔬 Each sample contains {generator.wavelength_points} PL + {generator.wavelength_points} absorption points")
    print(f"🧠 Ready for various ML approaches: CNN, Random Forest, Neural Networks")
    print(f"📈 Spectral features: peak positions, FWHM, intensities, symmetry, etc.")