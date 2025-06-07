# ml-models/generate_large_dataset.py
import numpy as np
import pandas as pd
from pathlib import Path

class CsPbBr3DataGenerator:
    """Generate realistic CsPbBr3 synthesis data based on physical principles"""
    
    def __init__(self):
        # Physical constants and optimal conditions
        self.optimal_conditions = {
            'cs_pb_ratio': 1.2,         # Optimal Cs:Pb stoichiometry
            'temperature': 150,         # Optimal temperature (°C)
            'residence_time': 120,      # Optimal residence time (s)
            'total_flow': 2.0          # Optimal total flow rate (mL/min)
        }
        
        # Realistic parameter ranges (narrowed to avoid obviously bad conditions)
        self.parameter_ranges = {
            'cs_flow_rate': (0.5, 2.2),      # mL/min - realistic range
            'pb_flow_rate': (0.6, 1.8),      # mL/min - realistic range
            'temperature': (100, 200),        # °C - practical synthesis range
            'residence_time': (60, 250)       # seconds - practical range
        }
        
        # More realistic quality thresholds
        self.quality_thresholds = {
            'excellent': 0.75,  # PLQY > 0.75
            'good': 0.60,       # PLQY 0.60-0.75
            'fair': 0.40,       # PLQY 0.40-0.60
            'poor': 0.0         # PLQY < 0.40
        }
    
    def calculate_cs_pb_ratio_effect(self, cs_flow, pb_flow):
        """Calculate effect of Cs:Pb stoichiometry on crystal quality"""
        ratio = cs_flow / pb_flow
        optimal_ratio = self.optimal_conditions['cs_pb_ratio']
        
        # More forgiving Gaussian response around optimal ratio
        ratio_effect = 0.5 + 0.5 * np.exp(-0.5 * ((ratio - optimal_ratio) / 0.4) ** 2)
        
        # Less harsh penalties for off-stoichiometry
        if ratio < 0.7:  # Pb-rich conditions
            ratio_effect *= 0.6  # 40% penalty (was 70%)
        elif ratio > 2.0:  # Cs-rich conditions  
            ratio_effect *= 0.7  # 30% penalty (was 40%)
            
        return ratio_effect
    
    def calculate_temperature_effect(self, temperature):
        """Calculate temperature effects on nucleation and growth"""
        optimal_temp = self.optimal_conditions['temperature']
        
        # More realistic temperature response
        if temperature < 110:
            # Gradual decrease for low temperatures
            temp_effect = 0.5 + 0.4 * (temperature - 100) / 20
        elif temperature > 180:
            # Gradual decrease for high temperatures  
            temp_effect = 0.85 - 0.3 * (temperature - 180) / 30
        else:
            # Broad optimal window
            temp_effect = 0.75 + 0.25 * np.exp(-0.5 * ((temperature - optimal_temp) / 30) ** 2)
            
        return np.clip(temp_effect, 0.4, 1.0)
    
    def calculate_residence_time_effect(self, residence_time, temperature):
        """Calculate residence time effects - depends on temperature"""
        optimal_time = self.optimal_conditions['residence_time']
        
        # Temperature affects required residence time (but less dramatically)
        temp_factor = 1 + (temperature - 150) / 100  # More gradual effect
        effective_optimal_time = optimal_time / temp_factor
        
        if residence_time < effective_optimal_time * 0.5:
            # Incomplete reaction - more gradual penalty
            time_effect = 0.4 + 0.4 * (residence_time / (effective_optimal_time * 0.5))
        elif residence_time > effective_optimal_time * 2.5:
            # Degradation - less harsh penalty
            time_effect = 0.8 - 0.2 * ((residence_time - effective_optimal_time * 2.5) / 100)
        else:
            # Normal range - broader optimal window
            time_effect = 0.6 + 0.4 * np.exp(-0.5 * ((residence_time - effective_optimal_time) / 80) ** 2)
            
        return np.clip(time_effect, 0.3, 1.0)
    
    def calculate_flow_rate_effect(self, cs_flow, pb_flow):
        """Calculate flow rate effects on mixing and residence time"""
        total_flow = cs_flow + pb_flow
        optimal_total = self.optimal_conditions['total_flow']
        
        if total_flow < 0.8:
            # Very slow flow - poor mixing
            flow_effect = 0.6 + 0.3 * total_flow / 0.8
        elif total_flow > 4.0:
            # Very fast flow - insufficient residence time
            flow_effect = 0.8 * np.exp(-(total_flow - 4.0) / 2.0) 
        else:
            # Normal range
            flow_effect = 0.85 + 0.15 * np.exp(-0.5 * ((total_flow - optimal_total) / 1.0) ** 2)
            
        return np.clip(flow_effect, 0.3, 1.0)
    
    def calculate_plqy(self, cs_flow, pb_flow, temperature, residence_time):
        """Calculate PLQY based on all synthesis parameters"""
        
        # Individual effects
        ratio_effect = self.calculate_cs_pb_ratio_effect(cs_flow, pb_flow)
        temp_effect = self.calculate_temperature_effect(temperature)
        time_effect = self.calculate_residence_time_effect(residence_time, temperature)
        flow_effect = self.calculate_flow_rate_effect(cs_flow, pb_flow)
        
        # Combined effect with more realistic base
        base_plqy = 0.35 + 0.55 * (ratio_effect * temp_effect * time_effect * flow_effect)
        
        # Add synergistic effects for realistic conditions
        if 0.9 <= cs_flow/pb_flow <= 1.6 and 130 <= temperature <= 170:
            base_plqy *= 1.15  # Realistic synergy bonus
            
        # Add some variability but less harsh
        noise = np.random.normal(0, 0.08)  # 8% random variation
        plqy = base_plqy + noise
        
        return np.clip(plqy, 0.15, 0.92)
    
    def calculate_emission_peak(self, cs_flow, pb_flow, temperature, plqy):
        """Calculate emission peak wavelength"""
        base_peak = 515.0  # CsPbBr3 characteristic peak
        
        # Size effects (temperature and ratio dependent)
        cs_pb_ratio = cs_flow / pb_flow
        
        # Higher temperature = larger particles = slight red shift
        temp_shift = (temperature - 150) * 0.02
        
        # Cs-rich conditions = slight blue shift (smaller particles)  
        ratio_shift = -(cs_pb_ratio - 1.2) * 2.0
        
        # Poor quality crystals = broader size distribution = red shift
        quality_shift = (0.75 - plqy) * 8.0
        
        # Combine effects
        emission_peak = base_peak + temp_shift + ratio_shift + quality_shift
        
        # Add noise
        noise = np.random.normal(0, 1.5)
        emission_peak += noise
        
        return np.clip(emission_peak, 505, 530)
    
    def calculate_fwhm(self, plqy, temperature, cs_flow, pb_flow):
        """Calculate Full Width Half Maximum"""
        base_fwhm = 22.0  # Base FWHM for good crystals
        
        # Higher PLQY = narrower peaks (better size distribution)
        quality_factor = (0.85 - plqy) * 30
        
        # Temperature effects on size distribution
        if temperature < 100 or temperature > 200:
            temp_broadening = 8.0
        else:
            temp_broadening = abs(temperature - 160) * 0.1
            
        # Stoichiometry effects
        ratio = cs_flow / pb_flow
        ratio_broadening = abs(ratio - 1.2) * 10
        
        fwhm = base_fwhm + quality_factor + temp_broadening + ratio_broadening
        
        # Add noise
        noise = np.random.normal(0, 2.0)
        fwhm += noise
        
        return np.clip(fwhm, 12, 60)
    
    def assign_quality_class(self, plqy):
        """Assign quality class based on PLQY"""
        if plqy >= self.quality_thresholds['excellent']:
            return 'excellent'
        elif plqy >= self.quality_thresholds['good']:
            return 'good'
        elif plqy >= self.quality_thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def generate_systematic_samples(self, n_samples=1000):
        """Generate systematic exploration of parameter space with realistic focus"""
        samples = []
        
        # Focus on more realistic parameter ranges
        cs_flows = np.linspace(0.6, 2.0, 10)        # Narrower, realistic range
        pb_flows = np.linspace(0.7, 1.6, 8)         # Narrower, realistic range  
        temperatures = np.linspace(110, 190, 12)     # Practical synthesis range
        residence_times = np.linspace(70, 220, 10)   # Practical range
        
        count = 0
        for cs_flow in cs_flows:
            for pb_flow in pb_flows:
                for temp in temperatures:
                    for res_time in residence_times:
                        if count >= n_samples:
                            break
                            
                        # Calculate properties
                        plqy = self.calculate_plqy(cs_flow, pb_flow, temp, res_time)
                        emission_peak = self.calculate_emission_peak(cs_flow, pb_flow, temp, plqy)
                        fwhm = self.calculate_fwhm(plqy, temp, cs_flow, pb_flow)
                        quality_class = self.assign_quality_class(plqy)
                        
                        samples.append({
                            'cs_flow_rate': round(cs_flow, 2),
                            'pb_flow_rate': round(pb_flow, 2),
                            'temperature': round(temp, 1),
                            'residence_time': round(res_time, 1),
                            'plqy': round(plqy, 4),
                            'emission_peak': round(emission_peak, 2),
                            'fwhm': round(fwhm, 2),
                            'quality_class': quality_class
                        })
                        count += 1
                        
                    if count >= n_samples:
                        break
                if count >= n_samples:
                    break
            if count >= n_samples:
                break
                
        return samples
    
    def generate_random_samples(self, n_samples=1000):
        """Generate random samples across parameter space"""
        samples = []
        
        for _ in range(n_samples):
            # Random parameter selection
            cs_flow = np.random.uniform(*self.parameter_ranges['cs_flow_rate'])
            pb_flow = np.random.uniform(*self.parameter_ranges['pb_flow_rate'])
            temperature = np.random.uniform(*self.parameter_ranges['temperature'])
            residence_time = np.random.uniform(*self.parameter_ranges['residence_time'])
            
            # Calculate properties
            plqy = self.calculate_plqy(cs_flow, pb_flow, temperature, residence_time)
            emission_peak = self.calculate_emission_peak(cs_flow, pb_flow, temperature, plqy)
            fwhm = self.calculate_fwhm(plqy, temperature, cs_flow, pb_flow)
            quality_class = self.assign_quality_class(plqy)
            
            samples.append({
                'cs_flow_rate': round(cs_flow, 2),
                'pb_flow_rate': round(pb_flow, 2), 
                'temperature': round(temperature, 1),
                'residence_time': round(residence_time, 1),
                'plqy': round(plqy, 4),
                'emission_peak': round(emission_peak, 2),
                'fwhm': round(fwhm, 2),
                'quality_class': quality_class
            })
            
        return samples
    
    def generate_dataset(self, total_samples=2000, systematic_ratio=0.6):
        """Generate complete dataset with mix of systematic and random sampling"""
        
        print(f"Generating {total_samples} samples...")
        print(f"Systematic samples: {int(total_samples * systematic_ratio)}")
        print(f"Random samples: {int(total_samples * (1 - systematic_ratio))}")
        
        # Generate systematic samples
        systematic_samples = self.generate_systematic_samples(
            int(total_samples * systematic_ratio)
        )
        
        # Generate random samples  
        random_samples = self.generate_random_samples(
            int(total_samples * (1 - systematic_ratio))
        )
        
        # Combine all samples
        all_samples = systematic_samples + random_samples
        
        # Create DataFrame
        df = pd.DataFrame(all_samples)
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Dataset generated successfully!")
        print(f"Total samples: {len(df)}")
        print(f"Quality distribution:")
        print(df['quality_class'].value_counts())
        print(f"\nPLQY statistics:")
        print(f"Mean PLQY: {df['plqy'].mean():.3f}")
        print(f"PLQY range: {df['plqy'].min():.3f} - {df['plqy'].max():.3f}")
        
        return df
    
    def save_dataset(self, df, filename='my_training_data_2000.csv'):
        """Save dataset to CSV"""
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        
        # Generate summary statistics
        print(f"\n{'='*50}")
        print("DATASET SUMMARY")
        print(f"{'='*50}")
        
        print(f"Parameter ranges:")
        for col in ['cs_flow_rate', 'pb_flow_rate', 'temperature', 'residence_time']:
            print(f"  {col}: {df[col].min():.2f} - {df[col].max():.2f}")
            
        print(f"\nProperty ranges:")
        for col in ['plqy', 'emission_peak', 'fwhm']:
            print(f"  {col}: {df[col].min():.3f} - {df[col].max():.3f}")
            
        print(f"\nQuality distribution:")
        quality_counts = df['quality_class'].value_counts()
        for quality, count in quality_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {quality}: {count} ({percentage:.1f}%)")

# Main execution
if __name__ == "__main__":
    # Create generator
    generator = CsPbBr3DataGenerator()
    
    # Generate 2000 samples
    dataset = generator.generate_dataset(total_samples=2000)
    
    # Save dataset
    generator.save_dataset(dataset, 'my_training_data_2000.csv')
    
    print(f"\n🎉 Successfully created 2000-sample training dataset!")
    print(f"📊 Ready for CNN training with much better performance expected!")