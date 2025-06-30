import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

class SlabWaveguide:
    def __init__(self, nc, nf, ns, wavelength):
        self.nc = nc
        self.nf = nf
        self.ns = ns
        self.wavelength = wavelength
        self.k0 = 2 * np.pi / wavelength
        
        # Calculate numerical aperture parameters
        self.na_core_sub = np.sqrt(nf**2 - ns**2)
        self.na_core_clad = np.sqrt(nf**2 - nc**2)
        

    def calculate_v_parameter(self, d):
        """Calculate V = k0 * d * sqrt(nf² - ns²)"""
        return self.k0 * d * self.na_core_sub
    
    def te_dispersion_equation(self, b, V, m):
        if b <= 0 or b >= 1:
            return float('inf')
        
        # Asymmetry parameter
        a = (self.ns**2 - self.nc**2) / (self.nf**2 - self.ns**2)
        
        # Left side of dispersion equation
        lhs = 2*V * np.sqrt(1 - b)
        
        # Right side of dispersion equation
        term1 = m * np.pi
        term2 = np.arctan(np.sqrt(b / (1 - b)))
        term3 = np.arctan(np.sqrt((b + a) / (1 - b)))
        rhs = term1 + term2 + term3
        
        return lhs - rhs
    
    def find_effective_index(self, V, m):
        if V <= 0:
            return None

        # Search for b parameter between 0 and 1
        b_min = 1e-6
        b_max = 1 - 1e-6
        
        # Check if a solution exists by evaluating at endpoints
        f_min = self.te_dispersion_equation(b_min, V, m)
        f_max = self.te_dispersion_equation(b_max, V, m)
        
        # If no sign change, no solution exists
        if f_min * f_max > 0:
            return None
        
        # Find the root
        b_solution = brentq(self.te_dispersion_equation, b_min, b_max, args=(V, m))
        
        # Convert b back to effective index
        neff = np.sqrt(self.ns**2 + b_solution * (self.nf**2 - self.ns**2))
        
        return neff
    
    def calculate_cutoff_v(self, m):
        """Calculate approximate cutoff V for mode m"""
        if m == 0:
            return 0.1  # TE0 cutoff is slightly above 0
        else:   
            # pi/4 being an approximation of arctan(sqrt(a))
            return (m * np.pi + np.pi/4)/2  # Higher order modes
    
    def plot_dispersion_curve(self, thickness_range=(1, 20), num_points=300):
        print("Calculating TE mode dispersion curves...")
        
        # Create thickness array
        thickness_values = np.linspace(thickness_range[0], thickness_range[1], num_points) * 1e-6
        
        # Store results
        modes_data = {}
        
        for i, thickness_2d in enumerate(thickness_values):
            if i % 50 == 0:
                print(f"Progress: {i/len(thickness_values)*100:.1f}%")
            
            d = thickness_2d / 2  # Half thickness
            V = self.calculate_v_parameter(d)
            
            # Find first 4 TE modes
            for m in range(4):  # TE0, TE1, TE2, TE3
                mode_name = f"TE{m}"
                
                # Check if this mode can exist at this V
                V_cutoff = self.calculate_cutoff_v(m)
                
                if V > V_cutoff:
                    neff = self.find_effective_index(V, m)
                    
                    if neff is not None and self.ns < neff < self.nf:
                        if mode_name not in modes_data:
                            modes_data[mode_name] = {'thickness': [], 'neff': [], 'V': []}
                        
                        modes_data[mode_name]['thickness'].append(thickness_2d * 1e6)
                        modes_data[mode_name]['neff'].append(neff)
                        modes_data[mode_name]['V'].append(V)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange']
        mode_labels = ['TE₀ (m=0)', 'TE₁ (m=1)', 'TE₂ (m=2)', 'TE₃ (m=3)']
        
        for i, mode_name in enumerate(['TE0', 'TE1', 'TE2', 'TE3']):
            if mode_name in modes_data and len(modes_data[mode_name]['thickness']) > 10:
                plt.plot(modes_data[mode_name]['thickness'], modes_data[mode_name]['neff'], 
                        color=colors[i], linewidth=3, label=mode_labels[i])
                print(f"Plotted {mode_name}: {len(modes_data[mode_name]['thickness'])} points")
            else:
                print(f"{mode_name}: No data")
        
        # Add reference lines
        plt.axhline(y=self.nc, color='gray', linestyle='--', alpha=0.7, 
                   linewidth=2, label=f'Cladding (nc = {self.nc})')
        plt.axhline(y=self.ns, color='brown', linestyle=':', alpha=0.7, 
                   linewidth=2, label=f'Substrate (ns = {self.ns})')
        plt.axhline(y=self.nf, color='black', linestyle='-', alpha=0.8, 
                   linewidth=2, label=f'Core (nf = {self.nf})')
        
        # Formatting
        plt.xlabel('Core Thickness 2d (μm)', fontsize=14, fontweight='bold')
        plt.ylabel('Effective Index N', fontsize=14, fontweight='bold')
        plt.title('TE Mode Dispersion Curves\nEffective Index vs Core Thickness', fontsize=16, fontweight='bold')
        plt.legend(fontsize=11, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xlim(thickness_range)
        plt.ylim(self.ns - 0.0005, self.nf + 0.0005)  # Y-axis starts just below substrate index
        
        plt.tight_layout()
        plt.show()
        
        # Print results
        print("\n" + "="*50)
        print("DISPERSION CURVE RESULTS")
        print("="*50)
        
        for mode_name in ['TE0', 'TE1', 'TE2', 'TE3']:
            if mode_name in modes_data and len(modes_data[mode_name]['thickness']) > 0:
                data = modes_data[mode_name]
                min_thickness = min(data['thickness'])
                min_idx = data['thickness'].index(min_thickness)
                cutoff_neff = data['neff'][min_idx]
                cutoff_V = data['V'][min_idx]
                
                print(f"\n{mode_name} Mode:")
                print(f"  Cutoff thickness: {min_thickness:.2f} μm")
                print(f"  Cutoff V parameter: {cutoff_V:.3f}")
                print(f"  Effective index at cutoff: {cutoff_neff:.5f}")
                
                # Find neff at 10 μm if exists
                if max(data['thickness']) >= 10.0:
                    neff_10um = np.interp(10.0, data['thickness'], data['neff'])
                    V_10um = np.interp(10.0, data['thickness'], data['V'])
                    print(f"  At 2d = 10 μm: N = {neff_10um:.5f}, V = {V_10um:.3f}")
        
        return modes_data

# Problem parameters
nc = 1.43    # upper cladding
nf = 1.45    # core layer  
ns = 1.44    # substrate
wavelength = 1.55e-6  # 1.55 μm

# Create waveguide and plot
waveguide = SlabWaveguide(nc, nf, ns, wavelength)
modes_data = waveguide.plot_dispersion_curve(thickness_range=(1,70), num_points=500)

print(f"\nComplete! Graph shows effective index vs core thickness for TE modes.")
print(f"Each curve represents a different mode (m = 0, 1, 2, 3)")