import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

class SlabWaveguideProfile:
    def __init__(self, nc, nf, ns, wavelength):
        self.nc = nc
        self.nf = nf
        self.ns = ns
        self.wavelength = wavelength
        self.k0 = 2 * np.pi / wavelength
        
        # Calculate numerical aperture parameters
        self.na_core_sub = np.sqrt(nf**2 - ns**2)

    def calculate_v_parameter(self, d):
        """Calculate V = k0 * d * sqrt(nf² - ns²)"""
        return self.k0 * d * self.na_core_sub
    
    def te_dispersion_equation(self, b, V, m):
        """TE mode dispersion equation: 2V*sqrt(1-b) = m*π + arctan(...) + arctan(...)"""
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
        """Find effective index for given V and mode number m"""
        if V <= 0:
            return None, None

        # Search for b parameter between 0 and 1
        b_min = 1e-8
        b_max = 1 - 1e-8
        
        try:
            # Check if a solution exists by evaluating at endpoints
            f_min = self.te_dispersion_equation(b_min, V, m)
            f_max = self.te_dispersion_equation(b_max, V, m)
            
            # If no sign change, no solution exists
            if f_min * f_max > 0:
                return None, None
            
            # Find the root
            b_solution = brentq(self.te_dispersion_equation, b_min, b_max, args=(V, m))
            
            # Convert b back to effective index
            neff = np.sqrt(self.ns**2 + b_solution * (self.nf**2 - self.ns**2))
            
            return neff, b_solution
        except:
            return None, None
    
    def calculate_field_parameters(self, neff, d):
        """Calculate field parameters kappa, gamma, sigma"""
        beta = neff * self.k0
        
        kappa_sq = self.nf**2 * self.k0**2 - beta**2
        gamma_sq = beta**2 - self.nc**2 * self.k0**2
        sigma_sq = beta**2 - self.ns**2 * self.k0**2
        
        if kappa_sq < 0 or gamma_sq < 0 or sigma_sq < 0:
            return None, None, None
            
        kappa = np.sqrt(kappa_sq)
        gamma = np.sqrt(gamma_sq)
        sigma = np.sqrt(sigma_sq)
        
        return kappa, gamma, sigma
    
    def calculate_field_profile(self, d, neff, position_range=None):
        """Calculate optical field profile for TE₀ mode"""
        
        if position_range is None:
            position_range = (-3*d, 3*d)
        
        # Calculate field parameters
        kappa, gamma, sigma = self.calculate_field_parameters(neff, d)
        
        if kappa is None:
            return None, None
        
        # Create position array
        x = np.linspace(position_range[0], position_range[1], 1000)
        
        # Calculate electric field Ey(x) for TE₀ mode (m=0, even mode)
        Ey = np.zeros_like(x)
        
        for i, pos in enumerate(x):
            if pos < -d:  # Substrate region (x < -d)
                # Even mode: A_sub = cos(kappa * d)
                A_sub = np.cos(kappa * d)
                Ey[i] = A_sub * np.exp(sigma * (pos + d))
                
            elif pos > d:  # Cladding region (x > d)
                # Even mode: A_clad = cos(kappa * d)
                A_clad = np.cos(kappa * d)
                Ey[i] = A_clad * np.exp(-gamma * (pos - d))
                
            else:  # Core region (-d <= x <= d)
                # Even mode: cos(kappa * x)
                Ey[i] = np.cos(kappa * pos)
        
        # Normalize field
        Ey = Ey / np.max(np.abs(Ey))
        
        return x, Ey
    
    def plot_te0_field(self, core_thickness_um):
        """Plot TE₀ optical field profile for single-mode operation"""
        
        d = core_thickness_um * 1e-6 / 2  # Convert to half-thickness in meters
        V = self.calculate_v_parameter(d)

        # Find TE₀ mode (m=0)
        neff, b = self.find_effective_index(V, 0)
        
        if neff is None or not (self.ns < neff < self.nf):
            print("TE₀ mode cannot propagate at this thickness!")
            return
        
        print(f"TE₀ effective index N = {neff:.5f}")
        print(f"TE₀ b parameter = {b:.3f}")
        
        # Calculate field profile
        x, Ey = self.calculate_field_profile(d, neff)
        
        if x is None:
            print("Error calculating field profile!")
            return
        
        # Create single plot for TE₀
        plt.figure(figsize=(12, 8))
        
        # Convert position to micrometers
        x_um = x * 1e6
        
        # Plot field profile
        plt.plot(x_um, Ey, color='blue', linewidth=4, label='TE₀ Electric Field E(x)')
        plt.fill_between(x_um, 0, Ey, alpha=0.3, color='blue', label='Field magnitude')
        
        # Mark core boundaries
        plt.axvline(-d*1e6, color='black', linestyle='--', linewidth=3, alpha=0.8, label='Core boundaries')
        plt.axvline(d*1e6, color='black', linestyle='--', linewidth=3, alpha=0.8)

        
        # Add horizontal line at zero
        plt.axhline(0, color='black', linestyle='-', alpha=0.4, linewidth=1)
        

        
        # Formatting
        plt.xlabel('Position x (μm)', fontsize=16, fontweight='bold')
        plt.ylabel('Electric Field E (a.u.)', fontsize=16, fontweight='bold')
        plt.title(f'TE₀ Mode Optical Field Profile\nSingle-Mode Operation (2d = {core_thickness_um} μm)', 
                 fontsize=18, fontweight='bold')
        plt.legend(fontsize=14, loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Set axis limits
        plt.xlim(-2.5*d*1e6, 2.5*d*1e6)
        plt.ylim(-1.2, 1.2)
        
        # Add parameter information box
        textstr = f'Waveguide Parameters:\nλ = {self.wavelength*1e6:.2f} μm\nV = {V:.3f}\nN = {neff:.5f}\nb = {b:.3f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=props, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Summary
        print(f"\nTE₀ Field Profile Summary:")
        print(f"• Single symmetric peak at center (x = 0)")
        print(f"• Exponential decay in cladding and substrate")
        print(f"• Field confined primarily in core region")
        print(f"• Effective index N = {neff:.5f} (between ns and nf)")

# Problem parameters
nc = 1.43    # upper cladding
nf = 1.45    # core layer  
ns = 1.44    # substrate
wavelength = 1.55e-6  # 1.55 μm

# Create waveguide object
waveguide = SlabWaveguideProfile(nc, nf, ns, wavelength)

# Plot TE₀ field profile for single-mode operation
total_core_thickness = 5.69  # μm (2d - your specific value)

print("=" * 60)
print("SINGLE-MODE TE₀ OPTICAL FIELD PROFILE")
print("=" * 60)
waveguide.plot_te0_field(core_thickness_um=total_core_thickness)