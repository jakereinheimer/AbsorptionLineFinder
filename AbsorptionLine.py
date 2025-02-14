import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.widgets import Slider
import os
from scipy.signal import find_peaks
from scipy.integrate import simpson
import pickle


# Constants
c = 2.9979e10  # Speed of light in cm/s
c_kms = 2.9979e5  # Speed of light in km/s

#helper functions
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Open the file in binary write mode
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)  # Pickle the object and write to file

class AbsorptionLine:
    def __init__(self, wavelength, flux, errors):

        self.wavelength=wavelength
        self.flux = flux
        self.errors = errors

        

        self.start=self.wavelength[0]
        self.stop=self.wavelength[-1]

        self.values = list(zip(self.wavelength,self.flux))  # This will hold tuples of (wavelength, flux)

        if np.any(self.errors==0):
            self.zero_error=True
        else:
            self.zero_error=False
        
        zero_flux_indices = [i for i, flux in enumerate(self.flux) if flux == 0.]  # Adjust the threshold as needed
        self.number_of_zeros_flux=len(zero_flux_indices)

        zero_error_indices = [i for i, error in enumerate(self.errors) if error == 0.]  # Adjust the threshold as needed
        self.number_of_zeros_error=len(zero_error_indices)

        if self.number_of_zeros_flux>=2:
            self.peak = ((self.wavelength[0]+self.wavelength[-1])/2)
        else:
            self.peak = self.wavelength[np.argmin(self.flux)]

        if self.number_of_zeros_flux>2:
            self.is_saturated=True
        else:
            self.is_saturated=False

        #self.EW=self.calculate_equivalent_width()
        self.EW=self.equivalent_width()

        self.center=int((self.wavelength[0]+self.wavelength[1])/2)

        self.suspected_line_loc=1
        
    
    def actual_equivalent_width(self):

        self.suspected_z=(self.peak-self.suspected_line_loc)/self.suspected_line_loc
        
        # Adjust wavelengths for redshift
        z_adjusted_wavelength = (1 / (1 + self.suspected_z)) * self.wavelength
        
        # Differences in the adjusted wavelengths
        delta_lambda = np.diff(z_adjusted_wavelength)
        
        # Calculate equivalent width
        self.actual_ew = np.sum((1 - self.flux[:-1]) * delta_lambda)
        
        # Calculate the error in equivalent width
        ew_errors = delta_lambda * self.errors[:-1]  # Assuming flux_errors array aligns with self.flux
        self.actual_ew_error = np.sqrt(np.sum(ew_errors**2))
        
        return self.actual_ew, self.actual_ew_error
    
    def equivalent_width(self):

        ew = simpson(1 - self.flux, x=self.wavelength) 

        return ew
    
    
    def find_N(self, z, f, lambda_0):
        
        # Adjust wavelengths for redshift
        z_adjusted_wavelength = (1 / (1 + z)) * self.wavelength

        # Calculate the optical depth, ensuring no zero or negative flux values
        tau_lambda = -np.log(np.clip(self.flux, a_min=1e-10, a_max=None))
        sigma_tau_lambda = self.errors / np.clip(self.flux, a_min=1e-10, a_max=None)

        # Calculate differences in the adjusted wavelengths
        delta_lambda = np.diff(z_adjusted_wavelength/10**8)

        # Approximate the integral using a simple sum
        integral_tau = np.sum(tau_lambda[:-1] * delta_lambda)
        sigma_integral_tau = np.sqrt(np.sum((sigma_tau_lambda[:-1] * delta_lambda)**2))

        # Constants in CGS
        m_e = 9.109 * 10**(-28)  # electron mass in grams
        c = 2.998 * 10**10     # speed of light in cm/s
        e = 4.8 * 10**(-10)

        # Assuming lambda_0 is provided in Angstroms (A), and needs to be in centimeters:
        # 1 Angstrom = 10^-8 cm
        lambda_0_cm = lambda_0 / 10**8  # converting from Angstroms to centimeters

        # Calculate N
        factor = (m_e * c**2) / (np.pi * e**2) / (f * lambda_0_cm**2)
        self.N = factor * integral_tau
        self.sigma_N = factor * sigma_integral_tau

        self.log_N=np.log10(self.N)
        self.log_sigma_N= self.sigma_N / self.N

        return self.N,self.sigma_N 


        
    def plot(self,name):
        
        plt.plot(self.wavelength,self.flux)
        plt.axvline(self.peak)
        plt.title(f"Equivalent width:{self.EW}")
        plt.savefig("test_plot/"+name+".png")
        plt.clf()

    def export(self, name):
        # Split the name to separate the directory path and the file name
        directory, filename = os.path.split(name)
        
        # Create the directory if it does not exist
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Create a DataFrame using a dictionary to ensure correct alignment of columns
        df = pd.DataFrame({
            'Flux': self.flux,
            'Wavelength': self.wavelength,
            'Error': self.errors
        })
        
        # Save the DataFrame to a CSV file
        df.to_csv(f"{name}.csv", index=False)

    def save(self,name):

        # Split the name to separate the directory path and the file name
        directory, filename = os.path.split(name)
        
        # Create the directory if it does not exist
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        save_object(self,name)
