import emcee
import numpy as np
from scipy.special import wofz
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import corner
import pandas as pd
import random
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve

from essential_functions import read_parameter

e = 4.8032e-10 # electron charge in stat-coulumb
m_e = 9.10938e-28 # electron mass
c = 2.9979e10 # cm/s
c_As = 2.9979e18
c_kms = 2.9979e5
k = 1.38065e-16 # erg/K

#not used:
def interpolate_spectrum(wavelength, flux, error, original_resolution, target_resolution):
    c = 299792.458  # Speed of light in km/s
    delta_lambda_original = (original_resolution / c) * wavelength
    delta_lambda_target = (target_resolution / c) * wavelength

    # Create new wavelength grid based on the target resolution
    new_wavelength = np.arange(wavelength[0], wavelength[-1], np.median(delta_lambda_target))

    # Interpolation functions
    flux_interp = interp1d(wavelength, flux, kind='linear', fill_value="extrapolate")
    error_interp = interp1d(wavelength, error, kind='linear', fill_value="extrapolate")

    # Interpolate flux and error
    new_flux = flux_interp(new_wavelength)
    new_error = error_interp(new_wavelength)

    return new_wavelength, new_flux, new_error

def resample_spectrum(wavelength, flux, error, new_length):
    # Create new wavelength grid with the specified number of points
    new_wavelength = np.linspace(wavelength[0], wavelength[-1], new_length)

    # Interpolation functions
    flux_interp = interp1d(wavelength, flux, kind='linear', fill_value="extrapolate")
    error_interp = interp1d(wavelength, error, kind='linear', fill_value="extrapolate")

    # Interpolate flux and error to the new wavelength grid
    new_flux = flux_interp(new_wavelength)
    new_error = error_interp(new_wavelength)

    return new_wavelength, new_flux, new_error


#used:
def find_N(wavelength,flux, z, f, lambda_0):
    
    # Adjust wavelengths for redshift
    z_adjusted_wavelength = (1 / (1 + z)) * wavelength

    # Calculate the optical depth, ensuring no zero or negative flux values
    tau_lambda = -np.log(np.clip(flux, a_min=1e-10, a_max=None))

    # Calculate differences in the adjusted wavelengths
    delta_lambda = np.diff(z_adjusted_wavelength/10**8)

    # Approximate the integral using a simple sum
    integral_tau = np.sum(tau_lambda[:-1] * delta_lambda)

    # Assuming lambda_0 is provided in Angstroms (A), and needs to be in centimeters:
    lambda_0_cm = lambda_0 / 10**8  # converting from Angstroms to centimeters

    # Calculate N
    factor = (m_e * c**2) / (np.pi * e**2) / (f * lambda_0_cm**2)
    N = factor * integral_tau

    log_N=np.log10(N)

    return log_N



def voigt(a, u):
   
   return wofz(u + 1j * a).real

def calctau(wave, z, logN, b, num, AtomDB,element):

    info=AtomDB[AtomDB['Transition'] == element]

    line_actual=info['Wavelength'].iloc[num]
    f=info['Strength'].iloc[num]
    gamma=np.float64(info['Tau'].iloc[num])


    wave0=(1+z)*line_actual
    
    # Go from logN to N
    N = 10.0**logN
    
    # calculate the rest-frame frequency in 1/s
    nu0 = c_As/wave0
    
    # Go into ion rest-frame
    wave_rest = wave#/(1 + z)
    
    # Calculate nu in the rest-frame
    nu_rest = c_As/wave_rest
    
    dopplerWidth = b*nu0/c_kms
    a = gamma/4.0/np.pi/dopplerWidth
    u = (nu_rest - nu0)/dopplerWidth
    
    H = voigt(a, u)
    
    # calculate tau
    #tau = N*np.pi*e**2/m_e/c*f/np.sqrt(np.pi)/dopplerWidth*H
    #tau = ((N*np.pi*(e**2)*(wave0_cm**2))/(m_e*c**2))*f*H
    tau = N*np.pi*e**2/m_e/c*f/np.sqrt(np.pi)/dopplerWidth*H
    
    return np.exp(-tau)

def kernel_gaussian(wave, wave_mean, sigma):
   
   kernel = 1/np.sqrt(2*sigma**2*np.pi)*np.exp(-(wave - wave_mean)**2/(2*sigma**2))
   kernel = kernel/np.sum(kernel)
   
   return kernel

def convolve_flux(wave,flux):
    # Assume that the wavelength interval is small
     wave_mean = np.mean(wave)
     resvel=read_parameter('resolution')
     dW_fwhm = resvel/c_kms*wave_mean
     dW_sigma = dW_fwhm/2/np.sqrt(2*np.log(2))
     
     #pixScale = wave[int(len(wave)/2)] - wave[int(len(wave)/2 - 1)]  
     pixScale = wave[1]-wave[0]
     dPix_sigma = dW_sigma/pixScale
     
     
     pix_kernel = np.concatenate((-1*np.arange(1, 10*dPix_sigma, 1), [0],
                                 np.arange(1, 10*dPix_sigma, 1)))
     pix_kernel.sort()
     
     pix_mean = 0.0
  
     kernel = kernel_gaussian(pix_kernel, pix_mean, dPix_sigma)
     # Continuum subtract and invert to prevent edge effects
     flux = flux - 1
     flux = flux*-1
     flux = convolve(flux, kernel, 'same')
     
     
     # Now undo continuum subtraction and inversion
     flux = flux*-1
     flux = 1 + flux

     return flux