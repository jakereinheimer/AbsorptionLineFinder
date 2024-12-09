import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.widgets import Slider
import os
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.signal import find_peaks
import emcee
import corner

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

        from scipy.integrate import simpson

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

    def voigt_sum(self, x, *params):
        """
        Returns the sum of multiple Voigt profiles.
        `params` should contain multiple sets of (center, amplitude, sigma, gamma).
        """
        assert len(params) % 4 == 0
        n_profiles = len(params) // 4
        result = np.zeros_like(x)
        for i in range(n_profiles):
            center, amplitude, sigma, gamma = params[4*i:4*i+4]
            result += amplitude * voigt_profile(x - center, sigma, gamma)
        return result

    def linfit(self,name):
        from scipy.signal import find_peaks
        # Detect peaks
        peaks, properties = find_peaks(self.flux, height=0.1)  # Adjust height threshold appropriately

        # Initial guesses for Voigt profile parameters for each peak
        initial_params = []
        for peak in peaks:
            amplitude = properties["peak_heights"][0]
            initial_params.extend([self.wavelength[peak], amplitude, 1.0, 1.0])  # center, amplitude, sigma, gamma

        # Fit the Voigt profiles to the data using curve_fit
        popt, pcov = curve_fit(self.voigt_sum, self.wavelength, self.flux, p0=initial_params,maxfev = 114000)

        # Calculate the Voigt profiles with the optimized parameters
        fitted_profiles = self.voigt_sum(self.wavelength, *popt)

        # Compute residuals (original flux minus the fitted Voigt profiles)
        residuals = self.flux - fitted_profiles

        # Fit a line to the residuals using np.linalg.lstsq
        A = np.vstack([self.wavelength, np.ones(len(self.wavelength))]).T
        m, c = np.linalg.lstsq(A, residuals, rcond=None)[0]

        # Optionally plot the results
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(self.wavelength, self.flux, label='Original Data')
        plt.plot(self.wavelength, fitted_profiles, label='Fitted Voigt Profiles')
        plt.plot(self.wavelength, m * self.wavelength + c, label='Linear Fit to Residuals')
        plt.legend()
        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
        plt.title('Voigt Profile and Linear Fit')
        plt.savefig(name+'.png')

        # Return linear fit parameters and Voigt fit parameters
        return {'slope': m, 'intercept': c, 'voigt_params': popt}




    def perform_mcmc_analysis(self):
        # Define the log likelihood function
        def log_likelihood(theta, x, y, yerr):
            # Reshape theta to fit multiple Voigt profiles
            params = theta.reshape(-1, 4)
            model = np.sum([1 - (p[1] * voigt_profile(x - p[0], p[2], p[3])) for p in params], axis=0)
            sigma2 = yerr ** 2
            return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
        
        def log_prior(theta):
            # Apply constraints on each parameter set
            params = theta.reshape(-1, 4)
            for p in params:
                center, amplitude, sigma, gamma = p
                if not (0 < sigma < 1 and 0 < gamma < 1 and 0 < amplitude < 1 and self.wavelength[0] < center < self.wavelength[-1]):
                    return -np.inf
            return 0.0
        
        def log_probability(theta, x, y, yerr):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(theta, x, y, yerr)

        # Initialize and run MCMC

        self.peaks, _ = find_peaks(-self.flux+1,height=.1)
        
        # Initialize MCMC parameters
        initial = []
        for peak in self.peaks:
            initial.append([self.wavelength[peak], max(self.flux), 1.0, 1.0])  # Initial guesses per peak

        nwalkers, ndim = 100, len(initial) * 4
        #nwalkers = max(100, 4 * ndim)
        pos = np.array(initial).flatten() + 0.01 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(self.wavelength, self.flux, self.errors))
        sampler.run_mcmc(pos, 5000)

        # Process MCMC results
        samples = sampler.get_chain(discard=100, thin=15, flat=True)
        self.fit_results = {
            'mean': np.mean(samples, axis=0),
            'median': np.median(samples, axis=0),
            'hpdi': np.percentile(samples, [2.5, 97.5], axis=0)
        }
        print(self.fit_results)
        return self.fit_results
    

    def plot_fit(self,name):
        # Extract parameters from the results
        best_fit_params = self.fit_results['mean']  # or 'median' or 'map', depending on your preference
        print(best_fit_params)

        modeled_flux = np.ones_like(self.wavelength)

        for i in range(len(self.peaks)):
            center=best_fit_params[4*i]
            amp=best_fit_params[4*i+1]
            sig=best_fit_params[4*i+2]
            gamma=best_fit_params[4*i+3]

            modeled_flux -= (amp * voigt_profile(self.wavelength - center, sig, gamma))
        
        # Generate the model Voigt profile using the best-fit parameters
        
        # Plot the observed data
        plt.figure(figsize=(10, 6))
        plt.step(self.wavelength, self.flux, 'k-', label='Observed Flux')
        plt.errorbar(self.wavelength, self.flux, yerr=self.errors, fmt='.', color='gray', alpha=0.5, label='Error')

        # Plot the fitted model
        plt.plot(self.wavelength, modeled_flux, 'r--', label='Fitted Voigt Profile')
        
        plt.title('Voigt Profile Fit to Spectral Data')
        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
        plt.legend()
        plt.savefig(f"{name}.png")
