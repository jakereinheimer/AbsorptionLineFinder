import emcee
import numpy as np
from scipy.special import wofz
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd
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

class mcmc_line:
    def __init__(self,example_line,elements,saturation_direction=None):

        self.microLines={}

        #self.params=[example_line.z,example_line.log_N,example_line.b]

        if saturation_direction is not None:
            self.saturation=True
        else:
            self.saturation=False

        #self.z=self.params[0]
        self.z=(example_line.peak-example_line.suspected_line)/example_line.suspected_line
        if saturation_direction is None:
            self.vel=redshift_to_velocity(self.z)
        elif saturation_direction == 'right':
            self.vel=redshift_to_velocity(self.z)+10
        elif saturation_direction == 'left':
            self.vel=redshift_to_velocity(self.z)-5

        #self.logN=self.params[1]
        #self.b=self.params[2]

        lowest_z=(example_line.wavelength[0]-example_line.suspected_line)/example_line.suspected_line
        highest_z=(example_line.wavelength[-1]-example_line.suspected_line)/example_line.suspected_line
        self.z_range=(lowest_z,highest_z)

        self.elements=elements

    def add_line(self,element,inp_line):

        self.microLines[element]=inp_line

    def is_within(self, z):
        return self.z_range[0] <= z <= self.z_range[1]

        
    def export_params(self):

        def get_logN_value(absorption_line):
            return absorption_line.logN

        params = [self.vel]

        self.line_dict={}
        for e in self.elements:
            self.line_dict[e]=[line for key, line in self.microLines.items() if key.split(' ')[0]==e]

        for e in self.elements:
        
            #line_to_use=min(line_dict.get(e), key=get_logN_value,default=None)
            line_to_use=self.line_dict.get(e)[0]

            if line_to_use==None:
                log_N=5
                b=2

            else:
                if self.saturation:
                    log_N = line_to_use.logN
                else:
                    log_N=line_to_use.logN

                b=len(line_to_use.wavelength)

            params.extend([log_N,b])

        return params
    
def velocity_to_redshift(velocity):
    # Converts velocity to redshift
    c = 299792.458  # Speed of light in km/s
    return (np.sqrt((velocity / c + 1)**2 / (1 - (velocity / c)**2)) - 1)

def redshift_to_velocity(redshift):
    # Converts redshift to velocity
    c = 299792.458  # Speed of light in km/s
    return c * ((1 + redshift)**2 - 1) / ((1 + redshift)**2 + 1)

def parse_statuses(statuses, initial_guesses):
    free_indices = []
    fixed_values = {}
    anchor_map = {}

    num_rows = len(statuses)
    num_cols = len(statuses[0])

    for i in range(num_rows):
        for j in range(num_cols):
            status = statuses[i][j]
            if status == 'free':
                free_indices.append((i, j))
            elif status == 'fixed':
                fixed_values[(i, j)] = initial_guesses[i][j]
            elif status.startswith('anchor_to:'):
                target = status.split(':')[1]
                if '_' in target:
                    ti, tj = map(int, target.split('_'))
                else:
                    ti = int(target)
                    tj = j  # anchor to the same parameter index
                anchor_map[(i, j)] = (ti, tj)
            else:
                raise ValueError(f"Unknown status {status} at ({i},{j})")
    
    return free_indices, fixed_values, anchor_map

def rebuild_full_params(free_values, free_indices, fixed_values, anchor_map, shape):
    full_params = np.zeros(shape)

    for (i, j), val in fixed_values.items():
        full_params[i, j] = val

    for idx, (i, j) in enumerate(free_indices):
        full_params[i, j] = free_values[idx]

    for (i, j), (ti, tj) in anchor_map.items():
        full_params[i, j] = full_params[ti, tj]

    return full_params.flatten()



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

def calctau(wave, z, logN, b, line):

    line_actual=line.suspected_line
    f=line.f
    gamma=line.gamma

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


import pandas as pd

def summarize_params(flat_samples, labels, elements, mcmc_lines,file_name):
    summary = []

    params_per_microline = (2 * len(elements)) + 1
    c = 299792.458  # speed of light in km/s

    reference_line_ind=0

    #reference data
    ref_param_block=flat_samples[:, reference_line_ind*params_per_microline:(reference_line_ind+1)*params_per_microline]

    ref_vel_samples = ref_param_block[:, 0]
    ref_vel_median = np.median(ref_vel_samples)
    ref_vel_low, ref_vel_high = np.percentile(ref_vel_samples, [16,84])


    for i, mcmc_line_obj in enumerate(mcmc_lines):

        '''for microline in mcmc_line_obj.microlines:
            if microline.is_saturated:
                pass'''

        #actual data
        param_block = flat_samples[:, i*params_per_microline:(i+1)*params_per_microline]

        vel_samples = param_block[:, 0]
        vel_median = np.median(vel_samples)
        vel_low, vel_high = np.percentile(vel_samples, [16,84])


        #now subtract to get relative velocity
        rel_median = vel_median - ref_vel_median
        rel_low = vel_low - ref_vel_low
        rel_high = vel_high - ref_vel_high
        
        # Add rows for each element
        for j, element in enumerate(elements):
            logN_samples = param_block[:, (j*2)+1]
            b_samples = param_block[:, (j*2)+2]

            try:
                print('saturation check')
                saturated = any(line.is_saturated for line in mcmc_line_obj.line_dict.get(e))
            except:
                saturated = False

            if saturated:
                print('got a saturation')
                logN_low = np.percentile(logN_samples, 5)
                b_high = np.percentile(b_samples, 95)
                logN_median = np.median(logN_samples)
                b_median = np.median(b_samples)

                summary.append({
                'Component': i+1,
                'Species': f"{element} saturated",
                f'dv_c (km/s)': f"{rel_median:.1f} (+{rel_high - rel_median:.2f}/-{rel_median - rel_low:.2f})",
                f'log N': f"{logN_median:.2f} (+{logN_high - logN_median:.2f}/-{logN_median - logN_low:.2f})",
                f'b (km/s)': f"{b_median:.1f} (+{b_high - b_median:.1f}/-{b_median - b_low:.1f})"
                })

            else:
                logN_median = np.median(logN_samples)
                logN_low, logN_high = np.percentile(logN_samples, [16, 84])
                b_median = np.median(b_samples)
                b_low, b_high = np.percentile(b_samples, [16, 84])

                summary.append({
                    'Component': i+1,
                    'Species': f"{element}",
                    f'dv_c (km/s)': f"{rel_median:.1f} (+{rel_high - rel_median:.2f}/-{rel_median - rel_low:.2f})",
                    f'log N': f"{logN_median:.2f} (+{logN_high - logN_median:.2f}/-{logN_median - logN_low:.2f})",
                    f'b (km/s)': f"{b_median:.1f} (+{b_high - b_median:.1f}/-{b_median - b_low:.1f})"
                })

    df = pd.DataFrame(summary)
    df.to_csv(f"static/Data/multi_mcmc/{file_name}.csv",index=False)
    return df
