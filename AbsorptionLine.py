import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.widgets import Slider
import os
from scipy.signal import find_peaks
from scipy.integrate import simpson
import pickle
import io
import base64
import smplotlib as sm
from scipy.integrate import simps

from essential_functions import read_atomDB,clear_directory
#from mcmc import run_mcmc


# Constants
e = 4.8032e-10 # electron charge in stat-coulumb
m_e = 9.10938e-28 # electron mass
c = 2.9979e10 # cm/s
c_As = 2.9979e18
c_kms = 2.9979e5
k = 1.38065e-16 # erg/K

AtomDB=read_atomDB()

#helper functions
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Open the file in binary write mode
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)  # Pickle the object and write to file

#____________________________________________________________________________________________________

class Absorber:
    def __init__(self,z,z_err,MgII_pair):

        self.z=z
        self.z_err=z_err

        self.lines={
            'MgII 2796.355099':MgII_pair[0],
            'MgII 2803.5322972':MgII_pair[1]
        }

        self.plot_base64_dict={}

    def add_line(self,key,line):

        self.lines[key]=line

    def export_data(self):

        df = pd.DataFrame([
            {'key': key, 'start': obj.start, 'end': obj.end}
            for key, obj in self.lines.items()
        ])

        df.to_csv(f"Absorbers/csvs/{self.z:.2f}_absorber.csv")

        save_object(self,f'Absorbers/objs/{self.z:.2f}_absorber.pkl')

    def list_found_elements(self):

        temp_dict={}

        for line in self.lines.keys():
            element=line.split(' ')[0]
            temp_dict[element]=0

        return temp_dict.keys()
    
    def make_vel_plot(self,element,index=0):
        
        #clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/velocity_plots/')

        import smplotlib

        c = 3e5  # Speed of light in km/s
        velocity_window = 200  # km/s, adjust as necessary

        # Filter absorption lines for the selected element
        lines = [line for key, line in self.lines.items() if key.split(' ')[0] == element]

        if not lines:
            print(f"No absorption lines found for {element}.")
            return

        for line in lines:
            line.find_N()
            line.actual_ew_func()

        strongest_line = None
        highest_ew = float('-inf')
        for line in lines:
            if line.actual_ew>highest_ew:
                highest_ew=line.actual_ew
                strongest_line=line

        reference_z=((strongest_line.MgII_wavelength[0]+strongest_line.MgII_wavelength[-1])/2 - strongest_line.suspected_line)/strongest_line.suspected_line



        fig, axs = plt.subplots(len(lines), 1, figsize=(10, 3 * len(lines)), squeeze=False,sharex=True,sharey=True)

        axs_flat = axs.ravel()

        for i,ax in enumerate(axs_flat):

            line=lines[i]

            reference_microline=(reference_z+1)*line.suspected_line

            full_velocity = (lines[i].extra_wavelength - reference_microline) / reference_microline * c
            velocity = (lines[i].MgII_wavelength - reference_microline) / reference_microline * c
            
            ax.step(full_velocity, lines[i].extra_flux, where='mid', label=f"Flux", color="grey")
            ax.step(full_velocity, lines[i].extra_errors, where='mid', label="Error", color="purple")

            ax.step(velocity, lines[i].MgII_flux, where='mid', label=f"Spectrum of {lines[i].peak:.2f} Å", color="black")

            for microline in line.mcmc_microlines:
                
                microline_vel=(microline.wavelength - reference_microline) / reference_microline * c

                ax.axvspan(microline_vel[0], microline_vel[-1], color='grey', alpha=0.5)
            
            '''
            for microline in line.microLines:
                micro_velocity = (microline.peak - reference_microline) / reference_microline * c
                ax.axvline(micro_velocity, color='red', linestyle='--')'''

            ax.set_title(
                        f"line: {line.suspected_line:.2f} Å at {line.peak:.2f} Å \n"
                        f"EW: {line.actual_ew:.2f} Å ± {line.actual_ew_error:.4f} Å \n"
                        f"log(N): {line.log_N:.2f} cm⁻² ± {np.log10(line.sigma_N_plus):.2f} cm⁻²",
                        fontsize=10
                    )
            
            ax.set_xlim(-velocity_window, velocity_window)

        # Label axes and configure layout
        axs_flat[-1].set_xlabel('Relative Velocity (km/s)', fontsize=12)

        for ax in axs[:, 0]:  # Set y-label for the first column
            ax.set_ylabel('Flux', fontsize=12)

        plt.suptitle(f"Velocity Profiles for {element}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for title

        plt.savefig(f"Absorbers/vel_plots/velPlot_{self.z:.2f}_{element}.png")
        #plt.savefig(f'static/Data/velocity_plots/velocityPlot_{element}.png')
        plt.savefig(f"static/Data/velocity_plots/velocityPlot_{element}_{index}.png")

        # Save plot to a BytesIO buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()

        # Encode plot to Base64
        buffer.seek(0)
        self.plot_base64_dict[element] = base64.b64encode(buffer.read()).decode('utf-8')

        #print(f"Done with {element}")


    def get_plot_base64(self, element):
        if element not in self.plot_base64_dict:
            self.make_vel_plot(element)  # Only generate if it hasn't been made
        return self.plot_base64_dict.get(element)
    
    def mcmc(self,element,nsteps,nwalkers):

        #run_mcmc(self,element,nsteps,nwalkers)
        pass

    def continue_mcmc(self, nsteps=1000,nwalkers=250):
        """Continue MCMC for the given element."""
        from mcmc import continue_mcmc  # Ensure import here to avoid circular imports
        continue_mcmc(self, nsteps,nwalkers)


    def multi_mcmc(self,elements,nsteps=1000,nwalkers=250):

        from mcmc import run_multi_mcmc

        run_multi_mcmc(self,elements,nsteps,nwalkers)

#____________________________________________________________________________________________________

class MicroAbsorptionLine:
    def __init__(self,wavelength, flux, errors, start_ind,end_ind):

        self.wavelength = wavelength
        self.flux = flux
        self.errors = errors

        self.number_of_zeros_error = np.count_nonzero(self.errors == 0.)

        self.global_start_ind=start_ind
        self.global_end_ind=end_ind

        self.peak_ind=np.argmin(self.flux)
        self.peak = self.wavelength[self.peak_ind]

        self.center=(self.wavelength[0]+self.wavelength[-1])/2

        self.z=None

        #to be updated during vpfit mgmatching
        self.suspected_line=None
        self.f=None
        self.gamma=None

        #to be updated during mcmc
        self.mcmc_z=None
        self.mcmc_vel=None
        self.mcmc_logN=None
        self.mcmc_b=None

        sat_count=np.count_nonzero(self.flux <= 0.1)
        if sat_count>2:
            self.is_saturated=True

            left=simps((1-self.flux[0:int(len(wavelength)/2)]),self.wavelength[0:int(len(wavelength)/2)])
            right=simps((1-self.flux[int(len(wavelength)/2):-1]),self.wavelength[int(len(wavelength)/2):-1])

            if left>=right:
                self.saturation_direction='left'
            elif right>left:
                self.saturation_direction='right'

        else:
            self.is_saturated=False
    
    
    def calcw(self):
   
        oneminusflux = 1-self.flux
        z_adjusted_wavelength = (1 / (1 + self.z)) * self.wavelength
        
        self.W = simps(oneminusflux, z_adjusted_wavelength)
            
        return self.W


    def calc_linear_N(self):
        
        wave_cm = self.suspected_line*1e-8
        W=self.calcw()

        if W<=0.:
            W=1e-5
        
        # Calculate equivalent width in linear regime and convert from cm to A
        N = W/(wave_cm**2/c**2)/np.pi/e**2*m_e/self.f/1e8

        self.N=N
        self.logN=np.log10(N)

        return N
        


    '''def find_N(self):

        z=self.z
        f=self.f
        lambda_0=self.suspected_line
        
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

        self.logN=np.log10(self.N)
        self.log_sigma_N= self.sigma_N / self.N

        return self.N,self.sigma_N '''

#____________________________________________________________________________________________________

class AbsorptionLineSystem:
    def __init__(self, vpfit,lines):

        self.vpfit=vpfit

        if isinstance(lines,list):
            self.microLines=lines

            self.start_ind=self.microLines[0].global_start_ind
            self.end_ind=self.microLines[-1].global_end_ind

        elif isinstance(lines,tuple):
            self.start_ind=lines[0]
            self.end_ind=lines[1]

        self.wavelength=vpfit.wavelength[self.start_ind:self.end_ind]
        self.flux = vpfit.flux[self.start_ind:self.end_ind]
        self.errors = vpfit.error[self.start_ind:self.end_ind]

        padding=500
        self.extra_wavelength=vpfit.wavelength[self.start_ind-padding:self.end_ind+padding]
        self.extra_flux=vpfit.flux[self.start_ind-padding:self.end_ind+padding]
        self.extra_errors=vpfit.error[self.start_ind-padding:self.end_ind+padding]

        self.start=self.wavelength[0]
        self.end=self.wavelength[-1]
        
        self.contains_zeros = np.any(self.errors == 0)
        
        zero_flux_indices = [i for i, flux in enumerate(self.flux) if flux == 0.]  # Adjust the threshold as needed
        self.number_of_zeros_flux=len(zero_flux_indices)

        self.number_of_zeros_error = np.count_nonzero(self.errors == 0.)

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

        self.center=(self.wavelength[0]+self.wavelength[-1])/2

        #initialize for later
        self.suspected_line=1

        self.f=1

        self.z=1

        self.z_err=1

        self.name=None

        self.possible_lines=[]

        self.mcmc_microlines=[]

    def update_line_attributes(self):

        row=AtomDB[AtomDB['Wavelength'] == self.suspected_line]

        self.f=float(row['Strength'])
        self.gamma=float(row['Tau'])

    def MgII_dimensions(self,z_low,z_high,MgII=False):

        if MgII:
            self.MgII_wavelength=self.wavelength
            self.MgII_flux = self.flux
            self.MgII_errors = self.errors

            return

        low_wavelength=(1+z_low)*self.suspected_line
        low_ind= np.argmin(np.abs(self.vpfit.wavelength - low_wavelength))

        high_wavelength=(1+z_high)*self.suspected_line
        high_ind=np.argmin(np.abs(self.vpfit.wavelength - high_wavelength))

        

        self.MgII_wavelength = self.vpfit.wavelength[low_ind:high_ind]
        self.MgII_flux = self.vpfit.flux[low_ind:high_ind]
        self.MgII_errors = self.vpfit.error[low_ind:high_ind]

    def give_data(self,start_z,stop_z):

        start_ind=np.argmin(np.abs(self.extra_wavelength-(1+start_z)*self.suspected_line))
        stop_ind=np.argmin(np.abs(self.extra_wavelength-(1+stop_z)*self.suspected_line))

        wavelength=self.extra_wavelength[start_ind:stop_ind]
        flux=self.extra_flux[start_ind:stop_ind]
        errors=self.extra_errors[start_ind:stop_ind]

        return wavelength,flux,errors

    '''
    def reset_later_params(self):
        self.suspected_line=1
        self.f=1
        self.z=1
        self.z_err=1
        self.name=None
        self.possible_lines=[]
    '''

    def update_microlines(self):
        for microline in self.microLines:
            microline.suspected_line=self.suspected_line
            microline.f=self.f
            microline.gamma=self.gamma

    def update_mcmc_microlines(self):
        for microline in self.mcmc_microlines:
            microline.suspected_line=self.suspected_line
            microline.f=self.f
            microline.gamma=self.gamma
    

    def calculate_N(self):

        N=0
        N_err=0

        self.update_microlines()

        for microline in self.microLines:

            found_N,found_N_err=microline.find_N()
            N+=found_N
            N_err+=found_N_err

        self.N=N
        self.N_err=N_err
        self.log_N=np.log10(N)
        self.log_N_err=np.log10(N_err)

            

    def actual_ew_func(self):

        self.suspected_z=(self.peak-self.suspected_line)/self.suspected_line
        
        # Adjust wavelengths for redshift
        z_adjusted_wavelength = (1 / (1 + self.suspected_z)) * self.MgII_wavelength
        
        # Differences in the adjusted wavelengths
        delta_lambda = np.diff(z_adjusted_wavelength)
        
        # Calculate equivalent width
        self.actual_ew = np.sum((1 - self.MgII_flux[:-1]) * delta_lambda)
        
        # Calculate the error in equivalent width
        ew_errors = delta_lambda * self.MgII_errors[:-1]  # Assuming flux_errors array aligns with self.flux
        self.actual_ew_error = np.sqrt(np.sum(ew_errors**2))
        
        return self.actual_ew, self.actual_ew_error
    
    def equivalent_width(self):

        self.ew = simpson(1 - self.flux, x=self.wavelength) 

        return self.ew
    

    def find_N(self):

        z=self.z
        f=self.f
        lambda_0=self.suspected_line

        # Adjust wavelengths for redshift
        z_adjusted_wavelength = (1 / (1 + z)) * self.wavelength

        # Calculate the normalized flux
        R = np.clip(self.flux, a_min=1e-10, a_max=None)
        
        # Calculate the uncertainty in R
        sigma_R = R * np.sqrt((self.errors / R)**2)  # Assuming no continuum fitting errors

        # Compute asymmetric uncertainties in optical depth
        sigma_tau_plus = np.log(R + sigma_R) - np.log(R)
        sigma_tau_minus = np.log(R) - np.log(np.clip(R - sigma_R, a_min=1e-10, a_max=None))

        # Compute the optical depth
        tau_lambda = -np.log(R)

        # Calculate differences in the adjusted wavelengths
        delta_lambda = np.diff(z_adjusted_wavelength / 10**8)

        # Approximate the integral using a simple sum
        integral_tau = np.sum(tau_lambda[:-1] * delta_lambda)

        # Propagate the asymmetric uncertainties
        sigma_integral_tau_plus = np.sqrt(np.sum((sigma_tau_plus[:-1] * delta_lambda) ** 2))
        sigma_integral_tau_minus = np.sqrt(np.sum((sigma_tau_minus[:-1] * delta_lambda) ** 2))

        # Constants in CGS
        m_e = 9.109 * 10**(-28)  # electron mass in grams
        c = 2.998 * 10**10       # speed of light in cm/s
        e = 4.8 * 10**(-10)      # elementary charge

        # Convert lambda_0 from Angstroms to centimeters
        lambda_0_cm = lambda_0 / 10**8  

        # Compute the column density
        factor = (m_e * c**2) / (np.pi * e**2) / (f * lambda_0_cm**2)
        self.N = factor * integral_tau

        # Compute asymmetric errors in N
        self.sigma_N_plus = factor * sigma_integral_tau_plus
        self.sigma_N_minus = factor * sigma_integral_tau_minus

        # Logarithmic errors
        self.log_N = np.log10(self.N)
        self.log_sigma_N_plus = 0.4343 * (self.sigma_N_plus / self.N)
        self.log_sigma_N_minus = 0.4343 * (self.sigma_N_minus / self.N)

        if self.log_sigma_N_plus >= self.log_sigma_N_minus:
            self.log_sigma_N = self.log_sigma_N_plus
        else:
            self.log_sigma_N = self.log_sigma_N_minus

        return self.N, self.sigma_N_plus


    def calc_linear_n(self):
        
        wave_cm = self.wavelength*1e-8
        W=self.actual_ew
        f=self.f
        
        # Calculate equivalent width in linear regime and convert from cm to A
        N = W/(wave_cm**2/c**2)/np.pi/e**2*m_e/f/1e8

        self.N=N
        self.log_N=np.log10(N)

        return N
        


        
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

    def update_line_attributes(self):

        row=AtomDB[AtomDB['Wavelength'] == self.suspected_line]

        self.f=float(row['Strength'])
        self.gamma=float(row['Tau'])

    def init_possible_lines(self):
        self.possilbe_lines=[]

    def extend_possible_lines(self,inp_list):
        self.possible_lines.extend(inp_list)

    def chose_best_line(self):

        self.sorted_absorptions = sorted(self.possible_lines, key=lambda line: abs((1+line[0])*line[2] - self.peak))

        if len(self.sorted_absorptions)==0:
            return None

        chosen_line=self.sorted_absorptions[0]

        if len(chosen_line)==1:
            return None

        self.name=f"{chosen_line[1]} {chosen_line[2]}"

        self.z=chosen_line[0]

        self.suspected_line=chosen_line[2]
        self.update_line_attributes()
        self.update_microlines()

        return chosen_line
    
    def find_n_peaks(self):

        peaks,_= find_peaks(self.flux,prominence=.1)

        self.peaks=peaks

        return peaks
    
    def find_mcmc_microlines(self):

        self.mcmc_microlines=[]

        peaks,_= find_peaks(1-self.MgII_flux,prominence=.1)

        self.peaks=peaks

        for peak_ind in peaks:
            left = peak_ind
            right = peak_ind
            
            # Go left
            while left > 0 and self.MgII_flux[left] <= self.MgII_flux[left - 1]:
                left -= 1
            
            # Go right
            while right < len(self.MgII_flux) - 1 and self.MgII_flux[right] <= self.MgII_flux[right + 1]:
                right += 1
            
            # Convert to global indices
            global_left = left + self.start_ind
            global_right = right + self.start_ind
            
            # Extract data slices
            wavelength_slice = self.MgII_wavelength[left:right+1]
            flux_slice = self.MgII_flux[left:right+1]
            error_slice = self.MgII_errors[left:right+1]
            
            # Create new microLine object (adjust according to your microLine class)
            #new_microline = deepcopy(self.microLines[0])  # Copy structure
            new_microline = MicroAbsorptionLine(wavelength_slice,flux_slice,error_slice,global_left,global_right)
            '''new_microline.wavelength = wavelength_slice
            new_microline.flux = flux_slice
            new_microline.errors = error_slice
            new_microline.global_start_ind = global_left
            new_microline.global_end_ind = global_right
            new_microline.peak_ind = peak_ind
            new_microline.peak = self.wavelength[peak_ind]'''
            new_microline.z=self.z
            new_microline.suspected_line=self.suspected_line

            #new_microline.find_N(self.z,self.f,self.suspected_line)
            
            # Append
            self.mcmc_microlines.append(new_microline)
    
        return self.mcmc_microlines

    
#____________________________________________________________________________________________________


class Custom_absorption_line:
    def __init__(self,wavelength,flux,errors,element,transition):

        self.wavelength = wavelength
        self.flux = flux
        self.errors = errors

        self.start=self.wavelength[0]
        self.stop=self.wavelength[-1]

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

        try:
            self.center=(self.wavelength[0]+self.wavelength[1])/2
        except:
            self.center=self.wavelength[0]

        self.suspected_element=element

        self.mcmc_microlines=[]

        #look up and find the correct transition and other aspects
        row = AtomDB[(AtomDB['Transition'] == self.suspected_element) & (AtomDB['Floor'] == int(transition))]

        if not row.empty:
            self.suspected_line = row['Wavelength'].iloc[0]  # Extracts the first item
            self.f = row['Strength'].iloc[0]
            self.gamma = row['Tau'].iloc[0]
        else:
            print("No matching rows found.")

        self.z=(self.peak-self.suspected_line)/self.suspected_line

        self.name=f'{self.suspected_element} {self.suspected_line}'

        self.plot()

    def bonus(self,wav,flux,error):

        self.extra_wavelength=wav
        self.extra_flux=flux
        self.extra_errors=error

        self.MgII_wavelength=self.wavelength
        self.MgII_flux=self.flux
        self.MgII_errors=self.errors

    def plot(self):
        
        plt.step(self.wavelength,self.flux,where="mid",color='black')
        plt.step(self.wavelength,self.errors,where="mid",color='purple')

        plt.title(f"Element:{self.suspected_element} \n Transition: {np.floor(self.suspected_line)}")

        plt.xlim((self.wavelength[0]-5,self.wavelength[-1]+5))

        # Save plot to a BytesIO buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()

        # Encode plot to Base64
        buffer.seek(0)
        self.plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        plt.clf()

    def update_line_attributes(self):

        row=AtomDB[AtomDB['Wavelength'] == self.suspected_line]

        self.f=float(row['Strength'])
        self.gamma=float(row['Tau'])


    def find_N(self):

        z=self.z
        f=self.f
        lambda_0=self.suspected_line

        # Adjust wavelengths for redshift
        z_adjusted_wavelength = (1 / (1 + z)) * self.wavelength

        # Calculate the normalized flux
        R = np.clip(self.flux, a_min=1e-10, a_max=None)
        
        # Calculate the uncertainty in R
        sigma_R = R * np.sqrt((self.errors / R)**2)  # Assuming no continuum fitting errors

        # Compute asymmetric uncertainties in optical depth
        sigma_tau_plus = np.log(R + sigma_R) - np.log(R)
        sigma_tau_minus = np.log(R) - np.log(np.clip(R - sigma_R, a_min=1e-10, a_max=None))

        # Compute the optical depth
        tau_lambda = -np.log(R)

        # Calculate differences in the adjusted wavelengths
        delta_lambda = np.diff(z_adjusted_wavelength / 10**8)

        # Approximate the integral using a simple sum
        integral_tau = np.sum(tau_lambda[:-1] * delta_lambda)

        # Propagate the asymmetric uncertainties
        sigma_integral_tau_plus = np.sqrt(np.sum((sigma_tau_plus[:-1] * delta_lambda) ** 2))
        sigma_integral_tau_minus = np.sqrt(np.sum((sigma_tau_minus[:-1] * delta_lambda) ** 2))

        # Constants in CGS
        m_e = 9.109 * 10**(-28)  # electron mass in grams
        c = 2.998 * 10**10       # speed of light in cm/s
        e = 4.8 * 10**(-10)      # elementary charge

        # Convert lambda_0 from Angstroms to centimeters
        lambda_0_cm = lambda_0 / 10**8  

        # Compute the column density
        factor = (m_e * c**2) / (np.pi * e**2) / (f * lambda_0_cm**2)
        self.N = factor * integral_tau

        # Compute asymmetric errors in N
        self.sigma_N_plus = factor * sigma_integral_tau_plus
        self.sigma_N_minus = factor * sigma_integral_tau_minus

        # Logarithmic errors
        self.log_N = np.log10(self.N)
        self.log_sigma_N_plus = 0.4343 * (self.sigma_N_plus / self.N)
        self.log_sigma_N_minus = 0.4343 * (self.sigma_N_minus / self.N)

        return self.N, self.sigma_N_plus, self.sigma_N_minus
    
    def give_data(self,start_z,stop_z):

        start_ind=np.argmin(np.abs(self.extra_wavelength-(1+start_z)*self.suspected_line))
        stop_ind=np.argmin(np.abs(self.extra_wavelength-(1+stop_z)*self.suspected_line))

        wavelength=self.extra_wavelength[start_ind:stop_ind]
        flux=self.extra_flux[start_ind:stop_ind]
        errors=self.extra_errors[start_ind:stop_ind]

        return wavelength,flux,errors
    
    def update_mcmc_microlines(self):
        for microline in self.mcmc_microlines:
            microline.suspected_line=self.suspected_line
            microline.f=self.f
            microline.gamma=self.gamma

    def find_mcmc_microlines(self):

        from copy import deepcopy

        self.mcmc_microlines=[]

        peaks,_= find_peaks(1-self.flux,prominence=.1)

        self.peaks=peaks

        for peak_ind in peaks:
            left = peak_ind
            right = peak_ind
            
            # Go left
            while left > 0 and self.flux[left] <= self.flux[left - 1]:
                left -= 1
            
            # Go right
            while right < len(self.flux) - 1 and self.flux[right] <= self.flux[right + 1]:
                right += 1
            
            # Convert to global indices
            global_left = 1
            global_right = 2
            
            # Extract data slices
            wavelength_slice = self.wavelength[left:right+1]
            flux_slice = self.flux[left:right+1]
            error_slice = self.errors[left:right+1]
            
            # Create new microLine object (adjust according to your microLine class)
            #new_microline = deepcopy(self.microLines[0])  # Copy structure
            new_microline = MicroAbsorptionLine(wavelength_slice,flux_slice,error_slice,global_left,global_right)
            '''new_microline.wavelength = wavelength_slice
            new_microline.flux = flux_slice
            new_microline.errors = error_slice
            new_microline.global_start_ind = global_left
            new_microline.global_end_ind = global_right
            new_microline.peak_ind = peak_ind
            new_microline.peak = self.wavelength[peak_ind]'''
            new_microline.suspected_line=self.suspected_line
            
            # Append
            self.mcmc_microlines.append(new_microline)
    
        return self.mcmc_microlines


#____________________________________________________________________________________________________

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

    def update_line_attributes(self):

        row=AtomDB[AtomDB['Wavelength'] == self.suspected_line_loc]

        self.f=float(row['Strength'])
        self.gamma=float(row['Tau'])


