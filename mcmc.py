#mcmc but both lines are being compared and model is being created every step

import emcee
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import corner
import os
import pickle
from multiprocessing import Pool

# Import your specific functions
from essential_functions import read_parameter, read_atomDB
from mcmc_functions import find_N, voigt, calctau, kernel_gaussian, convolve_flux

e = 4.8032e-10 # electron charge in stat-coulumb
m_e = 9.10938e-28 # electron mass
c = 2.9979e10 # cm/s
c_As = 2.9979e18
c_kms = 2.9979e5
k = 1.38065e-16 # erg/K

def load_object(filename):
    with open(filename, 'rb') as inp:  # Open the file in binary read mode
        return pickle.load(inp)  # Return the unpickled object
    
def total_model(params, lines, AtomDB, element, convolve_data=True):
    num_peaks = len(params) // 3
    models = []
    for i,line in enumerate(lines):
        model = np.ones_like(line.wavelength)
        for j in range(num_peaks):
            z, logN, b = params[3*j:3*j+3]
            model *= calctau(line.wavelength, z, logN, b, i, AtomDB=AtomDB, element=element)
        if convolve_data:
            model = convolve_flux(line.wavelength, model)
        models.append(model)
    return models

def log_likelihood(params, lines, AtomDB, element):
    models = total_model(params, lines, AtomDB, element)
    chi2 = 0
    for line, model in zip(lines, models):
        chi2 += np.sum(-0.5 * ((line.flux - model) / line.errors) ** 2)
    return chi2

def log_prior(params):
    for i in range(len(params) // 3):
        z, logN, b = params[3*i:3*i+3]
        if not (0 < z < 2 and 0 < b < 20 and 5 < logN < 30):
            return -np.inf
    return 0.0  # Uniform prior

def log_probability(params, lines, AtomDB, element):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, lines, AtomDB, element)


def run_mcmc(element, selection, nsteps=200, nwalkers=250):

    #data collection
    AtomDB=read_atomDB()

    data_loc=f'found_lines/{element}/'
    dir_list=os.listdir(data_loc)
    dir_list = [i for i in dir_list if f'line_{selection}' in i]
    dir_list = sorted(dir_list, key=lambda x: int(x.split(',')[-1]))

    line_list=[]
    for line in dir_list:
        line_list.append(load_object(data_loc+line))

    #--------------------------------------------------------------------------------------------------------------------------------------------
    #Preparing for mcmc

    #find peaks
    peaks,properties=find_peaks(-line_list[0].flux+1,height=1*np.std(line_list[0].flux))

    #CHANGE!!
    #peaks=[peaks[0]]

    # Calculate the bounds for each peak's equivalent width
    bounds = []
    for peak in peaks:
        left = peak
        while left > 0 and line_list[0].flux[left] < line_list[0].flux[left - 1]:
            left -= 1
        right = peak
        while right < len(line_list[0].flux) - 1 and line_list[0].flux[right] < line_list[0].flux[right + 1]:
            right += 1
        bounds.append((left, right))

    initial_guesses=[] #list of param items for each peak

    info=AtomDB[AtomDB['Transition'] == element]
    line_actual=info['Wavelength'].iloc[0]
    f=info['Strength'].iloc[0]

    for i,peak in enumerate(peaks):
        guess_z=(line_list[0].wavelength[peak]-line_actual)/line_actual
        wave_zone=line_list[0].wavelength[bounds[i][0]:bounds[i][1]]
        flux_zone=line_list[0].flux[bounds[i][0]:bounds[i][1]]
        guess_logN=find_N(wave_zone,flux_zone,guess_z,f,line_actual)
        guess_b=(bounds[i][1]-bounds[i][0])
        
        initial_guesses.extend([guess_z, guess_logN, guess_b]) #z,LogN,b

    #plot initial guess
    model_list=total_model(initial_guesses,line_list,AtomDB,element)
    fig, axs = plt.subplots(2, 1,figsize=(10,5),sharex=True)

    c = 299792.458  # Speed of light in km/s

    mid=int((len(initial_guesses)//3)/2)*3
    z=initial_guesses[mid]

    for i,model in enumerate(model_list):

        center_wavelength=(1+z)*info['Wavelength'].iloc[i]
        velocity = (line_list[i].wavelength-center_wavelength)/center_wavelength*c

        axs[i].step(velocity,line_list[i].flux, 'k', label='Observed Flux', where='mid')
        axs[i].step(velocity,line_list[i].errors, c='purple', label='Error', where='mid')


        axs[i].step(velocity,model,c='red',label="Model",where='mid')

        for j,peak in enumerate(peaks):
            axs[i].axvline(velocity[peak])
            if i==0:
                axs[i].text(velocity[peak],.2,f"{initial_guesses[j*3]:.5f}\n{initial_guesses[j*3+1]:.2f}\n{initial_guesses[j*3+2]:.2f}")
        
    axs[1].set_xlabel('Velocity (km/s)')  # Label only the bottom plot's x-axis
    axs[0].set_ylabel('Flux')
    axs[1].set_ylabel('Flux')

    plt.savefig(f"static/Data/mcmc/inititial_guess.png")

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    #mcmc

    initial_guesses = np.array(initial_guesses)

    ndim = len(initial_guesses)
    percent_off = 0.0001
    # Generate the position array
    pos = initial_guesses + (percent_off * initial_guesses * np.random.uniform(-1, 1, (nwalkers, ndim)))
    
    #12 max
    #5x faster!
    num_processes = os.cpu_count()
    with Pool(num_processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(line_list, AtomDB, element), pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

    # Analyzing the results
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    log_probs = sampler.get_log_prob(discard=100, thin=15, flat=True)
    map_params = flat_samples[np.argmax(log_probs)]

    #--------------------------------------------------------------------------------------------------------------------------------------------

    #plot results
    fig, axs = plt.subplots(2, 1,figsize=(10,5),sharex=True)
    model_list = total_model(map_params,line_list,AtomDB,element)

    c = 299792.458  # Speed of light in km/s

    mid=int((len(initial_guesses)//3)/2)*3
    z=initial_guesses[mid]

    num_peaks = len(map_params) // 3

    for i,model in enumerate(model_list):

        center_wavelength=(1+z)*info['Wavelength'].iloc[i]
        velocity = (line_list[i].wavelength-center_wavelength)/center_wavelength*c

        axs[i].step(velocity,line_list[i].flux, 'k', label='Observed Flux', where='mid')
        axs[i].step(velocity,line_list[i].errors, c='purple', label='Error', where='mid')


        axs[i].step(velocity,model,c='red',label="Model",where='mid')

        for j in range(num_peaks):

            peak_z, logN, b = map_params[3*j:3*j+3]

            peak_wavelength=(1+peak_z)*info['Wavelength'].iloc[i]
            velocity = (peak_wavelength-center_wavelength)/center_wavelength*c

            axs[i].axvline(velocity,c="black")
            axs[i].text(velocity+5,.2,f"Z={initial_guesses[j*3]:.5f}\nLogN={initial_guesses[j*3+1]:.2f}\nb={initial_guesses[j*3+2]:.2f}",fontsize=8)
        
    axs[0].legend()

    plt.xlim((-200,200))

    axs[1].set_xlabel('Velocity (km/s)')  # Label only the bottom plot's x-axis
    axs[0].set_ylabel('Flux')
    axs[1].set_ylabel('Flux')

    plt.savefig(f"static/Data/mcmc/mcmc_result.png")
    plt.clf()

    #plot trace plots of params
    def plot_trace(sampler, param_names):
        nwalkers, nsteps, ndim = sampler.chain.shape
        fig, axes = plt.subplots(ndim, figsize=(10, 3*ndim), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            for j in range(nwalkers):
                ax.plot(sampler.chain[j, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(sampler.chain[0]))
            ax.set_ylabel(param_names[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            #ax.set_title(f"Autocorrelation time: {tau[i]}")
        
        axes[-1].set_xlabel("Step number")
        plt.tight_layout(h_pad=0)
        plt.savefig("static/Data/mcmc/mcmc_trace.png")
        plt.clf()

    # Define parameter names based on your model setup
    param_names = ['Z', 'LogN', 'b'] * (len(initial_guesses) // 3)  # Adjust according to your model
    plot_trace(sampler, param_names)


    #corner plot
    figure = corner.corner(flat_samples[:,:], labels=param_names,  # Update labels as appropriate
                        quantiles=[0.16, 0.5, 0.84],  # Show the quantiles
                        show_titles=True, title_kwargs={"fontsize": 12})

    plt.savefig("static/Data/mcmc/mcmc_corner.png")
    plt.clf()


    #save the chain
    chain=sampler.get_chain()
    np.save('static/Data/mcmc/chain.npy', chain)  # Saves the chain

