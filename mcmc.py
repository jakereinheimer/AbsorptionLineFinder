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
from essential_functions import read_atomDB
from mcmc_functions import find_N, calctau, convolve_flux,mcmc_line, summarize_params,velocity_to_redshift,redshift_to_velocity
from AbsorptionLine import MicroAbsorptionLine

e = 4.8032e-10 # electron charge in stat-coulumb
m_e = 9.10938e-28 # electron mass
c = 2.9979e10 # cm/s
c_As = 2.9979e18
c_kms = 2.9979e5
k = 1.38065e-16 # erg/K

def creep(total_params,line_number,line_dict,elements,mcmc_lines,direction):

    num_params_per_line = 1 + 2 * len(elements)
    param_list_2d = np.array(total_params).reshape(-1, num_params_per_line)

    line_params=param_list_2d[line_number]


    import copy

    start_params=total_params

    start_model, start_chi2=total_multi_model(total_params, line_dict, elements, mcmc_lines,chi2=True)

    #chi2_values=[(start_chi2,total_params,start_model)]
    chi2_values=[]

    number_params_per_line=1+(2*len(elements))

    for i in range(1,20):

        adjusted_params=copy.deepcopy(start_params)
        adjusted_line_params=line_params.copy()

        if direction == "right":
            adjusted_line_params[0] += i
        elif direction == "left":
            adjusted_line_params[0] -= i
        else:
            raise ValueError("Direction must be 'right' or 'left'")

        adjusted_params.extend(adjusted_line_params)

        model, chi2=total_multi_model(adjusted_params, line_dict, elements, mcmc_lines,chi2=True)
        chi2_values.append((chi2,adjusted_params,model))

    sorted_models=sorted(chi2_values, key= lambda item: item[0])
    best_params=sorted_models[0][1]

    print('best params from creep')
    print(best_params)

    mcmc_lines.append(mcmc_lines[line_number])

    return best_params,mcmc_lines


def plot_fits(params, line_dict, elements, mcmc_lines,file_name):

    import smplotlib

    c = 3e5
    vel_window=200

    num_params_per_line = 1 + 2 * len(elements)
    param_list_2d = np.array(params).reshape(-1, num_params_per_line)


    models=total_multi_model(params,line_dict,elements,mcmc_lines,high_resolution=True)
    standard_models=total_multi_model(params,line_dict,elements,mcmc_lines)

    '''
    strongest_line = None
    highest_ew = float('-inf')
    for line in line_dict.values():
        try:
            if line.actual_ew>highest_ew:
                highest_ew=line.actual_ew
                strongest_line=line
        except:
            continue'''

    strongest_line=line_dict.get('MgII 2796.355099')
    if strongest_line is None:
        strongest_line=list(line_dict.values())[0]

    reference_z=(strongest_line.center - strongest_line.suspected_line)/strongest_line.suspected_line
    reference_velocity = redshift_to_velocity(reference_z)


    fig, axs = plt.subplots(len(line_dict.values()), 1, figsize=(10, 3 * len(line_dict.values())), squeeze=False,sharex=True,sharey=True)
    axs_flat = axs.ravel()

    for i, name in enumerate(line_dict.keys()):

        line=line_dict.get(name)

        ax=axs_flat[i]

        #chi squared
        obs_flux = line.MgII_flux
        model_flux = standard_models[name]
        errors = np.sqrt(line.MgII_errors)

        # Calculate chi-squared and reduced chi-squared
        chi_squared = np.sum(((obs_flux - model_flux) / errors) ** 2)
        degrees_of_freedom = len(obs_flux) - (len(mcmc_lines)*3)
        reduced_chi_squared = chi_squared / degrees_of_freedom if degrees_of_freedom != 0 else 0

        ax.text(vel_window-60, 0.3, f"$\chi^2_{{red}}={reduced_chi_squared:.2f}$")

        #actual plot
        reference_microline=(reference_z+1)*line.suspected_line

        full_velocity = (line.extra_wavelength - reference_microline) / reference_microline * c
        velocity =  (line.MgII_wavelength - reference_microline) / reference_microline * c
        
        ax.step(full_velocity, line.extra_flux, where='mid', label=f"Flux", color="black")
        ax.step(full_velocity, line.extra_errors, where='mid', label="Error", color="cyan")

        ax.step(velocity,standard_models.get(name), where='mid', label=f"Model", color="purple")

        ax.step(np.linspace(velocity[0],velocity[-1],len(velocity)*10), models.get(name), where='mid', label=f"Model", color="red")

        for i,line_params in enumerate(param_list_2d):

            z=velocity_to_redshift(line_params[0])
            wavelength = line.suspected_line * (1+z)
            velocity =  (wavelength - reference_microline) / reference_microline * c

            ax.vlines(velocity, ymin=1.2,ymax=1.3,color='blue')
        
        '''
        for microline in line.mcmc_microlines:
            microline_vel=(microline.wavelength - reference_microline) / reference_microline * c
            #ax.axvspan(microline_vel[0], microline_vel[-1], color='grey', alpha=0.3)
            ax.vlines(microline_vel[np.argmin(microline.flux)], ymin=1.2,ymax=1.3)'''
        
        ax.text(vel_window-60,0.1,f'{name.split(" ")[0]} {int(np.floor(float(name.split(" ")[1])))}')
        ax.set_xlim(-vel_window, vel_window)

    # Label axes and configure layout
    axs_flat[-1].set_xlabel('Relative Velocity (km/s)', fontsize=12)

    for ax in axs[:, 0]:  # Set y-label for the first column
        ax.set_ylabel('Flux', fontsize=12)

    plt.subplots_adjust(hspace=0)

    #plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for title

    plt.savefig(f"static/Data/multi_mcmc/{file_name}.png")

def load_object(filename):
    with open(filename, 'rb') as inp:  # Open the file in binary read mode
        return pickle.load(inp)  # Return the unpickled object
    
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Open the file in binary write mode
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)  # Pickle the object and write to file # Pickle the object and write to file
    
def total_model(params, lines, AtomDB, element, convolve_data=True):
    num_peaks = len(params) // 3
    models = []
    for i,line in enumerate(lines):
        model = np.ones_like(line.wavelength)
        for j in range(num_peaks):
            velocity, logN, b = params[3*j:3*j+3]
            z=velocity_to_redshift(velocity)
            model *= calctau(line.wavelength, z, logN, b, line)
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
        velocity, logN, b = params[3*i:3*i+3]
        if not (0 < velocity < redshift_to_velocity(2) and 0 < b < 20 and 5 < logN < 30):
            return -np.inf
    return 0.0  # Uniform prior

def log_probability(params, lines, AtomDB, element):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, lines, AtomDB, element)

def total_multi_model(params, line_dict, elements, mcmc_lines, convolve_data=True,high_resolution=False,chi2=False):

    params_per_microline=(2*len(elements))+1
    param_list_2d = np.array(params).reshape(-1, params_per_microline)

    models={}

    if chi2:
        chi_value=0

    for key,line in line_dict.items():

        if high_resolution:
            wavelength=np.linspace(line.MgII_wavelength[0],line.MgII_wavelength[-1],len(line.MgII_wavelength)*10)

        else:
            wavelength=line.MgII_wavelength

        models[key]=np.ones_like(wavelength)

        for i,line_params in enumerate(param_list_2d):

            velocity=line_params[0]

            for j,e in enumerate(elements):
                if e == key.split(' ')[0]:

                    logN=line_params[(j*2)+1]
                    b=line_params[(j*2)+2]

                    z=velocity_to_redshift(velocity)

                    models[key]*=calctau(wavelength,z,logN,b,line)

                    if convolve_data:
                        models[key] = convolve_flux(wavelength, models[key])

                    if chi2:
                        obs_flux = line.MgII_flux
                        model_flux = models[key]
                        errors = np.sqrt(line.MgII_errors)

                        # Calculate chi-squared and reduced chi-squared
                        chi_value+= np.sum(((obs_flux - model_flux) / errors) ** 2)

    if chi2:
        return models,chi_value
    else:
        return models

def log_multi_likelihood(params, lines, elements, mcmc_lines):

    models = total_multi_model(params, lines, elements, mcmc_lines)

    chi2 = 0

    for key,line in lines.items():
        model=models.get(key)
        #chi2 += np.sum(-0.5 * ((line.MgII_flux - model) / line.MgII_errors) ** 2)
        chi2 += np.sum(np.log(1/np.sqrt(2*np.pi)/line.MgII_errors) + -(line.MgII_flux - model)**2/2/line.MgII_errors**2)

    return chi2

def log_multi_prior(params,elements,mcmc_lines):

    params_per_microline=(2*len(elements))+1

    for i,line in enumerate(mcmc_lines):

        line_params=params[i*params_per_microline:(i*params_per_microline)+params_per_microline]

        velocity=line_params[0]

        for j,e in enumerate(elements):

            logN=line_params[(j*2)+1]
            b=line_params[(j*2)+2]

            if not (0 < velocity < redshift_to_velocity(2) and 0 < b < 20 and 0 < logN < 20):
                return -np.inf
            
    return 0.0

def log_multi_probability(params, lines, elements, mcmc_lines):
    lp = log_multi_prior(params,elements,mcmc_lines)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_multi_likelihood(params, lines, elements, mcmc_lines)



#again again again
def run_multi_mcmc(absorber,elements,nsteps=1000,nwalkers=250):

    if isinstance(absorber, dict):
        line_dict=absorber
        custom=True

    else:
        custom=False
        line_dict={}
        for e in elements:
            for key,line in absorber.lines.items():

                if e==key.split(' ')[0]:
                    line_dict[key]=line


    line_list = line_dict.values()

    '''
    for line in line_list:

        if custom:
            line.find_mcmc_microlines()

        line.update_mcmc_microlines()

        for microline in line.mcmc_microlines:

            #microline.actual_ew_func()
            microline.calc_linear_N()'''


    #_________________________________________________________________________________________


    #establish mcmc_lines
    mcmc_lines=[]

    #line=line_dict.get('MgII 2796.355099')

    def get_n_microlines(line):
            return len(line.mcmc_microlines)
    
    for line in line_list:
            line.find_mcmc_microlines()
    
    if custom:
        strong_line=max(line_list, key= get_n_microlines)
    else:
        strong_line=line_dict.get('MgII 2796.355099')

    '''
    if line is None:
        line=line_dict.get('MgII 2803.5322972')
    
    if line is None:
        return "oops"
    #line_ind=line_list.index(line)'''

    for microline in strong_line.mcmc_microlines:

        mcmc_lines.append(mcmc_line(microline,elements))

        #if microline.is_saturated:
        #    if microline.saturation_direction == 'right':
        #        mcmc_lines.append(mcmc_line(microline,elements,'right'))

        #   elif microline.saturation_direction == 'left':
        #        mcmc_lines.append(mcmc_line(microline,elements,'left'))

    #so this makes n number of lines, which represent different microlines in mgII that are present
    #next we need to go through and add lines which correspond to that z of the microline of other elments
    '''
    for line in line_list:

        element=line.name.split(' ')[0]

        for microline in line.mcmc_microlines:

            microline.mcmc_z=(microline.center-line.suspected_line)/line.suspected_line
            microline.mcmc_vel=redshift_to_velocity(microline.mcmc_z)
            microline.mcmc_logN=microline.logN
            microline.mcmc_b=(len(microline.wavelength))

            for mcmc_line_obj in mcmc_lines:
                if mcmc_line_obj.is_within(microline.mcmc_z):
                    mcmc_line_obj.add_line(element,microline)'''


    for line in line_list:

        element=line.name.split(' ')[0]
        line.mcmc_microlines=[]

        for mcmc_line_obj in mcmc_lines:

            wav,flux,er=line.give_data(mcmc_line_obj.z_range[0],mcmc_line_obj.z_range[1])

            temp_micro=MicroAbsorptionLine(wav,flux,er,0,1)

            temp_micro.suspected_line=line.suspected_line
            temp_micro.z=line.z
            temp_micro.f=line.f
            temp_micro.gamma=line.gamma

            temp_micro.calc_linear_N()

            temp_micro.mcmc_z=(temp_micro.peak-line.suspected_line)/line.suspected_line
            temp_micro.mcmc_vel=redshift_to_velocity(temp_micro.mcmc_z)
            temp_micro.mcmc_logN=temp_micro.logN
            temp_micro.mcmc_b=(len(temp_micro.wavelength))

            mcmc_line_obj.add_line(element,temp_micro)
            line.mcmc_microlines.append(temp_micro)






    
    #so now all the mcmc_lines have all the relevant microlines from every element
    #next we want the mcmc_lines to make their own initial params list

    initial_guesses=[]

    for line in mcmc_lines:

        initial_guesses.extend(line.export_params())

    param_list = [line.export_params() for line in mcmc_lines]
    initial_guesses_2d=np.array(param_list)

    print('initial guesses before creep')
    print(initial_guesses_2d)


    for i,microline in enumerate(strong_line.mcmc_microlines):

        if microline.is_saturated:
            if microline.saturation_direction == 'right':
                print('creeping right')
                initial_guesses,mcmc_lines=creep(initial_guesses,i,line_dict,elements,mcmc_lines,direction='right')

            elif microline.saturation_direction == 'left':
                print('creeping left')
                initial_guesses,mcmc_lines=creep(initial_guesses,i,line_dict,elements,mcmc_lines,direction='left')

    print('initial guesses after creep')
    print(initial_guesses)

    initial_guesses=optimize_params(initial_guesses,line_dict,elements,mcmc_lines)

    print('initial guesses after optimization')
    print(initial_guesses)

    #_________________________________________________________________________________________  
    #plot all initial guesses

    plot_fits(initial_guesses,line_dict,elements,mcmc_lines,'initial/initial_guesses')
    
    #_________________________________________________________________________________________ 
    #mcmc
    

    initial_guesses = np.array(initial_guesses)

    ndim = len(initial_guesses)
    percent_off = 0.0001
    # Generate the position array
    #pos = initial_guesses + (percent_off * initial_guesses * np.random.uniform(-1, 1, (nwalkers, ndim)))
    pos = initial_guesses + np.random.normal(0, percent_off * np.abs(initial_guesses), size=(nwalkers, ndim))
    
    #12 max
    #5x faster!
    num_processes = os.cpu_count()
    with Pool(num_processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_multi_probability, args=(line_dict, elements, mcmc_lines), pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

    # Analyzing the results
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    median_params = np.median(flat_samples,axis=0)

    log_probs = sampler.get_log_prob(discard=100, thin=15, flat=True)
    map_params = flat_samples[np.argmax(log_probs)]

    save_object(map_params,'static/Data/multi_mcmc/final/map_params.pkl')

    #_________________________________________________________________________________________  
    #plot all fits

    plot_fits(map_params,line_dict,elements,mcmc_lines,'final/final_models')


    #_________________________________________________________________________________________  
    #corner plot



    for i, mcmc_line_obj in enumerate(mcmc_lines):

        labels=[]
        labels.extend([f'velocity'])
        for e in elements:
            labels.extend([f'LogN_{e}',f'b_{e}'])

        # Filter samples and labels
        filtered_samples = []

        for j, label in enumerate(labels):
            # Check if the maximum absolute value of the parameter samples is above the threshold
            filtered_samples.append(flat_samples[:, j])

        # Generate the corner plot with filtered data
        figure = corner.corner(
            np.array(filtered_samples).T,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84], 
            show_titles=True, 
            title_kwargs={"fontsize": 12}
        )

        plt.savefig(f"/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/final/mcmc_corner_{i}.png")
        plt.clf()

    #_________________________________________________________________________________________  
    #trace plot

    threshold = .01

    def plot_trace(sampler, param_names, threshold=1.0, filter_below_threshold=False):
        nwalkers, nsteps, ndim = sampler.chain.shape

        # Determine which parameters to plot based on the threshold
        if filter_below_threshold:
            filtered_indices = []
            filtered_param_names = []
            for i, name in enumerate(param_names):
                # Flatten the parameter samples and check if max is above the threshold
                param_samples = sampler.chain[:, :, i].flatten()
                if np.max(np.abs(param_samples)) >= threshold:
                    filtered_indices.append(i)
                    filtered_param_names.append(name)
        else:
            filtered_indices = list(range(ndim))
            filtered_param_names = param_names

        # Plot only the filtered parameters
        fig, axes = plt.subplots(len(filtered_indices), figsize=(10, 3 * len(filtered_indices)), sharex=True)
        
        if len(filtered_indices) == 1:
            axes = [axes]

        for ax, i in zip(axes, filtered_indices):
            for j in range(nwalkers):
                ax.plot(sampler.chain[j, :, i], "k", alpha=0.3)
            ax.set_xlim(0, nsteps)
            ax.set_ylabel(filtered_param_names[filtered_indices.index(i)])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("Step number")
        plt.tight_layout(h_pad=0)
        plt.savefig("/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/final/mcmc_trace.png")
        plt.clf()

    plot_trace(sampler, labels, threshold=threshold, filter_below_threshold=True)

    #__________________________________________________________________________________________
    #make table like his plot

    df = summarize_params(flat_samples, labels, elements, mcmc_lines,'final/mcmc_results')
    #df.to_csv('multi_mcmc_results.csv')
    #df.to_latex(index=False, column_format='cccccc')

    #__________________________________________________________________________________________
    #save the chain
    chain=sampler.get_chain()
    np.save('static/Data/multi_mcmc/chain.npy', chain)  # Saves the chain

    save_object(line_dict,'static/Data/multi_mcmc/final/line_dict.pkl')
    
    for i,object in enumerate(mcmc_lines):
        save_object(i,f'static/Data/multi_mcmc/final/mcmc_lines{i}.pkl')




def run_mcmc(absorber,element, nsteps=1000, nwalkers=250, custom=False):

    #data collection
    AtomDB=read_atomDB()

    line_list = [line for key, line in absorber.lines.items() if key.split(' ')[0]==element]

    for line in line_list:
        line.calculate_N()
        line.update_line_attributes()

    #--------------------------------------------------------------------------------------------------------------------------------------------
    #Preparing for mcmc

    def get_n_microlines(line):
            return len(line.microLines)

    '''        def get_logN_value(absorption_line):
            return absorption_line.log_N'''
        
    line_to_use=max(line_list, key=get_n_microlines)
    line_to_use_ind=line_list.index(line_to_use)

    #find peaks
    #peaks,properties=find_peaks(-line_to_use.flux+1,height=1*np.std(line_to_use.flux))
    peaks=[]
    for i in line_to_use.microLines:
        peaks.append(i.peak_ind)

    initial_guesses=[] #list of param items for each peak

    #with microlines
    for i,microline in enumerate(line_to_use.microLines):
        guess_z=(microline.peak-line_to_use.suspected_line)/line_to_use.suspected_line
        guess_logN=microline.log_N
        guess_b=(len(microline.wavelength))
        
        initial_guesses.extend([guess_z, guess_logN, guess_b]) #z,LogN,b


    #plot initial guess
    model_list=total_model(initial_guesses,line_list,AtomDB,element)
    fig, axs = plt.subplots(len(line_list), 1,figsize=(10,5*len(line_list)),sharex=True)

    if len(line_list)==1:
        axs=[axs]

    c = 299792.458  # Speed of light in km/s

    mid=int((len(initial_guesses)//3)/2)*3
    z=initial_guesses[mid]

    for i,model in enumerate(model_list):

        center_wavelength=(1+z)*line_list[i].suspected_line
        velocity = (line_list[i].wavelength-center_wavelength)/center_wavelength*c

        axs[i].step(velocity,line_list[i].flux, 'k', label='Observed Flux', where='mid')
        axs[i].step(velocity,line_list[i].errors, c='purple', label='Error', where='mid')

        axs[i].step(velocity,model,c='red',label="Model",where='mid')

        axs[i].set_ylabel('Flux')

        for j,microline in enumerate(line_to_use.microLines):
            if i==line_to_use_ind:
                axs[i].axvline(velocity[microline.peak_ind])
                axs[i].text(velocity[microline.peak_ind],.2,f"{initial_guesses[j*3]:.5f}\n{initial_guesses[j*3+1]:.2f}\n{initial_guesses[j*3+2]:.2f}")
        
    axs[-1].set_xlabel('Velocity (km/s)')  # Label only the bottom plot's x-axis

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
    fig, axs = plt.subplots(len(line_list), 1,figsize=(10,5*len(line_list)),sharex=True)
    model_list = total_model(map_params,line_list,AtomDB,element)

    if len(line_list)==1:
        axs=[axs]

    c = 299792.458  # Speed of light in km/s

    mid=int((len(initial_guesses)//3)/2)*3
    z=initial_guesses[mid]

    num_peaks = len(map_params) // 3

    for i,model in enumerate(model_list):

        center_wavelength=(1+z)*line_list[i].suspected_line
        velocity = (line_list[i].wavelength-center_wavelength)/center_wavelength*c

        axs[i].step(velocity,line_list[i].flux, 'k', label='Observed Flux', where='mid')
        axs[i].step(velocity,line_list[i].errors, c='purple', label='Error', where='mid')


        axs[i].step(velocity,model,c='red',label="Model",where='mid')

        axs[i].set_ylabel('Flux')

        for j in range(num_peaks):

            peak_z, logN, b = map_params[3*j:3*j+3]

            peak_wavelength=(1+peak_z)*line_list[i].suspected_line
            velocity = (peak_wavelength-center_wavelength)/center_wavelength*c

            axs[i].axvline(velocity,c="black")
            axs[i].text(velocity+5,.2,f"Z={initial_guesses[j*3]:.5f}\nLogN={initial_guesses[j*3+1]:.2f}\nb={initial_guesses[j*3+2]:.2f}",fontsize=8)
        
    axs[0].legend()

    plt.xlim((-200,200))

    axs[-1].set_xlabel('Velocity (km/s)')  # Label only the bottom plot's x-axis

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

def run_mcmc(absorber,element, nsteps=1000, nwalkers=250, custom=False):

    #data collection
    AtomDB=read_atomDB()

    line_list = [line for key, line in absorber.lines.items() if key.split(' ')[0]==element]

    for line in line_list:
        line.calculate_N()
        line.update_line_attributes()

    #--------------------------------------------------------------------------------------------------------------------------------------------
    #Preparing for mcmc

    def get_logN_value(absorption_line):
        return absorption_line.log_N

    line_to_use=max(line_list, key=get_logN_value)
    line_to_use_ind=line_list.index(line_to_use)

    #find peaks
    #peaks,properties=find_peaks(-line_to_use.flux+1,height=1*np.std(line_to_use.flux))
    peaks=[]
    for i in line_to_use.microLines:
        peaks.append(i.peak_ind)
    '''
    # Calculate the bounds for each peak's equivalent width
    bounds = []
    for peak in peaks:
        left = peak
        while left > 0 and line_to_use.flux[left] < line_to_use.flux[left - 1]:
            left -= 1
        right = peak
        while right < len(line_to_use.flux) - 1 and line_to_use.flux[right] < line_to_use.flux[right + 1]:
            right += 1
        bounds.append((left, right))'''

    initial_guesses=[] #list of param items for each peak
    '''
    for i,peak in enumerate(peaks):
        guess_z=(line_to_use.wavelength[peak]-line_to_use.suspected_line)/line_to_use.suspected_line
        wave_zone=line_to_use.wavelength[bounds[i][0]:bounds[i][1]]
        flux_zone=line_to_use.flux[bounds[i][0]:bounds[i][1]]
        guess_logN=find_N(wave_zone,flux_zone,guess_z,line_to_use.f,line_to_use.suspected_line)
        guess_b=(bounds[i][1]-bounds[i][0])
        
        initial_guesses.extend([guess_z, guess_logN, guess_b]) #z,LogN,b'''

    #with microlines
    for i,microline in enumerate(line_to_use.microLines):
        guess_z=(microline.peak-line_to_use.suspected_line)/line_to_use.suspected_line
        guess_logN=microline.log_N
        guess_b=(len(microline.wavelength))
        
        initial_guesses.extend([guess_z, guess_logN, guess_b]) #z,LogN,b

    #plot initial guess
    model_list=total_model(initial_guesses,line_list,AtomDB,element)
    fig, axs = plt.subplots(len(line_list), 1,figsize=(10,5*len(line_list)),sharex=True)

    if len(line_list)==1:
        axs=[axs]

    c = 299792.458  # Speed of light in km/s

    mid=int((len(initial_guesses)//3)/2)*3
    z=initial_guesses[mid]

    for i,model in enumerate(model_list):

        center_wavelength=(1+z)*line_list[i].suspected_line
        velocity = (line_list[i].wavelength-center_wavelength)/center_wavelength*c

        axs[i].step(velocity,line_list[i].flux, 'k', label='Observed Flux', where='mid')
        axs[i].step(velocity,line_list[i].errors, c='purple', label='Error', where='mid')

        axs[i].step(velocity,model,c='red',label="Model",where='mid')

        axs[i].set_ylabel('Flux')

        for j,microline in enumerate(line_to_use.microLines):
            if i==line_to_use_ind:
                axs[i].axvline(velocity[microline.peak_ind])
                axs[i].text(velocity[microline.peak_ind],.2,f"{initial_guesses[j*3]:.5f}\n{initial_guesses[j*3+1]:.2f}\n{initial_guesses[j*3+2]:.2f}")
        
    axs[-1].set_xlabel('Velocity (km/s)')  # Label only the bottom plot's x-axis

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
    fig, axs = plt.subplots(len(line_list), 1,figsize=(10,5*len(line_list)),sharex=True)
    model_list = total_model(map_params,line_list,AtomDB,element)

    if len(line_list)==1:
        axs=[axs]

    c = 299792.458  # Speed of light in km/s

    mid=int((len(initial_guesses)//3)/2)*3
    z=initial_guesses[mid]

    num_peaks = len(map_params) // 3

    for i,model in enumerate(model_list):

        center_wavelength=(1+z)*line_list[i].suspected_line
        velocity = (line_list[i].wavelength-center_wavelength)/center_wavelength*c

        axs[i].step(velocity,line_list[i].flux, 'k', label='Observed Flux', where='mid')
        axs[i].step(velocity,line_list[i].errors, c='purple', label='Error', where='mid')


        axs[i].step(velocity,model,c='red',label="Model",where='mid')

        axs[i].set_ylabel('Flux')

        for j in range(num_peaks):

            peak_z, logN, b = map_params[3*j:3*j+3]

            peak_wavelength=(1+peak_z)*line_list[i].suspected_line
            velocity = (peak_wavelength-center_wavelength)/center_wavelength*c

            axs[i].axvline(velocity,c="black")
            axs[i].text(velocity+5,.2,f"Z={initial_guesses[j*3]:.5f}\nLogN={initial_guesses[j*3+1]:.2f}\nb={initial_guesses[j*3+2]:.2f}",fontsize=8)
        
    axs[0].legend()

    plt.xlim((-200,200))

    axs[-1].set_xlabel('Velocity (km/s)')  # Label only the bottom plot's x-axis

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

def continue_mcmc(nsteps,nwalkers):
    
    """Continue MCMC from the last chain for a given number of steps."""

    #load previous items
    chain=np.load('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/chain.npy')
    line_dict=load_object('static/Data/multi_mcmc/final/line_dict.pkl')

    mcmc_lines=[]
    files = os.listdir('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/final')
    filtered_files = [file for file in files if "mcmc_lines" in file]
    for i,object in enumerate(filtered_files):
        mcmc_lines.append(load_object(f'static/Data/multi_mcmc/final/mcmc_lines{i}.pkl'))

    elements_dict={}
    for key,value in line_dict.items():
        elements_dict[key.split(' ')[0]]=value
    elements=elements_dict.keys()

    # Get the last positions of the walkers
    '''
    nwalkers, _, ndim = chain.shape
    pos = chain[:, -1, :]

    flat_samples = pos
    median_params = np.median(flat_samples,axis=0)

    log_probs = sampler.get_log_prob(discard=100, thin=15, flat=True)
    map_params = flat_samples[np.argmax(log_probs)]'''

    map_params=load_object('static/Data/multi_mcmc/final/map_params.pkl')

    ndim = len(map_params)
    percent_off = 0.0001
    pos = map_params + np.random.normal(0, percent_off * np.abs(map_params), size=(nwalkers, ndim))

    # Run MCMC for additional steps
    num_processes = os.cpu_count()
    with Pool(num_processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_multi_probability, args=(line_dict, elements, mcmc_lines), pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

    # Analyzing the results
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    median_params = np.median(flat_samples,axis=0)

    log_probs = sampler.get_log_prob(discard=100, thin=15, flat=True)
    map_params = flat_samples[np.argmax(log_probs)]

    #_________________________________________________________________________________________  
    #plot all fits

    plot_fits(map_params,line_dict,elements,mcmc_lines,'final/final_models')


    #_________________________________________________________________________________________  
    #corner plot

    #threshold = .01

    #labels=[]
    for i,mcmc_line_obj in enumerate(mcmc_lines):
        labels=[]
        labels.extend([f'Component_{i}_velocity'])
        for e in elements:
            labels.extend([f'Component_{i}_{e}_LogN',f'Component_{i}_{e}_b'])

        figure = corner.corner(
        flat_samples[:, i], 
        labels=labels,
        quantiles=[0.16, 0.5, 0.84], 
        show_titles=True, 
        title_kwargs={"fontsize": 12}
        )

        plt.savefig("/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/final/mcmc_corner_component_{i}.png")
        plt.clf()

    # Filter samples and labels
    '''filtered_samples = []
    filtered_labels = []

    for i, label in enumerate(labels):
        # Check if the maximum absolute value of the parameter samples is above the threshold
        if np.max(np.abs(flat_samples[:, i])) >= threshold:
            filtered_samples.append(flat_samples[:, i])
            filtered_labels.append(label)

    # Convert the filtered samples to the correct shape for corner plot
    filtered_samples = np.array(filtered_samples).T

    # Generate the corner plot with filtered data
    figure = corner.corner(
        filtered_samples, 
        labels=filtered_labels,
        quantiles=[0.16, 0.5, 0.84], 
        show_titles=True, 
        title_kwargs={"fontsize": 12}
    )

    plt.savefig("/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/final/mcmc_corner.png")
    plt.clf()'''

    #_________________________________________________________________________________________  
    #trace plot

    threshold=.1

    def plot_trace(sampler, param_names, threshold=1.0, filter_below_threshold=False):
        nwalkers, nsteps, ndim = sampler.chain.shape

        # Determine which parameters to plot based on the threshold
        if filter_below_threshold:
            filtered_indices = []
            filtered_param_names = []
            for i, name in enumerate(param_names):
                # Flatten the parameter samples and check if max is above the threshold
                param_samples = sampler.chain[:, :, i].flatten()
                if np.max(np.abs(param_samples)) >= threshold:
                    filtered_indices.append(i)
                    filtered_param_names.append(name)
        else:
            filtered_indices = list(range(ndim))
            filtered_param_names = param_names

        # Plot only the filtered parameters
        fig, axes = plt.subplots(len(filtered_indices), figsize=(10, 3 * len(filtered_indices)), sharex=True)
        
        if len(filtered_indices) == 1:
            axes = [axes]

        for ax, i in zip(axes, filtered_indices):
            for j in range(nwalkers):
                ax.plot(sampler.chain[j, :, i], "k", alpha=0.3)
            ax.set_xlim(0, nsteps)
            ax.set_ylabel(filtered_param_names[filtered_indices.index(i)])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("Step number")
        plt.tight_layout(h_pad=0)
        plt.savefig("/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/final/mcmc_trace.png")
        plt.clf()

    plot_trace(sampler, labels, threshold=threshold, filter_below_threshold=True)

    #__________________________________________________________________________________________
    #make table like his plot

    df = summarize_params(flat_samples, labels, elements, mcmc_lines,'final/mcmc_results')
    #df.to_csv('multi_mcmc_results.csv')
    #df.to_latex(index=False, column_format='cccccc')

    #__________________________________________________________________________________________
    #save the chain
    chain=sampler.get_chain()
    np.save('static/Data/multi_mcmc/chain.npy', chain)  # Saves the chain

    save_object(line_dict,'static/Data/multi_mcmc/final/line_dict.pkl')
    
    for i,object in enumerate(mcmc_lines):
        save_object(i,f'static/Data/multi_mcmc/final/mcmc_lines{i}.pkl')

#____________________________________________________________________________________________________________
#initial param optimization with lmfit

def objective_function(params, line_dict, elements, mcmc_lines):
    """
    Calculate the total chi-squared for the model parameters.
    
    :param params: lmfit.Parameters object containing the model parameters
    :param line_dict: Dictionary of line data
    :param elements: List of elements involved
    :param mcmc_lines: List of MCMC line objects
    :return: Total chi-squared as a scalar value
    """
    # Convert lmfit.Parameters to a list of parameter values
    param_values = [params.get(key).value for key in params]

    #print('params')
    #print(param_values)
    
    # Calculate the model
    model = total_multi_model(param_values, line_dict, elements, mcmc_lines)
    
    residuals = []
    
    for key, line in line_dict.items():
        obs_flux = line.MgII_flux
        model_flux = model.get(key, np.zeros_like(obs_flux))
        errors = line.MgII_errors
        
        # Compute residuals
        current_residuals = (obs_flux - model_flux) / errors
        residuals.extend(current_residuals)
    
    # Debug: Check the computed residuals
    #print(f"Residuals: {residuals[:10]}")  # Print the first 10 residuals for checking
    
    return np.array(residuals)


        
def optimize_params(initial_params,line_dict,elements,mcmc_lines):

    from lmfit import Parameters, minimize, Minimizer

    num_params_per_line = 1 + (2 * len(elements))
    reshaped_array = np.array(initial_params).reshape(-1, num_params_per_line)

    params = Parameters()

    for i,line_params in enumerate(reshaped_array):

        params.add(f'Component_{i}_velocity', line_params[0], min=0.0,max=redshift_to_velocity(2))

        for j,e in enumerate(elements):

            params.add(f'Component_{i}_{e}_LogN',line_params[(2*j)+1],min=0.0,max=15.0)
            params.add(f'Component_{i}_{e}_b',line_params[(j*2)+2],min=0.0,max=15.0)

    #print('print')
    #print(params)



    #for i, value in enumerate(initial_params):

    #    params.add(f'param_{i}',value=value)

    #print(help(minimize))

    result = minimize(objective_function,params,args=(line_dict,elements,mcmc_lines),nan_policy='omit')
    #max_nfev=50000
    #print(help(result))


    #minimizer = Minimizer(objective_function, params, args=(line_dict,elements,mcmc_lines),nan_policy='omit')
    #result = minimizer.minimize(max_nfev=10000, ftol=1e-10, xtol=1e-10)
    #method='leastsq',

    optimized_params = [result.params[key].value for key in result.params]

    #print('optimized')
    #print(optimized_params)

    return optimized_params