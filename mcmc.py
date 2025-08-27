#mcmc but both lines are being compared and model is being created every step

import emcee
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import corner
import os
import pickle
from multiprocessing import Pool
import pandas as pd

# Import your specific functions
from essential_functions import read_atomDB
from mcmc_functions import find_N, calctau, convolve_flux,mcmc_line, summarize_params,velocity_to_redshift,redshift_to_velocity,parse_statuses,rebuild_full_params,read_atomic_mass,rebuild_full_samples,plot_trace,build_full_chain
from AbsorptionLine import MicroAbsorptionLine

e = 4.8032e-10 # electron charge in stat-coulumb
m_e = 9.10938e-28 # electron mass
c = 2.9979e10 # cm/s
c_As = 2.9979e18
c_kms = 2.9979e5
k = 1.38065e-16 # erg/K


def plot_fits(params, line_dict, elements, mcmc_lines,file_name,chain_review=False,show_components=False):

    import smplotlib

    c = 3e5
    vel_window=600

    num_params_per_line = 1 + 2 * len(elements)
    param_list_2d = np.array(params).reshape(-1, num_params_per_line)

    if show_components:
        models, component_models = total_multi_model(params, line_dict, elements, mcmc_lines, convolve_data=True, high_resolution=True, extra=True, individual_components=True)
    else:
        models=total_multi_model(params,line_dict,elements,mcmc_lines,convolve_data=True,high_resolution=True,extra=True)
    standard_models=total_multi_model(params,line_dict,elements,mcmc_lines,convolve_data=True)

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
    reference_z=load_object('static/Data/multi_mcmc/initial/ref_z.pkl')
    reference_velocity = redshift_to_velocity(reference_z)

    max_vel = np.max(np.abs(strongest_line.velocity))
    #if max_vel < 200:
    #    vel_window = 200
    if max_vel < 400:
        vel_window = 400
    elif max_vel < 600:
        vel_window = 600
    else:
        vel_window = 800


    fig, axs = plt.subplots(len(line_dict.values()), 1, figsize=(5, 3 * len(line_dict.values())), squeeze=False,sharex=True,sharey=True)
    axs_flat = axs.ravel()

    fig.text(0, 0.5, 'Normalized Flux', va='center', rotation='vertical', fontsize=20)

    for i, name in enumerate(line_dict.keys()):

        line=line_dict.get(name)

        #line.store_model(np.linspace(line.MgII_wavelength[0],line.MgII_wavelength[-1],len(line.MgII_wavelength)*10), models.get(name))

        ax=axs_flat[i]

        #chi squared
        obs_flux = line.MgII_flux
        model_flux = standard_models[name]
        errors = np.sqrt(line.MgII_errors)

        #lmfit chi squared
        residuals = (obs_flux - model_flux) / errors
        chi_squared = np.sum(residuals**2)
        ndof = len(obs_flux) - len(mcmc_lines) * (2 * len(elements) + 1)
        reduced_chi_squared = chi_squared / ndof if ndof > 0 else np.nan

        # Calculate chi-squared and reduced chi-squared
        #chi_squared = np.sum(((obs_flux - model_flux) / errors) ** 2)
        #degrees_of_freedom = len(obs_flux) - (len(mcmc_lines)*3)
        #reduced_chi_squared = chi_squared / degrees_of_freedom if degrees_of_freedom != 0 else 0

        #ax.text(0.7, 0.2, f"$\chi^2_{{red}}={reduced_chi_squared:.2f}$", transform=ax.transAxes)

        #actual plot
        reference_microline=(reference_z+1)*line.suspected_line

        full_velocity = line.extra_velocity
        velocity =  line.MgII_velocity

        
        #ax.step(full_velocity, line.extra_flux, where='mid', label=f"Flux", color="black")
        #ax.step(full_velocity, line.extra_errors, where='mid', label="Error", color="cyan")
        plot_flux_with_mask(ax, line.extra_velocity, line.extra_flux, line.extra_errors, line.masked_regions)


        #ax.step(velocity,standard_models.get(name), where='mid', label=f"Model", color="purple")

        high_res_full_velocity=np.linspace(full_velocity[0],full_velocity[-1],len(full_velocity)*10)

        line.store_model(high_res_full_velocity, models.get(name),reduced_chi_squared)

        #ax.step(high_res_full_velocity, models.get(name), where='mid', label=f"Model", color="red",linewidth=1)

        # Plot individual components
        if show_components:
            ax.step(high_res_full_velocity, models.get(name), where='mid', label=f"Model", color="red",linewidth=1)
            colors = plt.cm.viridis(np.linspace(0, 1, len(component_models.get(name, []))))
            for idx, component_flux in enumerate(component_models.get(name, [])):
                ax.plot(high_res_full_velocity, component_flux, color=colors[idx], linestyle='--', alpha=.7, linewidth=1.2)

        else:
            ax.step(high_res_full_velocity, models.get(name), where='mid', label=f"Model", color="red",linewidth=1)



        for i,line_params in enumerate(param_list_2d):

            #z=velocity_to_redshift(line_params[0])
            #wavelength = line.suspected_line * (1+z)
            #velocity =  (wavelength - reference_microline) / reference_microline * c

            ax.vlines(line_params[0], ymin=1.2,ymax=1.3,color='blue')

            #if 'initial' in file_name:
            #    ax.vlines(mcmc_lines[i].vel_range[0],ymin=0,ymax=1)
            #    ax.vlines(mcmc_lines[i].vel_range[1],ymin=0,ymax=1)

        if 'initial' in file_name:
            for i,line in enumerate(mcmc_lines):
                ax.axvline(line.vel_range[0], color='green', alpha=.3,linewidth=.5)
                ax.axvline(line.vel_range[1], color='red', alpha=.3, linewidth=.5)

                ax.axvspan(line.vel_range[0], line.vel_range[1], color='gray', alpha=0.1, label='Velocity range')

                if chain_review==False:
                    ax.text(param_list_2d[i][0], 1.35, f"|{i}", fontsize=6)
                else:
                    ax.text(param_list_2d[i][0], 1.35, f"|{i+1}", fontsize=6)
        
        ax.text(.7,0.1,f'{name.split(" ")[0]} {int(np.floor(float(name.split(" ")[1])))}',transform=ax.transAxes)
        ax.set_xlim(-vel_window, vel_window)

    # Label axes and configure layout
    axs_flat[-1].set_xlabel('Relative Velocity (km/s)', fontsize=12)

    plt.subplots_adjust(hspace=0)

    #plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for title
    if chain_review==False:
        plt.savefig(f"static/Data/multi_mcmc/{file_name}.png")
    else:
        plt.savefig(f"static/chain_review/{file_name}.png")

def load_object(filename):
    with open(filename, 'rb') as inp:  # Open the file in binary read mode
        return pickle.load(inp)  # Return the unpickled object
    
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Open the file in binary write mode
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)  # Pickle the object and write to file # Pickle the object and write to file


def total_multi_model(params, line_dict, elements, mcmc_lines, convolve_data=True, high_resolution=False, chi2=False, extra=False, individual_components=False):

    params_per_microline=(2*len(elements))+1
    param_list_2d = np.array(params).reshape(-1, params_per_microline)

    models = {}
    if individual_components:
        component_models = {key: [] for key in line_dict.keys()}

    if chi2:
        chi_value=0

    for key,line in line_dict.items():

        if high_resolution:
            if extra:
                #wavelength=np.linspace(line.extra_wavelength[0],line.extra_wavelength[-1],len(line.extra_wavelength)*10)
                velocity=np.linspace(line.extra_velocity[0],line.extra_velocity[-1],len(line.extra_velocity)*10)
            else:
                velocity=np.linspace(line.MgII_velocity[0],line.MgII_velocity[-1],len(line.MgII_velocity)*10)

        else:
            velocity=line.MgII_velocity

        models[key]=np.ones_like(velocity)

        for i,line_params in enumerate(param_list_2d):

            velocity_param=line_params[0]

            for j,e in enumerate(elements):
                if e == key.split(' ')[0]:

                    logN=line_params[(j*2)+1]
                    b=line_params[(j*2)+2]

                    tau = calctau(velocity, velocity_param, logN, b, line)
                    models[key] *= tau
                    if individual_components:
                        component_models[key].append(tau)

        if convolve_data:
            models[key] = convolve_flux(velocity, models[key], line.fwhm)
            if individual_components:
                component_models[key] = [convolve_flux(velocity, comp, line.fwhm) for comp in component_models[key]]

        if chi2:
            obs_flux = line.MgII_flux
            model_flux = models[key]
            errors = np.sqrt(line.MgII_errors)

            # Calculate chi-squared and reduced chi-squared
            chi_value+= np.sum(((obs_flux - model_flux) / errors) ** 2)

    if chi2:
        return models,chi_value
    elif individual_components:
        return models, component_models
    else:
        return models

def log_multi_likelihood(params, lines, elements, mcmc_lines,masked_regions):

    models = total_multi_model(params, lines, elements, mcmc_lines)

    chi2 = 0
    #logL=0

    for key, line in lines.items():
        model = models.get(key)
        flux = line.MgII_flux
        error = line.MgII_errors
        velocity = line.velocity

        mask = np.ones_like(flux, dtype=bool)
        if masked_regions and key in masked_regions:
            for vmin, vmax in masked_regions[key]:
                mask &= (velocity < vmin) | (velocity > vmax) 

        # Apply mask
        flux = flux[mask]
        error = error[mask]
        model = model[mask]

        chi2 += np.sum(np.log(1 / np.sqrt(2 * np.pi) / error) - (flux - model) ** 2 / (2 * error**2))
        #logL = -0.5 * np.sum(((flux - model)/error)**2)

    #return logL
    return chi2


def log_multi_prior(params,elements,mcmc_lines):

    num_params_per_line = 1 + 2 * len(elements)
    param_list_2d = np.array(params).reshape(-1, num_params_per_line)

    for i,line in enumerate(mcmc_lines):

        line_params=param_list_2d[i]

        velocity=line_params[0]

        vel_range=line.vel_range

        for j,e in enumerate(elements):

            logN=line_params[(j*2)+1]
            b=line_params[(j*2)+2]

            if not (vel_range[0] < velocity < vel_range[1] and .1 < b < 50 and 0 < logN < 25):
                return -np.inf
            
    return 0.0

def log_multi_probability(free_values, line_dict, elements, mcmc_lines,
                          free_indices, fixed_values, anchor_map, shape,masked_regions,
                          thermal_map=None, nonthermal_set=None):
    params = rebuild_full_params(free_values, free_indices, fixed_values, anchor_map,
                                 shape, thermal_map, nonthermal_set, elements)

    lp = log_multi_prior(params, elements, mcmc_lines)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_multi_likelihood(params, line_dict, elements, mcmc_lines,masked_regions)

def pre_mcmc(absorber,elements):

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
    
    ref_z=load_object('static/Data/multi_mcmc/initial/ref_z.pkl')
    for line in line_list:
        line.make_velocity(ref_z)

    #_________________________________________________________________________________________


    #establish mcmc_lines
    mcmc_lines=[]

    def get_n_microlines(line):
            return len(line.mcmc_microlines)
    
    for line in line_list:
            line.find_mcmc_microlines()

    #strong_line = max(line_list, key=get_n_microlines)
    

    #strong_line=line_dict.get('MgII 2796.355099')
    strong_line=line_dict.get('MgII 2803.5322972')
    #strong_line=line_dict.get('MgI 2852.96342')
    #strong_line=line_dict.get('FeII 2600.1720322')
    #strong_line=line_dict.get('FeII 2374.4599813')
    #strong_line=line_dict.get('FeII 2586.6492304')
    #strong_line=line_dict.get('FeII 2382.7639122')

    #____________________________________________________________________________

    for microline in strong_line.mcmc_microlines:

        mcmc_lines.append(mcmc_line(microline,elements))


    '''    #double check all mcmc_lines
    filtered_mcmc_lines=[]
    for mcmc_line_obj in mcmc_lines:
        has_data = False
        for line in line_list:
            wav, flux, er, vel = line.give_data(mcmc_line_obj.z_range[0], mcmc_line_obj.z_range[1])
            print()
            if len(wav) > 2:
                has_data = True
                break
        if has_data:
            filtered_mcmc_lines.append(mcmc_line_obj)

    mcmc_lines=filtered_mcmc_lines

    print('mcmc_lines')
    for mcmc_line_obj in mcmc_lines:
        print(mcmc_line_obj.microLines)'''


    for line in line_list:

        element=line.name.split(' ')[0]
        line.mcmc_microlines=[]

        for mcmc_line_obj in mcmc_lines:

            wav,flux,er,vel=line.give_data(mcmc_line_obj.z_range[0],mcmc_line_obj.z_range[1])

            temp_micro=MicroAbsorptionLine(wav,flux,er,0,1)
            temp_micro.take_vel(vel)

            temp_micro.suspected_line=line.suspected_line
            temp_micro.z=line.z
            temp_micro.f=line.f
            temp_micro.gamma=line.gamma

            temp_micro.calc_linear_N()

            temp_micro.mcmc_z=(temp_micro.peak-line.suspected_line)/line.suspected_line
            #temp_micro.mcmc_vel=redshift_to_velocity(temp_micro.mcmc_z)
            temp_micro.mcmc_vel=vel[temp_micro.peak_ind]
            temp_micro.mcmc_logN=temp_micro.logN
            temp_micro.mcmc_b=(len(temp_micro.wavelength))

            mcmc_line_obj.add_line(element,temp_micro)
            line.mcmc_microlines.append(temp_micro)


    initial_guesses=[]

    for line in mcmc_lines:

        initial_guesses.extend(line.export_params())

    param_list = [line.export_params() for line in mcmc_lines]
    initial_guesses_2d=np.array(param_list)

    print('initial guesses')
    print(initial_guesses_2d)

    initial_guesses=optimize_params(initial_guesses,line_dict,elements,mcmc_lines,1000)

    #now do the bic analysis:
    #initial_guesses, mcmc_lines, best_bic=select_best_model_bic(initial_guesses_2d,mcmc_lines,line_dict,elements)

    initial_guesses=optimize_params(initial_guesses,line_dict,elements,mcmc_lines,10000)

    #_________________________________________________________________________________________  
    #plot all initial guesses

    plot_fits(initial_guesses,line_dict,elements,mcmc_lines,'initial/initial_guesses')


    #save all the objects
    save_object(line_dict,'static/Data/multi_mcmc/initial/initial_line_dict.pkl')
    save_object(mcmc_lines,'static/Data/multi_mcmc/initial/initial_mcmc_lines.pkl')
    save_object(elements,'static/Data/multi_mcmc/initial/initial_element_list.pkl')

    #___________________________________________________________________________________________


    return initial_guesses,line_dict

def update_fit(parameters,elements,lmfit_iterations):

    #load neccesary objects
    line_dict=load_object('static/Data/multi_mcmc/initial/initial_line_dict.pkl')
    mcmc_lines=load_object('static/Data/multi_mcmc/initial/initial_mcmc_lines.pkl')

    params=np.array(parameters).flatten().tolist()

    if lmfit_iterations>0:
        print(f'running lmfit for {lmfit_iterations} iterations')
        params=optimize_params(params,line_dict,elements,mcmc_lines,lmfit_iterations)

    #create new plot
    plot_fits(params,line_dict,elements,mcmc_lines,'initial/initial_guesses')

    #save over old params with updated ones
    save_object(params,'static/Data/multi_mcmc/initial/initial_guesses.pkl')

    return params


def mcmc(initial_guesses,statuses, nsteps=1000,nwalkers=250):

    line_dict=load_object('static/Data/multi_mcmc/initial/initial_line_dict.pkl')
    elements=load_object('static/Data/multi_mcmc/initial/initial_element_list.pkl')
    mcmc_lines=load_object('static/Data/multi_mcmc/initial/initial_mcmc_lines.pkl')
    masked_regions = load_object("static/Data/multi_mcmc/initial/masked_regions.pkl")

    #_________________________________________________________________________________________ 
    #mcmc

    initial_guesses = np.array(initial_guesses)
    shape = initial_guesses.shape

    # Parse statuses
    free_indices, fixed_values, anchor_map, thermal_map, nonthermal_set = parse_statuses(statuses, initial_guesses)
    initial_free_values = [initial_guesses[i][j] for (i, j) in free_indices]


    #pos generation
    ndim = len(initial_free_values)
    params_per_line = 1 + 2 * len(elements)
    percent_off=.05

    #chat version
    pos = []
    for _ in range(nwalkers):
        walker_pos = []
        for (i, j) in free_indices:
            base_val = initial_guesses[i][j]
            
            if j == 0:  # velocity column
                vel_range = (mcmc_lines[i].vel_range[0],
                            mcmc_lines[i].vel_range[1])
                sampled_val = np.random.uniform(*vel_range)
            else:  # logN or b
                std = percent_off * abs(base_val) if base_val != 0 else percent_off
                sampled_val = np.random.normal(base_val, std)

            walker_pos.append(sampled_val)

        pos.append(walker_pos)

    pos = np.array(pos)

    #try out autocorrelation
    max_steps = nsteps
    autocorr_estimates = []

    num_processes = os.cpu_count()
    with Pool(num_processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_multi_probability,
                                        args=(line_dict, elements, mcmc_lines,
                                            free_indices, fixed_values, anchor_map, shape, masked_regions,
                                            thermal_map, nonthermal_set),
                                            )
        
    state = sampler.run_mcmc(pos, 100, progress=True)

    for i in range(100, max_steps + 1, 100):
        # Continue sampling
        sampler.run_mcmc(state, 100, progress=True)
        state = sampler.get_last_sample()

        if sampler.iteration < 100:
            continue

        try:
            '''
            # Estimate autocorrelation time
            tau = sampler.get_autocorr_time(tol=0)
            print(f'steps:{sampler.iteration}')
            autocorr_estimates.append(tau)

            # Check convergence: long enough chain and stable tau
            converged = np.all(sampler.iteration > 50 * tau)
            stable_tau = (
                len(autocorr_estimates) > 1
                and np.all(np.abs(autocorr_estimates[-1] - autocorr_estimates[-2]) / tau < 0.01)
            )'''

            chain = sampler.get_chain()  # shape (nwalkers, nsteps, nparams)
            rhat = gelman_rubin(chain)

            print(f'iteration:{sampler.iteration}')
            print("Gelman-Rubin R̂ values:")
            for i, val in enumerate(rhat):
                print(f"Param {i}: R̂ = {val:.3f}")

            gelman_converged = np.all(rhat < 1.05)

            if gelman_converged:
                print(f"Converged at iteration {sampler.iteration}")
                break

        except emcee.autocorr.AutocorrError:
            # Chain too short to estimate tau
            continue
    '''
    #12 max
    #5x faster!
    num_processes = os.cpu_count()
    with Pool(num_processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_multi_probability,
                                        args=(line_dict, elements, mcmc_lines,
                                            free_indices, fixed_values, anchor_map, shape,
                                            thermal_map, nonthermal_set),
                                            )

        sampler.run_mcmc(pos, nsteps, progress=True)'''

    # Analyzing the results
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    median_params = np.median(flat_samples,axis=0)

    log_probs = sampler.get_log_prob(discard=100, thin=15, flat=True)
    map_params = flat_samples[np.argmax(log_probs)]

    #rebuild params 
    map_params = flat_samples[np.argmax(log_probs)]

    map_params = rebuild_full_params(map_params, free_indices, fixed_values, anchor_map,
                                 shape, thermal_map, nonthermal_set, elements)
    
    
    median_params = np.median(flat_samples,axis=0)
    
    median_params = rebuild_full_params(median_params, free_indices, fixed_values, anchor_map,
                                 shape, thermal_map, nonthermal_set, elements)

    shape = initial_guesses.shape

    full_samples = rebuild_full_samples(
        flat_samples=flat_samples,
        free_indices=free_indices,
        fixed_values=fixed_values,
        anchor_map=anchor_map,
        thermal_map=thermal_map,
        nonthermal_set=nonthermal_set,
        elements=elements,
        shape=shape
    )

    full_chain = build_full_chain(
        sampler,
        free_indices=free_indices,
        fixed_values=fixed_values,
        anchor_map=anchor_map,
        thermal_map=thermal_map,
        nonthermal_set=nonthermal_set,
        elements=elements,
        shape=initial_guesses.shape
    )


    save_object(map_params,'static/Data/multi_mcmc/final/initial_guesses.pkl')

    #_________________________________________________________________________________________  
    #plot all fits

    plot_fits(map_params,line_dict,elements,mcmc_lines,'final/final_models')

    plot_fits(median_params,line_dict,elements,mcmc_lines,'final/median_fits')

    plot_fits(median_params,line_dict,elements,mcmc_lines,'final/median_params_indv_comps',show_components=True)

    
    #_________________________________________________________________________________________ 
    #labels

    labels = []
    for i in range(len(statuses)):
        for j in range(len(statuses[0])):
            param_type = (
                "velocity" if j == 0
                else f"LogN_{elements[(j - 1) // 2]}" if (j - 1) % 2 == 0
                else f"b_{elements[(j - 2) // 2]}"
            )
            status = statuses[i][j]
            if status != 'free':
                param_type += f" [{status}]"
            labels.append(param_type)

    #ranges
    ranges = []
    for i in range(full_samples.shape[1]):
        col = full_samples[:, i]
        std = np.std(col)
        if std < 1e-8:
            center = np.mean(col)
            ranges.append((center - 1e-3, center + 1e-3))  # Tiny visual range
        else:
            center = np.mean(col)
            ranges.append((center-(3*np.std(col)), center+(3*np.std(col))))

    ranges=np.array(ranges)



    #_________________________________________________________________________________________
    #corner

    for i,mcmc_line_obj in enumerate(mcmc_lines):

        params_per_microline = (2 * len(elements)) + 1
        param_block = full_samples[:, i * params_per_microline:(i + 1) * params_per_microline]
        range_block = [tuple(r) for r in ranges[i * params_per_microline:(i + 1) * params_per_microline]]

        figure = corner.corner(
            param_block,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            range=range_block
        )
        plt.savefig(f"static/Data/multi_mcmc/final/mcmc_corner_{i}.png")
        plt.clf()


    #_________________________________________________________________________________________
    #trace plot
    try:
        plot_trace(
            np.transpose(full_chain, (1, 0, 2)),
            labels=labels,
            save_path="static/Data/multi_mcmc/final/mcmc_trace.png"
        )
    except:
        pass

    #__________________________________________________________________________________________
    #make table like his plot

    df = summarize_params(full_samples, labels, elements, mcmc_lines,'final/mcmc_results')

    #____________
    #output csv
    from astropy.cosmology import Planck18 as cosmo

    our_2796=line_dict.get('MgII 2796.355099')
    ref_z=load_object('static/Data/multi_mcmc/initial/ref_z.pkl')

    # Convert to velocity relative to galaxy
    vmin = our_2796.velocity[0]
    vmax = our_2796.velocity[-1]


    # Create a list of rows, starting with metadata
    rows = [{
        "Object Name" : str(load_object('object_name.pkl')),
        "Galaxy Z": ref_z,
        "Integration Window": (vmin, vmax),
    }]

    #rows[0]['Projected Distance']=abs(cosmo.comoving_distance(ref_z).to('kpc')-cosmo.comoving_distance(our_2796.suspected_z).to('kpc'))
    rows[0]['Absorber Redshift']=f"{our_2796.suspected_z:.5f}"

    for key,value in line_dict.items():

        ew,ew_error=value.actual_ew_func()

        rows[0][f"{key.split(' ')[0]} {int(np.floor(float(key.split(' ')[1].strip())))} EW"]=f"{(1000*ew):.2f} +- {(1000*ew_error):.2f}"

    params_per_microline = (2 * len(elements)) + 1
    maps_2d = np.array(map_params).reshape(-1, params_per_microline)
    for i, mcmc_line_obj in enumerate(mcmc_lines):
        component=i+1
        param_block = full_samples[:, i * params_per_microline:(i + 1) * params_per_microline]

        # Velocity
        vel_samples = param_block[:, 0]
        vel_p16, vel_p50, vel_p84 = np.percentile(vel_samples, [16, 50, 84])
        rel_median = vel_p50
        rel_low = vel_p50 - vel_p16
        rel_high = vel_p84 - vel_p50

        if i>0:
            new_row={}
            for key,value in rows[0].items():
                new_row[key]=None

        for j, element in enumerate(elements):
            # logN
            logN_samples = param_block[:, (j * 2) + 1]
            logN_p16, logN_p50, logN_p84 = np.percentile(logN_samples, [16, 50, 84])
            logN_low = logN_p50 - logN_p16
            logN_high = logN_p84 - logN_p50

            # b
            b_samples = param_block[:, (j * 2) + 2]
            b_p16, b_p50, b_p84 = np.percentile(b_samples, [16, 50, 84])
            b_low = b_p50 - b_p16
            b_high = b_p84 - b_p50

            if i ==0:
                if j==0:
                    rows[0]['Component']=component
                    rows[0][f'Velocity (km/s)']=f"{rel_median:.1f} (+{rel_high:.2f}/-{rel_low:.2f})"
                    rows[0]["MAP Velocity"]=f"{maps_2d[i][0]:.3f}"

                rows[0][f"{element} LogN"]=f"{logN_p50:.2f} (+{logN_high:.2f}/-{logN_low:.2f})"
                rows[0][f"MAP {element} LogN"]= f"{maps_2d[i][(j*2)+1]:.3f}"
                rows[0][f"{element} b (km/s)"] = f"{b_p50:.1f} (+{b_high:.1f}/-{b_low:.1f})"
                rows[0][f"MAP {element} b (km/s)"] = f"{maps_2d[i][(j*2)+2]:.3f}"

            elif i>0:
                if j==0:
                    new_row['Component']=component
                    new_row[f'Velocity (km/s)']=f"{rel_median:.1f} (+{rel_high:.2f}/-{rel_low:.2f})"
                    new_row["MAP Velocity"]=f"{maps_2d[i][0]:.3f}"

                new_row[f"{element} LogN"]=f"{logN_p50:.2f} (+{logN_high:.2f}/-{logN_low:.2f})"
                new_row[f"MAP {element} LogN"]= f"{maps_2d[i][(j*2)+1]:.3f}"
                new_row[f"{element} b (km/s)"] = f"{b_p50:.1f} (+{b_high:.1f}/-{b_low:.1f})"
                new_row[f"MAP {element} b (km/s)"] = f"{maps_2d[i][(j*2)+2]:.3f}"
        
        if i>0:
            rows.append(new_row)

    #now total it up
    if len(mcmc_lines)!=1:

        new_row={}
        for key,value in rows[0].items():
            new_row[key]=None
        
        new_row['Component']="Total"

        for j, element in enumerate(elements):

            N_values=[]
            for i,mcmc_line in enumerate(mcmc_lines):
                
                N_values.append(10**(float(rows[i].get(f"{element} LogN").split(' ')[0])))

            the_sum=np.sum(np.array(N_values))

            new_row[f"{element} LogN"]=f"{np.log10(the_sum):.2f}"

        rows.append(new_row)

    # Create a dataframe
    df = pd.DataFrame(rows)

    df.to_csv('absorber_data.csv',index=False)

    #__________________________________________________________________________________________
    #save the chain
    chain=sampler.get_chain()
    np.save('static/Data/multi_mcmc/chain.npy', chain)  # Saves the chain

    save_object(line_dict,'static/Data/multi_mcmc/final/line_dict.pkl')
    save_object(mcmc_lines,'static/Data/multi_mcmc/final/mcmc_lines.pkl')

    print("Acceptance fraction:", np.mean(sampler.acceptance_fraction))

    #__________________________________________________________________________________________
    #move all the stuff to new folder
    import shutil 

    object_folder=f"mcmc_outputs/{str(load_object('object_name.pkl'))}"
    os.makedirs(object_folder,exist_ok=True)

    #copy final folder over
    target_folder=f"mcmc_outputs/{str(load_object('object_name.pkl'))}/final"
    os.makedirs(target_folder,exist_ok=True)
    previous_folder='/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/final/'
    for file in os.listdir(previous_folder):
        shutil.copy(previous_folder+file, target_folder)

    #copy initial folder over
    target_folder=f"mcmc_outputs/{str(load_object('object_name.pkl'))}/initial"
    os.makedirs(target_folder,exist_ok=True)
    previous_folder='/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/initial/'
    for file in os.listdir(previous_folder):
        shutil.copy(previous_folder+file, target_folder)

    #copy chain over
    shutil.copy('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/chain.npy',object_folder)

    #copy absorption csv file over too
    shutil.copy('absorber_data.csv',object_folder)
     


def continue_mcmc(nsteps, nwalkers):
    """Continue MCMC from the last chain for a given number of steps, honoring fixed/anchored statuses."""

    # Load data from disk
    prev_chain = np.load('static/Data/multi_mcmc/chain.npy')
    map_params = load_object('static/Data/multi_mcmc/final/map_params.pkl')

    line_dict = load_object('static/Data/multi_mcmc/final/line_dict.pkl')
    mcmc_lines = load_object('static/Data/multi_mcmc/final/mcmc_lines.pkl')
    elements = load_object('static/Data/multi_mcmc/initial/initial_element_list.pkl')
    statuses = load_object('static/Data/multi_mcmc/initial/initial_statuses.pkl')

    # Rebuild structure
    param_list_2d = load_object('static/Data/multi_mcmc/initial/initial_guesses.pkl')
    shape = np.array(param_list_2d).shape

    # Parse status maps
    free_indices, fixed_values, anchor_map, thermal_map, nonthermal_set = parse_statuses(statuses, initial_guesses)

    # Prepare starting positions
    percent_off = 0.0001
    pos = np.array(map_params) + np.random.normal(0, percent_off * np.abs(map_params), size=(nwalkers, len(map_params)))

    # Run MCMC
    with Pool(os.cpu_count()) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            len(map_params),
            log_multi_probability,
            args=(line_dict, elements, mcmc_lines, free_indices, fixed_values, anchor_map, shape),
            pool=pool
        )
        sampler.run_mcmc(pos, nsteps, progress=True)

    # Analyze samples
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    log_probs = sampler.get_log_prob(discard=100, thin=15, flat=True)
    map_params = flat_samples[np.argmax(log_probs)]

    #rebuild params 
    map_params = rebuild_full_params(free_values, free_indices, fixed_values, anchor_map,
                             shape, thermal_map, nonthermal_set, elements)
    full_samples = []
    for free_sample in flat_samples:
        full_vec = rebuild_full_params(free_values, free_indices, fixed_values, anchor_map,
                             shape, thermal_map, nonthermal_set, elements)
        full_samples.append(full_vec)
    full_samples = np.array(full_samples)

    save_object(map_params,'static/Data/multi_mcmc/final/map_params.pkl')

    # Rebuild param list for plot
    full_param_vec = rebuild_full_params(free_values, free_indices, fixed_values, anchor_map,
                             shape, thermal_map, nonthermal_set, elements)
    param_list_2d = np.array(full_param_vec).reshape(shape)

    # Plot fit
    plot_fits(param_list_2d, line_dict, elements, mcmc_lines, 'final/final_models')

    #_________________________________________________________________________________________  
    #corner plot
    params_per_microline = 1 + 2 * len(elements)

    for i, mcmc_line_obj in enumerate(mcmc_lines):
        labels = ['velocity']
        for e in elements:
            labels.extend([f'LogN_{e}', f'b_{e}'])

        # Extract the block of samples for this component
        start = i * params_per_microline
        end = start + params_per_microline
        param_block = full_samples[:, start:end]  # shape: (n_samples, params_per_microline)

        # Create the corner plot for just this component
        figure = corner.corner(
            param_block,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84], 
            show_titles=True, 
            title_kwargs={"fontsize": 12}
        )

        plt.savefig(f"static/Data/multi_mcmc/final/mcmc_corner_{i}.png")
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

    # Summary table
    df = summarize_params(full_samples, labels, elements, mcmc_lines, 'final/mcmc_results')

    # Save updated chain
    chain = sampler.get_chain()
    np.save('static/Data/multi_mcmc/chain.npy', chain)
    save_object(line_dict, 'static/Data/multi_mcmc/final/line_dict.pkl')
    save_object(mcmc_lines,'static/Data/multi_mcmc/final/mcmc_lines.pkl')


def objective_function(params, line_dict, elements, mcmc_lines):

    param_values = [params.get(key).value for key in params]
    
    # Calculate the model
    model = total_multi_model(param_values, line_dict, elements, mcmc_lines)
    
    residuals = []
    
    for key, line in line_dict.items():
        obs_flux = line.MgII_flux
        model_flux = model.get(key)
        errors = line.MgII_errors
        velocity = line.velocity

        mask = np.ones_like(obs_flux, dtype=bool)
        if len(line.masked_regions) > 0:
            for vmin, vmax in line.masked_regions:
                mask &= (velocity < vmin) | (velocity > vmax)

        # Apply mask
        obs_flux = obs_flux[mask]
        model_flux = model_flux[mask]
        errors = errors[mask]

        # Replace zero or near-zero errors to prevent division by zero
        safe_errors = np.where(errors <= 0, 1e-6, errors)

        current_residuals = (obs_flux - model_flux) / safe_errors
        residuals.extend(current_residuals)

    residuals = np.array(residuals)

    return np.sum(np.array(residuals)**2)


        
def optimize_params(initial_params,line_dict,elements,mcmc_lines, max_nfev):

    from lmfit import Parameters, minimize, Minimizer

    num_params_per_line = 1 + (2 * len(elements))
    reshaped_array = np.array(initial_params).reshape(-1, num_params_per_line)

    params = Parameters()

    for i,mcmc_line in enumerate(mcmc_lines):

        line_params=reshaped_array[i]

        min=mcmc_line.vel_range[0]
        max=mcmc_line.vel_range[1]

        if min==max:
            min-=.1

        params.add(f'Component_{i}_velocity', line_params[0], min=min,max=max)

        for j,e in enumerate(elements):

            params.add(f'Component_{i}_{e}_LogN',line_params[(2*j)+1],min=8,max=20.0)
            params.add(f'Component_{i}_{e}_b',line_params[(j*2)+2],min=1,max=20.0)

    result = minimize(objective_function,params,args=(line_dict,elements,mcmc_lines),nan_policy='omit',method='nelder',max_nfev=max_nfev )

    optimized_params = [result.params[key].value for key in result.params]

    return optimized_params

def select_best_model_bic(initial_params, mcmc_lines,line_dict, elements):
    from lmfit import Parameters
    from itertools import combinations

    logN_threshold=11

    # Split into certain and uncertain based on LogN
    certain_indices = [i for i, p in enumerate(initial_params) if p[1] >= logN_threshold]
    uncertain_indices = [i for i, p in enumerate(initial_params) if p[1] < logN_threshold]

    if len(uncertain_indices) == 0:
        print("All components are certain; returning full model only.")
        combo_list = [certain_indices]
        return initial_params,mcmc_lines,0
    else:
        # Create all combinations of uncertain indices
        def all_index_combinations(lst):
            result = []
            n = len(lst)
            for r in range(1, n + 1):
                result.extend(combinations(lst, r))
            return result

        combo_list = []
        for uncertain_combo in all_index_combinations(uncertain_indices):
            full_combo = sorted(certain_indices + list(uncertain_combo))
            combo_list.append(full_combo)

    best_bic = np.inf
    best_result = None
    best_components = None
    best_params = None

    for combo in combo_list:

        parameters=[initial_params[i] for i in combo]
        combo_mcmc_lines = [mcmc_lines[i] for i in combo]

        
        try:
            # Optimize parameters for this component set
            result_params = optimize_params(parameters, line_dict, elements, combo_mcmc_lines,100)

            model_dict = total_multi_model(result_params, line_dict, elements, combo_mcmc_lines)

            chi2 = 0
            n_data = 0

            for key, line in line_dict.items():
                obs_flux = line.MgII_flux
                model_flux = model_dict[key]
                errors = line.MgII_errors
                velocity = line.velocity

                mask = np.ones_like(obs_flux, dtype=bool)
                if len(line.masked_regions)>0:
                    for vmin, vmax in line.masked_regions:
                        mask &= (velocity < vmin) | (velocity > vmax)

                obs_flux = obs_flux[mask]
                model_flux = model_flux[mask]
                errors = errors[mask]

                chi2 += np.sum(((obs_flux - model_flux) / errors) ** 2)
                n_data += len(obs_flux)

            k = len(result_params)
            bic = chi2 + k * np.log(n_data)

            print(f'combo:{combo}')
            print(f'bic:{bic}')

            if bic < best_bic:
                best_bic = bic
                best_mcmc_lines = combo_mcmc_lines
                best_params = result_params

        except Exception as e:
            print(f"Skipping a model due to error: {e}")
            continue

    print(best_params)
    return best_params, best_mcmc_lines, best_bic


def plot_flux_with_mask(ax, velocity, flux, error, masked_regions):

    if len(masked_regions) == 0:
        ax.step(velocity, flux, where='mid', color='black', linestyle='-')
        ax.step(velocity, error, where='mid', color='cyan', linestyle='-')
        return

    mask = np.ones_like(velocity, dtype=bool)

    for vmin, vmax in masked_regions:
        mask &= (velocity < vmin) | (velocity > vmax)

    # Plot unmasked (solid)
    unmasked = ~((velocity >= vmin) & (velocity <= vmax))
    current_mask = mask.copy()

    # Plot unmasked regions
    i = 0
    while i < len(velocity):
        if current_mask[i]:
            start = i
            while i < len(velocity) and current_mask[i]:
                i += 1
            ax.step(velocity[start:i], flux[start:i], where='mid', color='black', linestyle='-')
            ax.step(velocity[start:i], error[start:i], where='mid', color='cyan', linestyle='-')
        else:
            i += 1

    # Plot masked regions (dashed)
    current_mask = ~mask
    i = 0
    while i < len(velocity):
        if current_mask[i]:
            start = i
            while i < len(velocity) and current_mask[i]:
                i += 1
            ax.step(velocity[start-1:i+1], flux[start-1:i+1], where='mid', color='black', linestyle=(0, (2, 2)), linewidth=.5)
            ax.step(velocity[start-1:i+1], error[start-1:i+1], where='mid', color='cyan', linestyle=(0, (2, 2)), linewidth=.5)
        else:
            i += 1

def gelman_rubin(chain):
    """
    Compute Gelman-Rubin R-hat diagnostic for a 3D MCMC chain.
    
    Parameters
    ----------
    chain : np.ndarray
        Shape (nwalkers, nsteps, nparams)
        
    Returns
    -------
    R_hat : np.ndarray
        R-hat value for each parameter
    """
    chain = np.transpose(chain, (1, 0, 2))

    m, n, p = chain.shape

    # Mean per chain (walker)
    chain_means = np.mean(chain, axis=1)
    chain_vars = np.var(chain, axis=1, ddof=1)

    overall_mean = np.mean(chain_means, axis=0)

    # Between-chain variance
    B = n * np.var(chain_means, axis=0, ddof=1)

    # Within-chain variance
    W = np.mean(chain_vars, axis=0)

    # Estimate of marginal posterior variance
    var_hat = (1 - 1/n) * W + B / n

    R_hat = np.sqrt(var_hat / W)
    return R_hat
