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
from mcmc_functions import find_N, calctau, convolve_flux,mcmc_line, summarize_params,velocity_to_redshift,redshift_to_velocity,parse_statuses,rebuild_full_params,read_atomic_mass,rebuild_full_samples,plot_trace,build_full_chain
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

    mcmc_lines.append(mcmc_lines[line_number])

    return best_params,mcmc_lines


def plot_fits(params, line_dict, elements, mcmc_lines,file_name):

    import smplotlib

    c = 3e5
    vel_window=400

    num_params_per_line = 1 + 2 * len(elements)
    param_list_2d = np.array(params).reshape(-1, num_params_per_line)


    models=total_multi_model(params,line_dict,elements,mcmc_lines,high_resolution=True,extra=True)
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
    reference_z=load_object('static/Data/multi_mcmc/initial/ref_z.pkl')
    reference_velocity = redshift_to_velocity(reference_z)


    fig, axs = plt.subplots(len(line_dict.values()), 1, figsize=(10, 3 * len(line_dict.values())), squeeze=False,sharex=True,sharey=True)
    axs_flat = axs.ravel()

    fig.text(0.04, 0.5, 'Normalized Flux', va='center', rotation='vertical', fontsize=20)

    for i, name in enumerate(line_dict.keys()):

        line=line_dict.get(name)

        #line.store_model(np.linspace(line.MgII_wavelength[0],line.MgII_wavelength[-1],len(line.MgII_wavelength)*10), models.get(name))

        ax=axs_flat[i]

        #chi squared
        obs_flux = line.MgII_flux
        model_flux = standard_models[name]
        errors = np.sqrt(line.MgII_errors)
        #errors= line.MgII_errors
        #errors= np.sqrt(1/line.MgII_errors)

        # Calculate chi-squared and reduced chi-squared
        chi_squared = np.sum(((obs_flux - model_flux) / errors) ** 2)
        degrees_of_freedom = len(obs_flux) - (len(mcmc_lines)*3)
        reduced_chi_squared = chi_squared / degrees_of_freedom if degrees_of_freedom != 0 else 0

        ax.text(vel_window-120, 0.3, f"$\chi^2_{{red}}={reduced_chi_squared:.2f}$")

        #actual plot
        reference_microline=(reference_z+1)*line.suspected_line

        full_velocity = (line.extra_wavelength - reference_microline) / reference_microline * c 
        velocity =  (line.MgII_wavelength - reference_microline) / reference_microline * c 

        
        ax.step(full_velocity, line.extra_flux, where='mid', label=f"Flux", color="black")
        ax.step(full_velocity, line.extra_errors, where='mid', label="Error", color="cyan")

        ax.step(velocity,standard_models.get(name), where='mid', label=f"Model", color="purple")

        high_res_full_velocity=np.linspace(full_velocity[0],full_velocity[-1],len(full_velocity)*10)

        line.store_model(high_res_full_velocity, models.get(name),reduced_chi_squared)

        ax.step(high_res_full_velocity, models.get(name), where='mid', label=f"Model", color="red")


        for i,line_params in enumerate(param_list_2d):

            z=velocity_to_redshift(line_params[0])
            wavelength = line.suspected_line * (1+z)
            velocity =  (wavelength - reference_microline) / reference_microline * c

            ax.vlines(velocity, ymin=1.2,ymax=1.3,color='blue')
        
        ax.text(vel_window-120,0.1,f'{name.split(" ")[0]} {int(np.floor(float(name.split(" ")[1])))}')
        ax.set_xlim(-vel_window, vel_window)

    # Label axes and configure layout
    axs_flat[-1].set_xlabel('Relative Velocity (km/s)', fontsize=12)

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

def total_multi_model(params, line_dict, elements, mcmc_lines, convolve_data=True,high_resolution=False,chi2=False,extra=False):

    params_per_microline=(2*len(elements))+1
    param_list_2d = np.array(params).reshape(-1, params_per_microline)

    models={}

    if chi2:
        chi_value=0

    for key,line in line_dict.items():

        if high_resolution:
            if extra:
                wavelength=np.linspace(line.extra_wavelength[0],line.extra_wavelength[-1],len(line.extra_wavelength)*10)
            else:
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

def log_multi_likelihood(params, lines, elements, mcmc_lines, masked_regions=None):

    models = total_multi_model(params, lines, elements, mcmc_lines)

    chi2 = 0

    for key, line in lines.items():
        model = models.get(key)
        flux = line.MgII_flux
        error = line.MgII_errors
        velocity = (line.MgII_wavelength - (1 + line.z) * line.suspected_line) / ((1 + line.z) * line.suspected_line) * c_kms

        if masked_regions and key in masked_regions:
            mask = (velocity < masked_regions[key]["xmin"]) | (velocity > masked_regions[key]["xmax"])
        else:
            mask = np.ones_like(flux, dtype=bool)

        # Apply mask
        flux = flux[mask]
        error = error[mask]
        model = model[mask]

        chi2 += np.sum(np.log(1 / np.sqrt(2 * np.pi) / error) - (flux - model) ** 2 / (2 * error**2))

    return chi2


def log_multi_prior(params,elements,mcmc_lines):

    params_per_microline=(2*len(elements))+1

    for i,line in enumerate(mcmc_lines):

        line_params=params[i*params_per_microline:(i*params_per_microline)+params_per_microline]

        velocity=line_params[0]

        vel_range=(redshift_to_velocity(line.z_range[0]),redshift_to_velocity(line.z_range[1]))

        for j,e in enumerate(elements):

            logN=line_params[(j*2)+1]
            b=line_params[(j*2)+2]

            if not (vel_range[0] < velocity < vel_range[1] and 0 < b < 20 and 0 < logN < 20):
                return -np.inf
            
    return 0.0

def log_multi_probability(free_values, line_dict, elements, mcmc_lines,
                          free_indices, fixed_values, anchor_map, shape,
                          thermal_map=None, nonthermal_set=None):
    params = rebuild_full_params(free_values, free_indices, fixed_values, anchor_map,
                                 shape, thermal_map, nonthermal_set, elements)

    lp = log_multi_prior(params, elements, mcmc_lines)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_multi_likelihood(params, line_dict, elements, mcmc_lines)

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
    

    strong_line=line_dict.get('MgII 2796.355099')

    print('strong line')
    print(strong_line.wavelength)

    for microline in strong_line.mcmc_microlines:

        mcmc_lines.append(mcmc_line(microline,elements))

    print(mcmc_lines)
    print(len(mcmc_lines))


    for line in line_list:

        element=line.name.split(' ')[0]
        line.mcmc_microlines=[]

        for mcmc_line_obj in mcmc_lines:

            print(line.name)

            print(mcmc_line_obj.z_range[0],mcmc_line_obj.z_range[1])

            wav,flux,er=line.give_data(mcmc_line_obj.z_range[0],mcmc_line_obj.z_range[1])

            print(wav,flux,er)

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


    initial_guesses=[]

    for line in mcmc_lines:

        initial_guesses.extend(line.export_params())

    param_list = [line.export_params() for line in mcmc_lines]
    initial_guesses_2d=np.array(param_list)

    


    for i,microline in enumerate(strong_line.mcmc_microlines):

        if microline.is_saturated:
            if microline.saturation_direction == 'right':
                print('creeping right')
                initial_guesses,mcmc_lines=creep(initial_guesses,i,line_dict,elements,mcmc_lines,direction='right')

            elif microline.saturation_direction == 'left':
                print('creeping left')
                initial_guesses,mcmc_lines=creep(initial_guesses,i,line_dict,elements,mcmc_lines,direction='left')

    initial_guesses=optimize_params(initial_guesses,line_dict,elements,mcmc_lines)

    #_________________________________________________________________________________________  
    #plot all initial guesses

    plot_fits(initial_guesses,line_dict,elements,mcmc_lines,'initial/initial_guesses')


    #save all the objects
    save_object(line_dict,'static/Data/multi_mcmc/initial/initial_line_dict.pkl')
    save_object(mcmc_lines,'static/Data/multi_mcmc/initial/initial_mcmc_lines.pkl')
    save_object(elements,'static/Data/multi_mcmc/initial/initial_element_list.pkl')

    return initial_guesses,line_dict

def update_fit(parameters,elements):

    #load neccesary objects
    line_dict=load_object('static/Data/multi_mcmc/initial/initial_line_dict.pkl')
    mcmc_lines=load_object('static/Data/multi_mcmc/initial/initial_mcmc_lines.pkl')

    params=np.array(parameters).flatten().tolist()

    #create new plot
    plot_fits(params,line_dict,elements,mcmc_lines,'initial/initial_guesses')

    #save over old params with updated ones
    save_object(parameters,'static/Data/multi_mcmc/initial/initial_guesses.pkl')


def mcmc(initial_guesses,statuses, nsteps=1000,nwalkers=250):

    line_dict=load_object('static/Data/multi_mcmc/initial/initial_line_dict.pkl')
    elements=load_object('static/Data/multi_mcmc/initial/initial_element_list.pkl')
    mcmc_lines=load_object('static/Data/multi_mcmc/initial/initial_mcmc_lines.pkl')
    masked_regions = load_object("static/Data/multi_mcmc/initial/masked_regions.pkl")

    print('Arrays')
    print(initial_guesses)
    print(statuses)

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
                vel_range = (redshift_to_velocity(mcmc_lines[i].z_range[0]),
                            redshift_to_velocity(mcmc_lines[i].z_range[1]))
                sampled_val = np.random.uniform(*vel_range)
            else:  # logN or b
                std = percent_off * abs(base_val) if base_val != 0 else percent_off
                sampled_val = np.random.normal(base_val, std)

            walker_pos.append(sampled_val)

        pos.append(walker_pos)

    pos = np.array(pos)

    print('pos')
    print(len(pos[0]))
    print(pos)

    '''
    pos=[]
    for n in range(nwalkers):

        current_pos=[]

        for i,mcmc_line in enumerate(mcmc_lines):

            vel_range=(redshift_to_velocity(mcmc_line.z_range[0]),redshift_to_velocity(mcmc_line.z_range[1]))

            current_pos.append(np.random.uniform(vel_range[0],vel_range[1]))

            for j, e in enumerate(elements):
                logN_index = i * params_per_line + 1 + 2 * j
                b_index = i * params_per_line + 2 + 2 * j

                logN_val = initial_free_values[logN_index]
                b_val = initial_free_values[b_index]

                logN_sample = np.random.normal(logN_val, percent_off * abs(logN_val))
                b_sample = np.random.normal(b_val, percent_off * abs(b_val))

                current_pos.append(logN_sample)
                current_pos.append(b_sample)

        pos.append(current_pos)

    pos=np.array(pos)'''

    #12 max
    #5x faster!
    num_processes = os.cpu_count()
    with Pool(num_processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_multi_probability,
                                        args=(line_dict, elements, mcmc_lines,
                                            free_indices, fixed_values, anchor_map, shape,
                                            thermal_map, nonthermal_set),
                                            moves=emcee.moves.DEMove())

        sampler.run_mcmc(pos, nsteps, progress=True)

    # Analyzing the results
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    median_params = np.median(flat_samples,axis=0)

    log_probs = sampler.get_log_prob(discard=100, thin=15, flat=True)
    map_params = flat_samples[np.argmax(log_probs)]

    #rebuild params 
    map_params = flat_samples[np.argmax(log_probs)]

    map_params = rebuild_full_params(map_params, free_indices, fixed_values, anchor_map,
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

    print('full chain')
    print(full_chain)
    print(full_chain.shape)


    save_object(map_params,'static/Data/multi_mcmc/final/initial_guesses.pkl')

    #_________________________________________________________________________________________  
    #plot all fits

    plot_fits(map_params,line_dict,elements,mcmc_lines,'final/final_models')

    
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



    #_________________________________________________________________________________________
    #corner
    figure = corner.corner(
        full_samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        range=ranges
    )
    plt.savefig(f"static/Data/multi_mcmc/final/mcmc_corner.png")
    plt.clf()


    #_________________________________________________________________________________________
    #trace plot
    plot_trace(
        np.transpose(full_chain, (1, 0, 2)),
        labels=labels,
        save_path="static/Data/multi_mcmc/final/mcmc_trace.png"
    )


    #chat corner plot
    '''
    params_per_microline = 1 + 2 * len(elements)
    for i, mcmc_line_obj in enumerate(mcmc_lines):
        labels = []
        cols_to_plot = []
        transforms = []
        for j, base_label in enumerate(['velocity'] + [f'LogN_{e}' if k % 2 == 0 else f'b_{e}' for e in elements for k in range(2)]):
            status = statuses[i][j]
            label = base_label

            if status == 'fixed':
                label += " [fixed]"
            elif status.startswith('anchor_to'):
                label += " [anchored]"
            elif status.startswith('thermal:'):
                label += " [thermal]"
            elif status == 'non-thermal':
                label += " [non-thermal]"

            labels.append(label)
            cols_to_plot.append(j)

            # For thermal parameters, store scaling factor to apply later
            if status.startswith('thermal:'):
                ref_element = status.split(':')[1]
                ref_index = elements.index(ref_element)
                ref_b_idx = 1 + ref_index * 2 + 1
                target_index = (j - 1) // 2
                ref_mass = read_atomic_mass(ref_element)
                target_mass = read_atomic_mass(elements[target_index])
                scale_factor = 1 / np.sqrt(target_mass / ref_mass)
                transforms.append(("scale", j, ref_b_idx, scale_factor))
            elif status.startswith('anchor_to:'):
                anchor_i = int(status.split(':')[1])
                transforms.append(("copy", anchor_i, j))
            elif status == 'non-thermal':
                # Non-thermal b params will be matched to MgII
                mgii_idx = elements.index("MgII")
                mgii_b_col = 1 + 2 * mgii_idx + 1
                transforms.append(("copy", mgii_b_col, j))
            else:
                transforms.append(None)

        start = i * params_per_microline
        end = start + params_per_microline
        full_samples = np.array(full_samples)
        param_block = full_samples[:, start:end]

        # Apply transformations
        for t in transforms:
            if t is None:
                continue
            if t[0] == "copy":
                _, src, dest = t
                param_block[:, dest] = param_block[:, src]
            elif t[0] == "scale":
                _, dest, src, scale_factor = t
                param_block[:, dest] = param_block[:, src] * scale_factor


        # Noise for fixed lines
        for j in range(param_block.shape[1]):
            if np.std(param_block[:, j]) < 1e-10:
                param_block[:, j] += np.random.normal(0, 1e-6, size=param_block.shape[0])

        fig = corner.corner(
            param_block,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84], 
            show_titles=True, 
            title_kwargs={"fontsize": 12}
        )
        plt.savefig(f"static/Data/multi_mcmc/final/mcmc_corner_{i}.png")
        plt.clf()'''


    '''
    #_________________________________________________________________________________________  
    #corner plot
    params_per_microline = 1 + 2 * len(elements)

    full_samples = np.array(full_samples)

    

    for i, mcmc_line_obj in enumerate(mcmc_lines):

        labels = []
        for j, base_label in enumerate(['velocity'] + [f'LogN_{e}' if k % 2 == 0 else f'b_{e}' for e in elements for k in range(2)]):
            status = statuses[i][j]
            if status == 'fixed':
                labels.append(f'{base_label} [fixed]')
            elif status.startswith('anchor_to'):
                labels.append(f'{base_label} [anchored]')
            else:
                labels.append(base_label)


        # Extract the block of samples for this component
        start = i * params_per_microline
        end = start + params_per_microline
        param_block = full_samples[:, start:end]  # shape: (n_samples, params_per_microline)

        # Add tiny noise to constant columns
        for col in range(param_block.shape[1]):
            std = np.std(param_block[:, col])
            if std < 1e-10:
                param_block[:, col] += np.random.normal(0, 1e-6, size=param_block.shape[0])

        # Create the corner plot for just this component
        figure = corner.corner(
            param_block,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84], 
            show_titles=True, 
            title_kwargs={"fontsize": 12}
        )

        plt.savefig(f"static/Data/multi_mcmc/final/mcmc_corner_{i}.png")
        plt.clf()'''
    
    #chat trace
    '''
    def plot_trace(sampler, param_names, free_indices, statuses, threshold=1e-4):
        chain = sampler.get_chain()
        nwalkers, nsteps, ndim = chain.shape

        # Reconstruct labels for only free parameters
        varying_indices = []
        filtered_names = []

        for k in range(ndim):
            std = np.std(chain[:, :, k])
            i, j = free_indices[k]
            label = param_names[i * len(statuses[0]) + j]  # flatten 2D label list
            if std > threshold:
                filtered_names.append(label)
            else:
                filtered_names.append(label + " [flat]")
            varying_indices.append(k)

        fig, axes = plt.subplots(len(varying_indices), figsize=(10, 3 * len(varying_indices)), sharex=True)

        if len(varying_indices) == 1:
            axes = [axes]

        for ax, k in zip(axes, varying_indices):
            for j in range(nwalkers):
                ax.plot(chain[j, :, k], "k", alpha=0.3)
            std = np.std(chain[:, :, k])
            if std < threshold:
                mean_val = np.mean(chain[:, :, k])
                ax.axhline(mean_val, color='red', linestyle='--', alpha=0.6)
            ax.set_xlim(0, nsteps)
            ax.set_ylabel(filtered_names[varying_indices.index(k)])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("Step number")
        plt.tight_layout(h_pad=0)
        plt.savefig("static/Data/multi_mcmc/final/mcmc_trace.png")
        plt.clf()


    plot_trace(sampler, labels, free_indices, statuses)'''




    '''
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
        plt.clf()'''

    #plot_trace(sampler, labels)

    #__________________________________________________________________________________________
    #make table like his plot

    df = summarize_params(full_samples, labels, elements, mcmc_lines,'final/mcmc_results')
    #df.to_csv('multi_mcmc_results.csv')
    #df.to_latex(index=False, column_format='cccccc')

    #__________________________________________________________________________________________
    #save the chain
    chain=sampler.get_chain()
    np.save('static/Data/multi_mcmc/chain.npy', chain)  # Saves the chain

    save_object(line_dict,'static/Data/multi_mcmc/final/line_dict.pkl')
    save_object(mcmc_lines,'static/Data/multi_mcmc/final/mcmc_lines.pkl')

    print("Acceptance fraction:", np.mean(sampler.acceptance_fraction))

    try:
        tau = sampler.get_autocorr_time()
        print("Autocorrelation time:", tau)
    except emcee.autocorr.AutocorrError:
        print("Warning: Chain too short to estimate autocorrelation time")

    
    #for i,object in enumerate(mcmc_lines):
    #    save_object(i,f'static/Data/multi_mcmc/final/mcmc_lines{i}.pkl')


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