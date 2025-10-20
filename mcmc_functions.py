import numpy as np
from scipy.special import wofz
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import convolve
import pickle

from essential_functions import read_parameter

#e = 4.8032e-10 # electron charge in stat-coulumb
e = 4.80320425 * 10**-10
#m_e = 9.10938e-28 # electron mass
m_e = 9.1094 *10**-28
c = 2.9979e10 # cm/s
c_As = 2.9979e18
c_kms = 2.9979e5
k = 1.38065e-16 # erg/K

def load_object(filename):
    with open(filename, 'rb') as inp:  # Open the file in binary read mode
        return pickle.load(inp)  # Return the unpickled object


class mcmc_line:
    def __init__(self,example_line,elements,saturation_direction=None):

        self.microLines={}

        #self.z=self.params[0]
        self.z=(example_line.peak-example_line.suspected_line)/example_line.suspected_line

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
        
        vels=[microline.mcmc_vel for microline in list(self.microLines.values())]

        self.vel_range=(list(self.microLines.values())[0].velocity[0],list(self.microLines.values())[0].velocity[-1])

        params = [np.mean(np.array(vels))]

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

                log_N=line_to_use.logN

                b=len(line_to_use.wavelength)#/np.sqrt(2)

            params.extend([log_N,b])

        return params
    

def read_atomic_mass(element):
    masses = {
        'H': 1.0079, 'He': 4.0026, 'C': 12.011, 'N': 14.007, 'O': 15.999,
        'Mg': 24.305, 'Si': 28.085, 'Fe': 55.845, 'Zn': 65.38
    }
    return masses.get(element, 28.0)
    
def velocity_to_redshift(velocity):
    # Converts velocity to redshift
    c = 299792.458  # Speed of light in km/s
    return (np.sqrt((velocity / c + 1)**2 / (1 - (velocity / c)**2)) - 1)

def redshift_to_velocity(redshift):
    # Converts redshift to velocity
    c = 299792.458  # Speed of light in km/s
    return c * ((1 + redshift)**2 - 1) / ((1 + redshift)**2 + 1)

def build_full_chain(
    sampler,
    free_indices,
    fixed_values,
    anchor_map,
    thermal_map,
    nonthermal_set,
    elements,
    shape
):
    raw_chain = sampler.get_chain()  # (n_walkers, n_steps, n_free)
    nwalkers, nsteps, nfree = raw_chain.shape
    total_params = shape[0] * shape[1]

    full_chain = np.zeros((nwalkers, nsteps, total_params))

    for w in range(nwalkers):
        full_chain[w] = rebuild_full_samples(
            flat_samples=raw_chain[w],  # shape (n_steps, n_free)
            free_indices=free_indices,
            fixed_values=fixed_values,
            anchor_map=anchor_map,
            thermal_map=thermal_map,
            nonthermal_set=nonthermal_set,
            elements=elements,
            shape=shape
        )

    return full_chain



def plot_trace_block(chain_swp, labels, save_path, threshold=1e-8):
    """
    Plot traces for a block of params.
    Expects chain_swp with shape (n_steps, n_walkers, n_params).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    ch = np.asarray(chain_swp)
    if ch.ndim != 3:
        raise ValueError(f"Expected 3-D chain (steps,walkers,params), got {ch.shape}")

    n_steps, n_walkers, n_params = ch.shape

    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2.5 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    for p in range(n_params):
        ax = axes[p]
        # ch[:, :, p] is (steps, walkers); plotting 2-D draws one line per column (i.e., per walker).
        ax.plot(ch[:, :, p], alpha=0.3, lw=0.6, color='black')

        std = np.std(ch[:, :, p])
        if std < threshold:
            mean_val = np.mean(ch[:, :, p])
            ax.axhline(mean_val, color='red', linestyle='--', alpha=0.7)

        ax.set_ylabel(labels[p])

    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)








def plot_full_trace_all_walkers(
    sampler,
    free_indices,
    fixed_values,
    anchor_map,
    thermal_map,
    nonthermal_set,
    elements,
    shape,
    labels,
    save_path,
    threshold=1e-8
):
    chain = sampler.get_chain()  # shape: (n_walkers, n_steps, n_free)
    nwalkers, nsteps, nfree = chain.shape

    full_chain = np.zeros((nwalkers, nsteps, shape[0] * shape[1]))

    # Rebuild full chain
    for w in range(nwalkers):
        full_chain[w] = rebuild_full_samples(
            chain[w], free_indices, fixed_values, anchor_map,
            thermal_map, nonthermal_set, elements, shape
        )

    nparams = full_chain.shape[2]

    fig, axes = plt.subplots(nparams, figsize=(10, 3 * nparams), sharex=True)
    if nparams == 1:
        axes = [axes]

    for i in range(nparams):
        ax = axes[i]
        for w in range(nwalkers):
            ax.plot(full_chain[w, :, i], alpha=0.3, lw=0.8,c='black')

        if np.std(full_chain[:, :, i]) < threshold:
            ax.axhline(np.mean(full_chain[:, :, i]), color='red', linestyle='--', alpha=0.6)

        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("Step number")
    plt.tight_layout(h_pad=0)
    plt.savefig(save_path)
    plt.clf()



def rebuild_full_samples(
    flat_samples,
    free_indices,
    fixed_values,
    anchor_map,
    thermal_map,
    nonthermal_set,
    elements,
    shape
):

    n_samples = flat_samples.shape[0]
    n_rows, n_cols = shape
    total_params = n_rows * n_cols
    full_samples = np.zeros((n_samples, total_params))

    # --- Step 1: Fill fixed values ---
    for (i, j), val in fixed_values.items():
        flat_index = i * n_cols + j
        full_samples[:, flat_index] = val

    # --- Step 2: Fill free values ---
    for k, (i, j) in enumerate(free_indices):
        flat_index = i * n_cols + j
        full_samples[:, flat_index] = flat_samples[:, k]

    # --- Step 3: Fill anchored parameters ---

    for (i, j), ref_element in anchor_map.items():
        target_element_index = (j - 1) // 2
        target_element = elements[target_element_index]

        ref_index = elements.index(ref_element)
        ref_b_col = 1 + 2 * ref_index + 1
        ref_flat_index = i * n_cols + ref_b_col
        target_flat_index = i * n_cols + j

        full_samples[:, target_flat_index] = full_samples[:, ref_flat_index]
    
    '''
    for (i, j), (ti, tj) in anchor_map.items():
        src_index = ti * n_cols + tj
        dest_index = i * n_cols + j
        full_samples[:, dest_index] = full_samples[:, src_index]'''

    # --- Step 4: Fill thermal parameters ---
    def read_atomic_mass(element):
        masses = {
            'H': 1.0079, 'He': 4.0026, 'C': 12.011, 'N': 14.007, 'O': 15.999,
            'Mg': 24.305, 'Si': 28.085, 'Fe': 55.845, 'Zn': 65.38
        }
        return masses.get(element, 28.0)

    for (i, j), ref_element in thermal_map.items():
        target_element_index = (j - 1) // 2
        target_element = elements[target_element_index]

        ref_index = elements.index(ref_element)
        ref_b_col = 1 + 2 * ref_index + 1
        ref_flat_index = i * n_cols + ref_b_col

        target_flat_index = i * n_cols + j

        ref_mass = read_atomic_mass(ref_element[:2])
        target_mass = read_atomic_mass(target_element[:2])
        scale_factor = 1 / np.sqrt(target_mass / ref_mass)

        full_samples[:, target_flat_index] = full_samples[:, ref_flat_index] * scale_factor

    # --- Step 5: Fill non-thermal parameters (shared b values) ---
    for i in range(n_rows):
        b_indices = [j for (ii, j) in nonthermal_set if ii == i]
        if not b_indices:
            continue
        try:
            mgii_index = elements.index("MgII")
            mgii_b_col = 1 + 2 * mgii_index + 1
        except ValueError:
            mgii_b_col = b_indices[0]  # fallback

        ref_flat_index = i * n_cols + mgii_b_col

        for j in b_indices:
            dest_flat_index = i * n_cols + j
            full_samples[:, dest_flat_index] = full_samples[:, ref_flat_index]

    return full_samples


def parse_statuses(statuses, initial_guesses):
    free_indices = []
    fixed_values = {}
    anchor_map = {}
    thermal_map = {}   # NEW
    nonthermal_set = set()  # NEW

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
                target = (status.split(':')[1])
                anchor_map[(i, j)] = target
            elif status.startswith('thermal:'):
                element = status.split(':')[1]
                thermal_map[(i, j)] = element
            elif status == 'non-thermal':
                nonthermal_set.add((i, j))
            else:
                raise ValueError(f"Unknown status {status} at ({i},{j})")

    # Exclude thermal/non-thermal from sampling
    free_indices = [idx for idx in free_indices if idx not in thermal_map and idx not in nonthermal_set]

    return free_indices, fixed_values, anchor_map, thermal_map, nonthermal_set

def rebuild_full_params(free_values, free_indices, fixed_values, anchor_map, shape,
                        thermal_map=None, nonthermal_set=None, elements=None):
    full_params = np.zeros(shape)

    for (i, j), val in fixed_values.items():
        full_params[i, j] = val

    for idx, (i, j) in enumerate(free_indices):
        full_params[i, j] = free_values[idx]
    '''
    for (i, j), (ti, tj) in anchor_map.items():
        full_params[i, j] = full_params[ti, tj]
    '''
    for (i, j), element in anchor_map.items():
            ref_index = elements.index(element)
            target_index = (j - 1) // 2
            ref_b_idx = 1 + ref_index * 2 + 1
            ref_b = full_params[i, ref_b_idx]
            full_params[i, j] = ref_b

    if thermal_map:
        for (i, j), element in thermal_map.items():
            ref_index = elements.index(element)
            target_index = (j - 1) // 2
            ref_mass = read_atomic_mass(element[0:2])
            target_mass = read_atomic_mass(elements[target_index][0:2])
            ref_b_idx = 1 + ref_index * 2 + 1
            ref_b = full_params[i, ref_b_idx]
            full_params[i, j] = ref_b / np.sqrt(target_mass / ref_mass)

    if nonthermal_set:
        for i in range(shape[0]):
            # Get all b-indices for this component
            b_indices = [j for (ii, j) in nonthermal_set if ii == i]
            if b_indices:
                try:
                    mgii_idx = elements.index("MgII")
                    mgii_b_col = 1 + 2 * mgii_idx + 1  # velocity + 2*logN/b per element + b offset
                    mgii_b_val = full_params[i, mgii_b_col]
                except ValueError:
                    print(f"[WARNING] MgII not found in elements list. Defaulting to first non-thermal b.")
                    mgii_b_val = full_params[i, b_indices[0]]  # fallback if MgII not present

                for j in b_indices:
                    full_params[i, j] = mgii_b_val


    return full_params.flatten()



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



def voigt(x, y):

    z=x+(1j*y)

    return wofz(z).real
    #return wofz(u + 1j * a).real

def calctau(velocity,ref_vel,logN, b, line):

    
    f=line.f
    gamma=line.gamma
    
    # Go from logN to N
    N = 10.0**logN #cm^-2

    lambda_array=line.suspected_line*(1+(velocity/c_kms)) #angstroms
    lambda_naut=line.suspected_line*(1+(ref_vel/c_kms))   #angstroms

    delta_lambda_d = (b/c_kms)*lambda_naut #angstroms

    x_lambda=(lambda_array-lambda_naut)/delta_lambda_d #dimensionless (angstrom/angstrom)
    y=(gamma*lambda_naut**2)/(4*np.pi*c_As*delta_lambda_d) #dimensionless

    H=voigt(x_lambda,y)

    numerator=N*np.pi*(e**2)*(lambda_naut**2)*f #I worked out the units and it ends up being cm/angstroms
    denomenator=m_e*(c_As**2)*np.sqrt(np.pi)*delta_lambda_d * 1e-8 #comes out to angstroms, convert to cm to have dimensionless units

    tau=(numerator/denomenator)*H # dimensionless

    #now calc flux from tau
    return np.exp(-tau)
    '''
    # calculate the rest-frame frequency in 1/s
    nu0 = c_As/line.suspected_line
    nu_ref = nu0 * (1 - ref_vel / c_kms)

    # Calculate nu in the rest-frame
    nu_rest = nu0 * (1 - velocity / c_kms)

    #this should be lambda naut
    delta_lambda_d = (b/c_kms)*(c_kms/nu_rest)#b*nu0/c_kms
    a = gamma/4.0/np.pi/delta_lambda_d
    u = (nu_rest - nu_ref)/delta_lambda_d

    H = voigt(a, u)

    # calculate tau
    numerator=
    tau = N*np.pi*e**2/m_e/c*f/np.sqrt(np.pi)/delta_lambda_d*H

    return np.exp(-tau)'''

def kernel_gaussian(wave, wave_mean, sigma):
   
   kernel = 1/np.sqrt(2*sigma**2*np.pi)*np.exp(-(wave - wave_mean)**2/(2*sigma**2))
   kernel = kernel/np.sum(kernel)
   
   return kernel

def convolve_flux(vel,flux,fwhm):
     
     dW_fwhm = fwhm
     dW_sigma = dW_fwhm/2.355
     
     #pixScale = wave[int(len(wave)/2)] - wave[int(len(wave)/2 - 1)]  
     pixScale = vel[1]-vel[0]
     dPix_sigma = dW_sigma/pixScale
     
     
     pix_kernel = np.concatenate((-1*np.arange(1, 10*dPix_sigma, 1), [0],
                                 np.arange(1, 10*dPix_sigma, 1)))
     pix_kernel.sort()
     
     pix_mean = 0.0
  
     kernel = kernel_gaussian(pix_kernel, pix_mean, dPix_sigma)

     # Continuum subtract and invert to prevent edge effects
     #flux = flux - 1
     #flux = flux*-1
     flux = convolve(flux, kernel, 'same')
     
     
     # Now undo continuum subtraction and inversion
     #flux = flux*-1
     #flux = 1 + flux

     return flux


import pandas as pd

def summarize_params(flat_samples, labels, elements, mcmc_lines, file_name):
    summary = []
    params_per_microline = (2 * len(elements)) + 1


    for i, mcmc_line_obj in enumerate(mcmc_lines):
        param_block = flat_samples[:, i * params_per_microline:(i + 1) * params_per_microline]

        # Velocity
        vel_samples = param_block[:, 0]
        vel_p16, vel_p50, vel_p84 = np.percentile(vel_samples, [16, 50, 84])
        rel_median = vel_p50
        rel_low = vel_p50 - vel_p16
        rel_high = vel_p84 - vel_p50

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

            summary.append({
                'Component': i + 1,
                'Species': f"{element}",
                f'dv_c (km/s)': f"{rel_median:.1f} (+{rel_high:.2f}/-{rel_low:.2f})",
                f'log N': f"{logN_p50:.2f} (+{logN_high:.2f}/-{logN_low:.2f}) ; ifsat ({np.percentile(logN_samples, 5):.2f}) ; ifnodetect ({np.percentile(logN_samples, 95):.2f})",
                f'b (km/s)': f"{b_p50:.1f} (+{b_high:.1f}/-{b_low:.1f}) ; ifsat ({np.percentile(b_samples, 95):.1f}) ; ifnodetect ({np.percentile(b_samples, 95):.1f})"
            })

    df = pd.DataFrame(summary)
    df.to_csv(f"static/Data/multi_mcmc/{file_name}.csv", index=False)
    return df
