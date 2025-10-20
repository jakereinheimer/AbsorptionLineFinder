import pickle
import numpy as np
import matplotlib.pyplot as plt


from mcmc import pre_mcmc,update_fit,mcmc, plot_fits

#helper functions
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Open the file in binary write mode
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)  # Pickle the object and write to file

def load_object(filename):
    with open(filename, 'rb') as inp:  # Open the file in binary read mode
        return pickle.load(inp)  # Return the unpickled object


def chain_upload_(save_dir,class_flags=None):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import corner

    def gelman_rubin(chains):  # chains: shape (n_steps, n_walkers)
        """
        Computes the Gelman-Rubin R̂ statistic for a single parameter across walkers.
        """
        m = chains.shape[1]  # number of chains (walkers)
        n = chains.shape[0]  # samples per chain

        chain_means = np.mean(chains, axis=0)
        chain_vars = np.var(chains, axis=0, ddof=1)
        grand_mean = np.mean(chain_means)

        # Between-chain variance
        B = n * np.var(chain_means, ddof=1)

        # Within-chain variance
        W = np.mean(chain_vars)

        # Estimated variance of the target distribution
        var_hat = (1 - 1/n) * W + (1/n) * B

        R_hat = np.sqrt(var_hat / W)
        return R_hat
    


    chain=np.load(os.path.join(save_dir,"chain.npy"))

    burn = int(chain.shape[0] * 0.5)
    chain = chain[burn:, :, :]

    mcmc_lines=load_object(os.path.join(save_dir,'final/mcmc_lines.pkl'))
    line_dict=load_object(os.path.join(save_dir,'final/line_dict.pkl'))
    params = load_object(os.path.join(save_dir,'final/initial_guesses.pkl'))
    map_params = load_object(os.path.join(save_dir,'final/initial_guesses.pkl'))
    elements = load_object(os.path.join(save_dir,'initial/initial_element_list.pkl'))
    num_params_per_line = 1 + 2 * len(elements)
    params = np.array(params).reshape(-1, num_params_per_line)
    statuses = np.array(load_object(os.path.join(save_dir,'initial/initial_statuses.pkl')))
    column_names=load_object(os.path.join(save_dir,'initial/column_names.pkl'))
    num_params_per_line = len(column_names)

    map_params=map_params.reshape(-1, num_params_per_line)

    # chain: shape (steps, walkers, parameters)
    flattened = chain.reshape(-1, chain.shape[-1])  # shape: (10100*250, 10)
    median_params = np.median(flattened, axis=0)
    #plot_fits(median_params,line_dict,elements,mcmc_lines,'initial_fit_plot',chain_review=True)

    #plot_fits(map_params.flatten(),line_dict,elements,mcmc_lines,'initial_fit_plot',chain_review=True)
    plot_fits(map_params.flatten(),line_dict,elements,mcmc_lines,'initial_fit_plot',chain_review=True,show_components=True)

    # --- End Map Values ___

    all_summary_tables=[]
    n_components = chain.shape[-1] // num_params_per_line

    detection_flags={}

    for comp_idx in range(n_components):
        start = comp_idx * num_params_per_line
        end = (comp_idx + 1) * num_params_per_line
        comp_chain = chain[:, :, start:end]

        comp_chain_flat = comp_chain.reshape(-1, comp_chain.shape[-1])

        
        '''# --- TRACE PLOTS ---
        if map_params[comp_idx][1]>14:
            fig, axs = plt.subplots(num_params_per_line, figsize=(10, 2 * num_params_per_line), sharex=True)
            for j in range(num_params_per_line):
                
                axs[j].plot(comp_chain_flat[:, j], alpha=1, lw=0.01)

                param_chain = comp_chain[:, :, j]
                r_hat = gelman_rubin(param_chain)
                axs[j].text(0.02,0.02,f"R = {r_hat:.3f}", transform=axs[j].transAxes)

                axs[j].set_ylabel(f'{column_names[j]} ({statuses[comp_idx,j]})')

            axs[-1].set_xlabel("Step")
            fig.suptitle(f"Component {comp_idx + 1} Trace")
            trace_path = os.path.join('static/chain_review/trace', f"trace_component_{comp_idx}.png")
            fig.tight_layout()
            fig.savefig(trace_path)
            plt.close(fig)'''
        '''
        if map_params[comp_idx][1]>14:
            # --- CORNER PLOT ---
            comp_chain_flat = comp_chain.reshape(-1, comp_chain.shape[-1])
            fig = corner.corner(
                comp_chain_flat,
                labels=[f'{column_names[j]} ({statuses[comp_idx,j]})' for j in range(num_params_per_line)],
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 12}
            )

            corner_path = os.path.join('static/chain_review/triangle', f"triangle_component_{comp_idx}.png")
            fig.savefig(corner_path)
            plt.close(fig)'''

        # --- Summary table ---
        summary_table = []

        detection_flags[comp_idx]={}

        for j in range(num_params_per_line):
            samples = comp_chain_flat[:, j]

            #calc ew for relevant line
            parameter=column_names[j]
            if parameter!='Velocity':
                
                if "MgII" in parameter:
                    line=line_dict.get('MgII 2796.355099')
                    element="MgII"
                    MgII_b=map_params[comp_idx][2]
                    if MgII_b>10:
                        MgII_b=10
                elif "FeII" in parameter:
                    element="FeII"
                    line=line_dict.get('FeII 2600.1720322')
                    if line==None:
                        line=line_dict.get('FeII 2586.6492304')
                        if line == None:
                            line=line_dict.get('FeII 2382.7639122')
                elif "CaII" in parameter:
                    element="CaII"
                    line=line_dict.get('CaII 3934.774716')
                    if len(line.wavelength)<2:
                        line=line_dict.get('CaII 3969.5897875')
                elif "MgI" in parameter:
                    element="MgI"
                    line=line_dict.get('MgI 2852.96342')

                vmin,vmax = mcmc_lines[comp_idx].vel_range

                mask = (line.velocity >= vmin) & (line.velocity <= vmax)
                if not np.any(mask):
                    return None  # no valid region

                lam = line.wavelength[mask]
                f = line.flux[mask]
                f_err = line.errors[mask]  # Make sure this exists

                # Compute Δλ for each bin
                try:
                    dlam = np.gradient(lam)

                    # Compute equivalent width
                    ew = np.sum((1 - f) * dlam)

                    # Compute error on EW
                    ew_err = np.sqrt(np.sum((dlam * f_err) ** 2))
                except:
                    print(element)
                    print(parameter)
                    print(lam)

                    ew=0
                    ew_err=0

                detection_flag=detection_flags.get(comp_idx).get(element)
                if detection_flag is None:

                    if ew<2*ew_err:
                        detection_flags.get(comp_idx)[element]="not_detected"

                    #elif ew>.8:
                    #    detection_flags.get(comp_idx)[element]="saturated"

                    else:
                        detection_flags.get(comp_idx)[element]="detected"

                #calc N from ew
                wave_cm = line.suspected_line*1e-8
                f=line.f
                m_e = 9.109 * 10**(-28)  # electron mass in grams
                c = 2.998 * 10**10       # speed of light in cm/s
                e = 4.8 * 10**(-10)

                # Conversion constant
                K = (wave_cm**2 / c**2) * (np.pi * e**2 / m_e) * f * 1e8

                # EW and its uncertainty
                # Assume you already have ew and ew_err from previous step
                N = ew / K
                N_err = ew_err / K

                # Logarithmic column density and uncertainty
                logN = np.log10(N)
                logN_err = N_err / (N * np.log(10))

                eq_width = f"{int(ew*1000)} +/- {int(1000*ew_err)}"
                logN_from_EW = f"{logN:.2f} +/- {logN_err:.2f}"

            else:
                eq_width = None
                logN_from_EW = None

            '''
            if 'b' in column_names[j]:
                # Use the marginalized chain for UL95 calc
                b_samples   = comp_chain_flat[:, j]
                logn_samples = comp_chain_flat[:, j-1]  # assumes LogN immediately precedes b

                # Sweep tolerance wide -> narrow
                tols, ul95s, counts = sweep_ul95_vs_tol(
                    b_samples, logn_samples,
                    target_b=MgII_b,
                    tol_max=6.0, tol_min=0.1, step=0.1,
                    percentile=95, min_n=100  # tune min_n to your chain size
                )

                # Pick smallest stable tol (plateau within eps dex over a small window)
                chosen_tol, ul95_chosen = pick_stable_tol(
                    tols, ul95s,
                    eps=0.02,       # stability threshold in dex
                    window_size=3   # require a short run of stability
                )

                # Save plot
                plot_path = plot_ul95_vs_tol(
                    tols, ul95s, chosen_tol,
                    comp_idx=comp_idx, element=element,
                    out_dir=f'sanity/ul95_tol_comp{comp_idx}'
                )

                # Use this as your non-detection UL95
                logn_nondetection = ul95_chosen

            else:
                logn_nondetection = np.nan
                chosen_tol = np.nan'''
            
            if 'b' in column_names[j]:

                logn_nondetection=np.log10((2*ew_err)/K)
                chosen_tol=0
            
            else:

                logn_nondetection=0
                chosen_tol=0
                
            

            median = np.percentile(samples, 50)
            low = np.percentile(samples, 16)
            high = np.percentile(samples, 84)
            map_val = map_params[comp_idx,j]
            p95 = np.percentile(samples, 95)
            p5 = np.percentile(samples, 5)

            def build_summary_table(summary_table,samples,comp_idx,j,percision):

                median = np.percentile(samples, 50)
                low = np.percentile(samples, 16)
                high = np.percentile(samples, 84)
                map_val = map_params[comp_idx,j]
                p95 = np.percentile(samples, 95)
                p5 = np.percentile(samples, 5)

                summary_table.append({
                    "param": column_names[j],
                    "median": f"{median:.{percision}f} ± ({high - median:.{percision}f},{median - low:.{percision}f})",
                    "median value":np.round(median,percision),
                    "map": f"{map_val:.{percision}f}" + f" ± ({high:.{percision}f},{low:.{percision}f})",
                    "map useful":f"{map_val:.{percision}f} ± ({high - median:.{percision}f},{median - low:.{percision}f})",
                    "map only":f"{map_val:.{percision}f}",
                    "map value":np.round(map_val,percision),
                    "upper sigma":f"{high:.{percision}f}",
                    "upper sigma value":np.round(high-median,percision),
                    "lower sigma":f"{low:.{percision}f}",
                    "lower sigma value":np.round(median-low,percision),
                    "p95": f"{p95:.{percision}f}",
                    "p95 value":np.round(p95,percision),
                    "p5": f"{p5:.{percision}f}",
                    "p5 value":np.round(p5,percision),
                    "ew": eq_width,
                    "logN": logN_from_EW,
                    "logn_nondetection":np.round(logn_nondetection,percision)
                })

                return summary_table

            #corner plot
            if 'logn mgii' in column_names[j].lower():

                if abs(high-median)>.3 or abs(median-low)>.3:
                    # --- CORNER PLOT ---
                    comp_chain_flat = comp_chain.reshape(-1, comp_chain.shape[-1])
                    fig = corner.corner(
                        comp_chain_flat,
                        labels=[f'{column_names[j]} ({statuses[comp_idx,j]})' for j in range(num_params_per_line)],
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 12}
                    )

                    corner_path = os.path.join('static/chain_review/triangle', f"triangle_component_{comp_idx}.png")
                    fig.savefig(corner_path)
                    plt.close(fig)

            if 'logn' in column_names[j].lower():

                if high > median+.3:
                    if median>11:
                        detection_flags.get(comp_idx)[element]="saturated"

                if low < median-.3:
                    if median>11:
                        detection_flags.get(comp_idx)[element]="saturated"
                

                if map_val<11:
                    detection_flags.get(comp_idx)[element]="not_detected"

                if map_val>15:
                    detection_flags.get(comp_idx)[element]="saturated"

                summary_table=build_summary_table(summary_table,samples,comp_idx,j,percision=2)
            
            else:

                summary_table=build_summary_table(summary_table,samples,comp_idx,j,percision=1)

        all_summary_tables.append(summary_table)

    save_object(detection_flags,'static/chain_review/detection_flags.pkl')

    save_object(all_summary_tables,'static/chain_review/summary_tables.pkl')

    if class_flags:
        class_flags_df=pd.read_csv(class_flags)
        detection_flags=class_flags_df.to_dict(orient="index")

    return n_components,all_summary_tables,detection_flags,elements


def sweep_ul95_vs_tol(b_samples, logn_samples, target_b=10.0,
                      tol_max=6.0, tol_min=0.1, step=0.1,
                      percentile=95, min_n=100):
    """
    Returns tols (descending), ul95s, counts.
    Uses a hard window: |b - target_b| <= tol.
    Only computes percentile if >= min_n samples at that tol (else NaN).
    """
    m = np.isfinite(b_samples) & np.isfinite(logn_samples)
    b = b_samples[m]; ln = logn_samples[m]

    tols = np.arange(tol_max, tol_min - 1e-9, -step)  # wide -> narrow
    ul95s, counts = [], []
    for tol in tols:
        mask = np.abs(b - target_b) <= tol
        n = int(mask.sum())
        counts.append(n)
        if n >= min_n:
            ul95s.append(np.percentile(ln[mask], percentile))
        else:
            ul95s.append(np.nan)
    return tols, np.array(ul95s), np.array(counts)

def pick_stable_tol(tols, ul95s, eps=0.02, window_size=3):
    """
    Pick the SMALLEST tol (i.e., most narrow) where UL95 is stable:
    max-min over the last `window_size` values <= eps.
    `tols` is descending; we scan from wide->narrow and return the last idx in the
    first stable window we encounter when moving toward smaller tolerances.
    """
    n = len(tols)
    chosen_idx = None
    # slide a window and record the last index (smallest tol) in that stable window
    for i in range(n - window_size + 1):
        win = ul95s[i:i+window_size]
        if np.all(np.isfinite(win)):
            if (np.max(win) - np.min(win)) <= eps:
                chosen_idx = i + window_size - 1  # smallest tol in this stable window
                # keep going to allow even smaller tol windows later
    if chosen_idx is None:
        # fallback: smallest tol with a finite UL95
        finite = np.where(np.isfinite(ul95s))[0]
        if finite.size == 0:
            return np.nan, np.nan
        chosen_idx = finite[-1]
    return float(tols[chosen_idx]), float(ul95s[chosen_idx])

def plot_ul95_vs_tol(tols, ul95s, chosen_tol, comp_idx, element, out_dir='static/ul95_tol'):
    """
    Save UL95(logN) vs tol plot and return Flask path.
    """
    os.makedirs(out_dir, exist_ok=True)
    safe_element = element.replace(' ', '').replace('/', '')
    outpath = os.path.join(out_dir, f'comp{comp_idx+1}_{safe_element}.png')

    plt.figure()
    plt.plot(tols, ul95s, marker='o')
    if np.isfinite(chosen_tol):
        plt.axvline(chosen_tol, linestyle='--')
    plt.title(f'UL95(logN) vs tolerance — comp {comp_idx+1}, {element}')
    plt.xlabel('tolerance (km/s) for |b-10| ≤ tol')
    plt.ylabel('UL95(logN) [dex]')
    plt.grid(True)
    plt.gca().invert_xaxis()  # optional: show wide -> narrow left->right
    plt.savefig(outpath, dpi=160, bbox_inches='tight')
    plt.close()
    return '/' + outpath

#gaussian start


def weighted_percentile(data, weights, q):
    """
    Compute the weighted qth percentile of data (0 <= q <= 100).
    """
    data, weights = map(np.array, (data, weights))
    sorter = np.argsort(data)
    data, weights = data[sorter], weights[sorter]

    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.interp(q/100, cdf, data)

def kernel_logn_ul95(b_samples, logn_samples, target_b=10.0, sigma_b=1.0):
    """
    Gaussian-kernel weighted 95th percentile of LogN around target_b in b.
    """
    mask = np.isfinite(b_samples) & np.isfinite(logn_samples)
    b, ln = b_samples[mask], logn_samples[mask]
    if b.size == 0:
        return np.nan

    # Gaussian weights centered at target_b
    weights = np.exp(-0.5 * ((b - target_b) / sigma_b)**2)
    if weights.sum() == 0:
        return np.nan

    return weighted_percentile(ln, weights, 95)

def adaptive_kernel_logn_ul95(b_samples, logn_samples,
                              target_b=10.0,
                              sigma_min=0.1, sigma_max=6.0,
                              step=0.1,
                              eps=0.02):
    """
    Shrinks sigma_b until UL95 stabilizes (variation < eps dex).
    """
    sigmas = np.arange(sigma_min, sigma_max+1e-9, step)
    ul95s = []
    for s in sigmas:
        ul95s.append(kernel_logn_ul95(b_samples, logn_samples,
                                      target_b=target_b, sigma_b=s))
    ul95s = np.array(ul95s)

    chosen_idx = None
    for i in range(1, len(sigmas)):
        if np.isfinite(ul95s[i]) and np.isfinite(ul95s[i-1]):
            if abs(ul95s[i] - ul95s[i-1]) <= eps:
                chosen_idx = i
                break
    if chosen_idx is None:
        return ul95s[-1], sigmas[-1], sigmas, ul95s
    else:
        return ul95s[chosen_idx], sigmas[chosen_idx], sigmas, ul95s
    
def plot_ul95_vs_sigma(sigmas, ul95s, chosen_sigma, comp_idx, element, out_dir='static/ul95_sigma'):
    os.makedirs(out_dir, exist_ok=True)
    safe_element = element.replace(' ', '').replace('/', '')
    outpath = os.path.join(out_dir, f'comp{comp_idx+1}_{safe_element}.png')

    plt.figure()
    plt.plot(sigmas, ul95s, marker='o')
    if np.isfinite(chosen_sigma):
        plt.axvline(chosen_sigma, linestyle='--')
    plt.title(f'UL95(logN) vs σ_b — comp {comp_idx+1}, {element}')
    plt.xlabel('σ_b (km/s)')
    plt.ylabel('UL95(logN) [dex]')
    plt.grid(True)
    plt.savefig(outpath, dpi=160, bbox_inches='tight')
    plt.close()
    return '/' + outpath  # flask-static URL



def marginalize_component_(component_index,reference_element,target_element,percentile):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import corner

    summary_path = 'static/chain_review/summary_tables.pkl'
    chain_path   = 'chain_upload/user_upload/chain.npy'
    column_path  = 'chain_upload/user_upload/initial/column_names.pkl'
    status_path  = 'chain_upload/user_upload/initial/initial_statuses.pkl'
    element_path = 'chain_upload/user_upload/initial/initial_element_list.pkl'

    # --- Load persisted state ---
    chain          = np.load(chain_path)                      # shape: (n_steps, n_walkers, n_params_total)
    column_names   = load_object(column_path)                 # names for ONE component in order
    statuses       = load_object(status_path)                 # shape: (n_components, params_per_comp)
    elements       = load_object(element_path)                # e.g., ["MgII", "FeII", ...]
    summary_tables = load_object(summary_path)                # list of per-component summaries

    n_components = len(summary_tables)
    n_steps, n_walkers, n_params_total = chain.shape

    # Infer params per component and slice out the selected component
    num_params_per_line = n_params_total // n_components
    start = component_index * num_params_per_line
    end   = (component_index + 1) * num_params_per_line

    # Component view
    comp_chain = chain[:, :, start:end]                               # (n_steps, n_walkers, num_params_per_line)
    comp_chain_flat = comp_chain.reshape(-1, num_params_per_line)      # (n_steps*n_walkers, num_params_per_line)

    # === Marginalize by FeII thermal floor (mask samples) ===
    element_masses = {
        'HI': 1.0079, 'MgII': 24.305, 'FeII': 55.845, 'MgI': 24.305,
        'CIV': 12.011, 'SiII': 28.0855, 'OVI': 15.999, 'CaII': 40.078
    }

    # Map each element to its "b" index within this component
    element_b_indices = {
        el: j
        for el in elements
        for j, name in enumerate(column_names)
        if name.startswith("b ") and el in name
    }

    mask_flat = np.ones(comp_chain_flat.shape[0], dtype=bool)
    if reference_element in element_b_indices and target_element in element_b_indices:
        ref_idx = element_b_indices[reference_element]
        tgt_idx = element_b_indices[target_element]

        # percentile of reference element's b
        b_ref_vals = comp_chain_flat[:, ref_idx]
        b_ref_min = np.nanpercentile(b_ref_vals, percentile)
        b_ref_max = np.nanpercentile(b_ref_vals,100-percentile)

        m_ratio = np.sqrt(
            element_masses.get(reference_element, 1.0) /
            element_masses.get(target_element, 1.0)
        )

        floor = b_ref_min / m_ratio
        ceiling = b_ref_max / m_ratio

        print(f"b_min={floor}")
        print(f'b_max={ceiling}')

        # keep only samples where target b meets the floor (or is non-finite)
        tgt_vals = comp_chain_flat[:, tgt_idx]
        mask_flat &= (~np.isfinite(tgt_vals)) | ((tgt_vals >= floor) & (tgt_vals <= ceiling))


    # ---- Apply mask to the COMPONENT ONLY and write back into `chain` ----
    # "Cut" = set removed samples to NaN (preserve array shape; downstream stats can use nan-aware functions)
    comp_chain_flat_masked = comp_chain_flat.copy()
    comp_chain_flat_masked[~mask_flat, :] = np.nan
    comp_chain_masked = comp_chain_flat_masked.reshape(n_steps, n_walkers, num_params_per_line)

    # Overwrite the component slice in the original chain
    chain[:, :, start:end] = comp_chain_masked

    # Persist updated chain
    np.save(chain_path, chain)

    # --- Rebuild plots for this component using the masked data ---
    # TRACE (plot each walker for each param; ignore NaNs naturally in plotting)
    trace_dir = 'static/chain_review/trace'
    os.makedirs(trace_dir, exist_ok=True)
    fig, axs = plt.subplots(num_params_per_line, figsize=(10, 2 * num_params_per_line), sharex=True)
    if num_params_per_line == 1:
        axs = [axs]
    for j in range(num_params_per_line):
        # plot each walker
        for w in range(n_walkers):
            axs[j].plot(comp_chain_masked[:, w, j], alpha=0.3, lw=0.5)
        axs[j].set_ylabel(f'{column_names[j]} ({statuses[component_index][j]})')
    axs[-1].set_xlabel("Step")
    fig.suptitle(f"Component {component_index + 1} Trace (post-marginalization)")
    trace_path = os.path.join(trace_dir, f"trace_component_{component_index}.png")
    fig.tight_layout()
    fig.savefig(trace_path, dpi=160)
    plt.close(fig)

    # CORNER — flatten across steps & walkers, drop NaNs per parameter
    corner_dir = 'static/chain_review/triangle'
    os.makedirs(corner_dir, exist_ok=True)
    flat_for_corner = comp_chain_masked.reshape(-1, num_params_per_line)
    # remove rows that are all-NaN so corner doesn't choke
    ok_rows = ~np.all(~np.isfinite(flat_for_corner), axis=1)
    flat_for_corner = flat_for_corner[ok_rows]
    if flat_for_corner.size > 0:
        fig = corner.corner(flat_for_corner, labels=column_names, show_titles=True, title_fmt=".3f")
        corner_path = os.path.join(corner_dir, f"triangle_component_{component_index}.png")
        fig.savefig(corner_path, dpi=160, bbox_inches='tight')
        plt.close(fig)
    else:
        corner_path = None  # no valid samples remain

    # --- Recompute summary for this component (nan-aware) ---
    summary = []
    # Use flattened, masked samples (across steps & walkers)
    for j in range(num_params_per_line):
        samples = flat_for_corner[:, j] if flat_for_corner.size > 0 else np.array([np.nan])

        # nan-aware percentiles (if all NaN -> result NaN)
        median = np.nanpercentile(samples, 50) if np.isfinite(samples).any() else np.nan
        low     = np.nanpercentile(samples, 16) if np.isfinite(samples).any() else np.nan
        high     = np.nanpercentile(samples, 84) if np.isfinite(samples).any() else np.nan
        p95    = np.nanpercentile(samples, 95) if np.isfinite(samples).any() else np.nan
        p5     = np.nanpercentile(samples, 5)  if np.isfinite(samples).any() else np.nan

        name = column_names[j]
        existing = summary_tables[component_index][j] if (
            component_index < len(summary_tables) and
            j < len(summary_tables[component_index])
        ) else {}

        map_val=existing.get("map only")
        eq_width=existing.get("ew")
        logN_from_EW=existing.get("logN")
        logn_nondetection=existing.get("logn_nondetection")
        chosen_tol=existing.get("non_detection_tolerance")

        def build_summary_table(summary_table,samples,comp_idx,j,percision):

                median = np.percentile(samples, 50)
                low = np.percentile(samples, 16)
                high = np.percentile(samples, 84)
                map_val = median#map_params[comp_idx,j]
                p95 = np.percentile(samples, 95)
                p5 = np.percentile(samples, 5)

                summary_table.append({
                    "param": column_names[j],
                    "median": f"{median:.{percision}f} ± ({high - median:.{percision}f},{median - low:.{percision}f})",
                    "median value":np.round(median,percision),
                    "map": f"{map_val:.{percision}f}" + f" ± ({high:.{percision}f},{low:.{percision}f})",
                    "map useful":f"{map_val:.{percision}f} ± ({high - median:.{percision}f},{median - low:.{percision}f})",
                    "map only":f"{map_val:.{percision}f}",
                    "map value":np.round(map_val,percision),
                    "upper sigma":f"{high:.{percision}f}",
                    "upper sigma value":np.round(high-median,percision),
                    "lower sigma":f"{low:.{percision}f}",
                    "lower sigma value":np.round(median-low,percision),
                    "p95": f"{p95:.{percision}f}",
                    "p95 value":np.round(p95,percision),
                    "p5": f"{p5:.{percision}f}",
                    "p5 value":np.round(p5,percision),
                    "ew": eq_width,
                    "logN": logN_from_EW,
                })

                return summary_table


        if 'logn' in column_names[j].lower():

            summary=build_summary_table(summary,samples,component_index,j,2)
        
        else:
            summary=build_summary_table(summary,samples,component_index,j,1)

    summary_tables[component_index] = summary
    save_object(summary_tables, summary_path)

    detection_flags = load_object('static/chain_review/detection_flags.pkl')

    return n_components,summary_tables,detection_flags,elements


def generate_csv_(flags):

    import os
    import pandas as pd
    import numpy as np
    import re

    summary_tables = load_object('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/chain_review/summary_tables.pkl')
    save_dir = os.path.join("chain_upload", "user_upload")
    elements = load_object(os.path.join(save_dir, 'initial/initial_element_list.pkl'))
    line_dict = load_object(os.path.join(save_dir,'final/line_dict.pkl'))
    n_components = len(summary_tables)

    our_2796=line_dict.get('MgII 2796.355099')
    ref_z=load_object(os.path.join(save_dir,'initial/ref_z.pkl'))

    vmin = our_2796.velocity[0]
    vmax = our_2796.velocity[-1]


    # Create a list of rows, starting with metadata
    rows = [{
        "Object Name" : str(load_object('object_name.pkl')),
        "Galaxy Z": ref_z,
        "Integration Window": (int(vmin), int(vmax)),
    }]

    lines_to_find={
            'MgII 2803.5322972':2796.354,
            'MgII 2796.355099':2803.531,
            'FeII 2600.1720322':2600.1720322,
            'FeII 2586.6492304':2586.6492304,
            'FeII 2382.7639122':2382.7639122,
            'FeII 2374.4599813':2374.4599813,
            'FeII 2344.2126822':2344.2126822,
            'MgI 2852.96342': 2852.96342,
            'CaII 3934.774716':3934.774716,
            'CaII 3969.5897875':3969.5897875
        }
    
    full_line_dict=load_object(os.path.join(save_dir,'initial/full_line_dict.pkl'))
    
    for key,value in lines_to_find.items():

        line=line_dict.get(key,None)

        if line is None:
            #unused line
            try:
                line=full_line_dict.get(key)

                ew,ew_error=line.actual_ew_func()

                wave_cm = line.suspected_line*1e-8
                f=line.f
                m_e = 9.109 * 10**(-28)  # electron mass in grams
                c = 2.998 * 10**10       # speed of light in cm/s
                e = 4.8 * 10**(-10)

                # Conversion constant
                K = (wave_cm**2 / c**2) * (np.pi * e**2 / m_e) * f * 1e8

                # EW and its uncertainty
                # Assume you already have ew and ew_err from previous step
                N = (2*ew_error) / K

                # Logarithmic column density and uncertainty
                logN = np.log10(N)

                rows[0][f"{key.split(' ')[0]} {int(np.floor(float(key.split(' ')[1].strip())))} EW"]=f"<{int(2*1000*ew_error)}"
                rows[0][f"{key.split(' ')[0]} {int(np.floor(float(key.split(' ')[1].strip())))} Optical Depth"]=f"<{logN:.2f}"
            except:
                #line doesn't exist in the data
                rows[0][f"{key.split(' ')[0]} {int(np.floor(float(key.split(' ')[1].strip())))} EW"]=f"---"
                rows[0][f"{key.split(' ')[0]} {int(np.floor(float(key.split(' ')[1].strip())))} Optical Depth"]="---"

        else:
            ew,ew_error=line.actual_ew_func()

            if ew<(2*ew_error):
                non_detected=True
            else:
                non_detected=False

            if non_detected:
                rows[0][f"{key.split(' ')[0]} {int(np.floor(float(key.split(' ')[1].strip())))} EW"]=f"<{int(2*1000*ew_error)}"
            else:
                rows[0][f"{key.split(' ')[0]} {int(np.floor(float(key.split(' ')[1].strip())))} EW"]=f"{(int(1000*ew))} +- {int(1000*ew_error)}"

            #N calc from ew
            wave_cm = line.suspected_line*1e-8
            f=line.f
            m_e = 9.109 * 10**(-28)  # electron mass in grams
            c = 2.998 * 10**10       # speed of light in cm/s
            e = 4.8 * 10**(-10)

            # Conversion constant
            K = (wave_cm**2 / c**2) * (np.pi * e**2 / m_e) * f * 1e8

            # EW and its uncertainty
            # Assume you already have ew and ew_err from previous step
            N = ew / K
            N_err = ew_error / K

            # Logarithmic column density and uncertainty
            logN = np.log10(N)
            logN_err = N_err / (N * np.log(10))

            if non_detected:
                N=(2*ew_error)/K
                temp_logN=np.log10(N)
                rows[0][f"{key.split(' ')[0]} {int(np.floor(float(key.split(' ')[1].strip())))} Optical Depth"]=f"<{temp_logN:.2f}"
            else:
                rows[0][f"{key.split(' ')[0]} {int(np.floor(float(key.split(' ')[1].strip())))} Optical Depth"]=f"{(logN):.2f} +- {(logN_err):.2f}"


    for i in range(n_components):
        if i==0:
            row=rows[0]
        else:
            row = {}

        row['Component'] = i + 1

        # Velocity
        velocity = {row_['param']: row_ for row_ in summary_tables[i] if row_['param'] == 'Velocity'}
        row['Velocity string'] = velocity.get('Velocity', {}).get('map', '---')

        row['Velocity value'] = velocity.get('Velocity', {}).get('map value', '---')
        row['Velocity +1sig'] = velocity.get('Velocity', {}).get('upper sigma value', '---')
        row['Velocity -1sig'] = velocity.get('Velocity', {}).get('lower sigma value', '---')

        for element in elements:
            flag = flags.get(f"flag_component_{i}_{element}", "detected")  # default to 'detected'
            params = {row_['param']: row_ for row_ in summary_tables[i] if element in row_['param']}

            logN = params.get(f"LogN {element}", {})
            b = params.get(f"b {element}", {})

            # Add EW and logN from EW
            ew_val = logN.get('ew', '---')
            logN_from_EW = logN.get('logN', '---')

            row[f"{element} EW"] = ew_val if ew_val is not None else '---'
            row[f"{element} LogN from EW"] = logN_from_EW if logN_from_EW is not None else '---'

            if flag == "saturated":
                row[f"{element} LogN"] = logN.get('p5 value', '---')
                row[f"{element} LogN string"] = f">{logN.get('p5 value', '---')}"

                row[f"{element} LogN +1sig"] = logN.get('upper sigma value','---')
                row[f"{element} LogN -1sig"] = logN.get('lower sigma value','---')

                row[f"{element} b"] = b.get('p95 value', '---')
                row[f"{element} b string"] = f"<{b.get('p95 value', '---')}"

                row[f"{element} b +1sig"] = b.get('upper sigma value', '---')
                row[f"{element} b -1sig"] = b.get('lower sigma value', '---')

                row[f"{element} status"] = flag
            elif flag == "not_detected":
                row[f"{element} LogN"] = b.get('logn_nondetection', '---')
                row[f"{element} LogN string"] = f"<{b.get('logn_nondetection', '---')}"

                row[f"{element} LogN +1sig"] = logN.get('upper sigma value','---')
                row[f"{element} LogN -1sig"] = logN.get('lower sigma value','---')

                row[f"{element} b"] = 10
                row[f"{element} b string"] = f"10"

                row[f"{element} status"] = flag
            else:  # detected
                row[f"{element} LogN string"] = logN.get('map useful','---')

                row[f"{element} LogN"] = logN.get('map value','---')
                row[f"{element} LogN +1sig"] = logN.get('upper sigma value','---')
                row[f"{element} LogN -1sig"] = logN.get('lower sigma value','---')

                row[f"{element} b string"] = b.get('map useful','---')

                row[f"{element} b"] = b.get('map value', '---')
                row[f"{element} b +1sig"] = b.get('upper sigma value', '---')
                row[f"{element} b -1sig"] = b.get('lower sigma value', '---')

                row[f"{element} status"] = 'detected'

                #row[f"{element} LogN"] = logN.get('map useful', '---')
                #row[f"{element} b"] = b.get('map useful', '---')
        if i>0:
            rows.append(row)

    # total section
    if len(rows) >= 1:
        total_row = {"Component": "Total", "Velocity": None}
        ln10 = np.log(10)

        for element in elements:
            has_lt = False
            has_gt = False

            N_detect = []            # linear N for detections (central)
            dlogp = []               # +1σ in dex for detections
            dlogm = []               # -1σ in dex for detections

            N_lower_parts = []       # pieces that contribute to a strict lower bound
            N_upper_parts = []       # pieces that contribute to a strict upper bound

            for r in rows:
                val = str(r.get(f"{element} LogN", "---")).strip()
                if val.startswith("---"):
                    continue

                # classify this component
                if val.startswith("<"):
                    has_lt = True
                    # Component gives an upper limit: add its numeric bound to the total upper bound
                    try:
                        mu_log = float(val[1:])
                        N_upper_parts.append(10.0**mu_log)
                    except:
                        pass
                    # we do NOT add it to N_detect (since it's not a detection)
                    continue

                if val.startswith(">"):
                    has_gt = True
                    # Component gives a lower limit: add its numeric bound to the total lower bound
                    try:
                        mu_log = float(val[1:])
                        N_lower_parts.append(10.0**mu_log)
                    except:
                        pass
                    continue

                # detection: central value in dex already parsed elsewhere
                try:
                    mu_log = float(str(r.get(f"{element} LogN value", val)))
                except:
                    # fallback: extract first number in the string
                    m_mu = re.search(r"([0-9]*\.?[0-9]+)", val)
                    if not m_mu:
                        continue
                    mu_log = float(m_mu.group(1))

                # skip absurd entries
                if mu_log > 20:
                    continue

                N_i = 10.0**mu_log
                N_detect.append(N_i)

                # asymmetric dex errors for detections
                updex = r.get(f"{element} LogN +1sig", None)
                downdex = r.get(f"{element} LogN -1sig", None)
                if updex is not None and downdex is not None:
                    dlogp.append(float(updex))
                    dlogm.append(float(downdex))

                # detections contribute to BOTH lower and upper running bounds
                N_lower_parts.append(N_i)
                N_upper_parts.append(N_i)

            # If nothing at all:
            if len(N_lower_parts) == 0 and len(N_upper_parts) == 0:
                total_row[f"{element} LogN"] = "---"
                total_row[f"{element} b"] = "---"
                continue

            # If any censored values exist
            if has_lt or has_gt:
                N_lower = np.sum(N_lower_parts) if len(N_lower_parts) else 0.0
                N_upper = np.sum(N_upper_parts) if len(N_upper_parts) else 0.0

                if has_gt and not has_lt:
                    # Pure lower limit
                    total_row[f"{element} LogN"] = f">{np.log10(max(N_lower, 1e-99)):.2f}"
                elif has_lt and not has_gt:
                    # Pure upper limit
                    total_row[f"{element} LogN"] = f"<{np.log10(max(N_upper, 1e-99)):.2f}"
                else:
                    # Both present => report an interval
                    lo = np.log10(max(N_lower, 1e-99))
                    hi = np.log10(max(N_upper, 1e-99))
                    total_row[f"{element} LogN"] = f"({lo:.2f}, {hi:.2f})"
            else:
                # All detections: do full nonlinear asymmetric propagation
                N_detect = np.array(N_detect, dtype=float)
                dlogp = np.array(dlogp, dtype=float)
                dlogm = np.array(dlogm, dtype=float)

                N_tot = np.sum(N_detect)

                # per-component linear sigmas
                sigp_i = ln10 * N_detect * dlogp
                sigm_i = ln10 * N_detect * dlogm

                sigp_tot = np.sqrt(np.sum(sigp_i**2)) if len(sigp_i) else 0.0
                sigm_tot = np.sqrt(np.sum(sigm_i**2)) if len(sigm_i) else 0.0

                # transform back to dex (nonlinear)
                N_hi = N_tot + sigp_tot
                N_lo = max(N_tot - sigm_tot, 1e-99)

                logN_tot = np.log10(N_tot)
                dlog_plus  = np.log10(N_hi) - logN_tot
                dlog_minus = logN_tot - np.log10(N_lo)

                total_row[f"{element} LogN string"] = f"{logN_tot:.2f} +- ({dlog_plus:.2f},{dlog_minus:.2f})"
                total_row[f"{element} LogN"] = np.round(logN_tot,2)
                total_row[f"{element} LogN +1sig"] = np.round(dlog_plus,2)
                total_row[f"{element} LogN -1sig"] = np.round(dlog_minus,2)

            total_row[f"{element} b"] = "---"

        rows.append(total_row)


    df = pd.DataFrame(rows)

    # Write CSV
    os.makedirs("static/csv_outputs", exist_ok=True)
    csv_path = os.path.join("static/csv_outputs", f"absorber_summary_{str(load_object('object_name.pkl'))}.csv")
    df.to_csv(csv_path, index=False)

    # --- Save classification flags to CSV ---
    classification_rows = []

    for i in range(n_components):
        row = {"Component": i + 1}
        for element in elements:
            flag = flags.get(f"flag_component_{i}_{element}", "detected")
            row[element] = flag
        classification_rows.append(row)

    # Create dataframe and write to CSV
    classification_df = pd.DataFrame(classification_rows)
    os.makedirs("static/csv_outputs", exist_ok=True)
    class_csv_path = os.path.join("static/csv_outputs", f"classification_flags_{str(load_object('object_name.pkl'))}.csv")
    classification_df.to_csv(class_csv_path, index=False)

    return 1



def column_name_fix(col: str) -> str:

    if 'EW' in col:
        return col + " (mÅ)"
    return col


import re
def format_column_name(col: str) -> str:
    """
    Turn raw column names into LaTeX-friendly headers.
    Assumes columns like:
      - 'MgII LogN', 'FeII b', 'Velocity', 'Galaxy Z', 'Integration Window'
      - Keeps EW columns and others as-is unless you add cases below.
    """

    raw = col.strip()

    if "EW" in raw:
        return raw
    
    # AOD / Optical Depth columns, e.g. "MgII 2803 Optical Depth"
    if re.search(r"\bOptical Depth\b", raw):
        head = re.sub(r"\s*Optical Depth\s*$", "", raw)  # remove trailing "Optical Depth"
        head = head.strip()
        return f"{head} AOD: $\\log_{{10}}(N\\,(\\mathrm{{cm}}^{{-2}}))$"

    # Common simple fields
    if raw.lower() == "velocity" or raw.startswith("Velocity"):
        return r"$\mathit{Velocity}$ (km s$^{-1}$)"
    if raw == "Galaxy Z":
        return r"Galaxy $z$"
    if raw.lower().startswith("integration window"):
        return r"Integration Window (km s$^{-1}$)"
    if "Optical Depth" in raw:
        return r"Optical Depth: $\log_{10} N\,(\mathrm{cm}^{-2})$"

    # Totals from EW (avoid treating 'Total' as an element)
    if "Total LogN from EW" in raw:
        return r"Total $\log_{10} N$ from EW"

    # Try to split "Element Something" like "MgII LogN", "FeII b"
    parts = re.split(r"\s+", raw, maxsplit=1)
    element = parts[0]
    rest = parts[1] if len(parts) > 1 else ""

    # Element logN
    if "LogN" in raw:
        # e.g., "MgII LogN" -> "MgII $\log_{10} N\,(\mathrm{cm}^{-2})$"
        return rf"{element} $\log_{{10}} (N/(\mathrm{{cm}}^{{-2}}))$"

    # Element b
    # match ' b' or exactly 'b' (case sensitive like your columns)
    if rest == "b" or raw.endswith(" b"):
        # e.g., "FeII b" -> "FeII $\mathit{b}$ (km s$^{-1}$)"
        return rf"{element} $\mathit{{b}}$ (km s$^{{-1}}$)"

    # Fallback: return unchanged
    return raw

import re

_ASYMM_MATH_RE = re.compile(
    r'^\s*\$(?P<m>[+-]?\d+(?:\.\d+)?)\s*(?:\\pm|±)\s*\(\s*(?P<hi>[+-]?\d+(?:\.\d+)?)\s*,\s*(?P<lo>[+-]?\d+(?:\.\d+)?)\s*\)\s*\$\s*$'
)

def _decimals(s: str) -> int:
    m = re.search(r'\.(\d+)', s)
    return len(m.group(1)) if m else 0

def asymmify_math(val: str) -> str:
    """
    '$66.8 \\pm (68.0,66.2)$' -> '$66.8^{+1.2}_{-0.6}$'
    Leaves non-matching values unchanged.
    """
    if not isinstance(val, str):
        return val

    m = _ASYMM_MATH_RE.match(val.strip())
    if not m:
        return val

    m_str, hi_str, lo_str = m.group('m', 'hi', 'lo')
    m_val  = float(m_str)
    hi_val = float(hi_str)
    lo_val = float(lo_str)

    # Ensure correct ordering
    #high = max(hi_val, lo_val)
    #low  = min(hi_val, lo_val)

    up   = max(0.0, hi_val)
    down = max(0.0, lo_val)

    # Choose a neat precision based on input
    digits = max(_decimals(m_str), _decimals(hi_str), _decimals(lo_str))
    digits = min(max(digits, 1), 3)  # clamp to 1..3
    fmt = f"{{:.{digits}f}}"

    return f"${fmt.format(m_val)}^{{+{fmt.format(up)}}}_{{-{fmt.format(down)}}}$"


import re
import pandas as pd

_ELEM_LAM_RE = re.compile(r'^(?P<elem>[A-Z][a-z]?(?:I{1,3}|V{1,3}))\s*(?P<lam>\d{3,4})')

def _elem_lambda_from(col: str):
    """
    Return ('MgII', '2796') if the start of the string looks like 'MgII 2796 ...',
    else (None, None).
    """
    m = _ELEM_LAM_RE.match(col.strip())
    if not m:
        return None, None
    return m.group('elem'), m.group('lam')

def _pretty_elem_lambda(elem: str, lam: str) -> str:
    """
    'MgII', '2796' -> 'Mg II $\\lambda2796$'
    """
    # insert a space before trailing Roman numerals
    elem_pretty = re.sub(r'(I{1,3}|V{1,3})$', r' \1', elem)
    return rf'{elem_pretty} $\lambda{lam}$'

def build_ew_multiindex(ew_df: pd.DataFrame) -> pd.DataFrame:
    df = ew_df.copy()

    # Compact left labels
    #rename_left = {}
    #if 'Galaxy Z' in df.columns:
    #    rename_left['Galaxy Z'] = r'$z$'
    #if 'Integration Window' in df.columns:
    #    rename_left['Integration Window'] = r'$\Delta v$ (km s$^{-1}$)'
    #df.rename(columns=rename_left, inplace=True)

    left_cols = [c for c in ['LOS', r"Galaxy $z$", r"Integration Window (km s$^{-1}$)"] if c in df.columns]

    # --- collect EW/AOD pairs with metadata (elem, lambda) ---
    pairs_meta, seen, elem_order = [], set(), {}
    for c in df.columns:
        if 'EW' not in c:
            continue
        elem, lam = _elem_lambda_from(c)  # you already have this helper
        if not elem or not lam:
            continue
        base = f"{elem} {lam}"
        if base in seen:
            continue
        seen.add(base)

        # find partner AOD / Optical Depth column
        partner = None
        pat = re.compile(rf'^{re.escape(elem)}\s*{re.escape(lam)}.*(AOD|Optical\s*Depth)', re.IGNORECASE)
        for cc in df.columns:
            if pat.search(cc):
                partner = cc
                break

        # stable element order: first time we see an element defines its group order
        if elem not in elem_order:
            elem_order[elem] = len(elem_order)

        top = _pretty_elem_lambda(elem, lam)  # e.g., "Mg II $\lambda2796$"
        pairs_meta.append((elem_order[elem], int(lam), top, c, partner))

    # --- sort within element by wavelength (2796 before 2803), keep element group order ---
    pairs_meta.sort(key=lambda t: (t[0], t[1]))

    # --- build ordered columns + MultiIndex (keep your AOD units fix) ---
    ordered = left_cols[:]
    mi = [(col, '') for col in left_cols]

    for _, _, top, ew_col, partner in pairs_meta:
        if ew_col in df.columns:
            ordered.append(ew_col)
            mi.append((top, r'EW (mÅ)'))
        if partner and partner in df.columns:
            ordered.append(partner)
            mi.append((top, r'AOD $\log_{10}(N\,(\mathrm{cm}^{-2}))$'))

    df = df.loc[:, ordered].copy()
    df.columns = pd.MultiIndex.from_tuples(mi)
    return df


def build_colfmt_for_groups(mi) -> str:
    """
    mi: MultiIndex of columns from ew_df_mi
    Returns a LaTeX column_format like 'ccc|cc|cc|...' so element groups are bracketed.
    """
    tops = list(mi.get_level_values(0))
    # count left block at the start (no \lambda in label)
    left_len = 0
    for t in tops:
        if r'\lambda' in t:
            break
        left_len += 1

    # start with left block (centered for compactness)
    fmt = ['c'] * max(left_len, 0)
    if left_len > 0:
        fmt.append('|')  # vertical line before first element group

    # walk remaining columns, grouping by top label
    i = left_len
    while i < len(tops):
        top = tops[i]
        # count how many columns share this top (group size)
        j = i
        while j < len(tops) and tops[j] == top:
            j += 1
        k = j - i  # group width
        fmt.extend(['c'] * k)
        fmt.append('|')  # close this group with a vertical line
        i = j

    return ''.join(fmt)
import re

def enforce_ew_header_rule(tex: str, use_midrule: bool = False) -> str:
    """
    Ensure NO rule between top (element) header row and second (EW/AOD) row,
    and ensure EXACTLY one rule after the EW/AOD row (before data).
    """
    lines = tex.splitlines()

    def is_rule(s: str) -> bool:
        return bool(re.match(r'^\s*\\(hline|midrule)\s*$', s))

    def ends_row(s: str) -> bool:
        return s.rstrip().endswith(r'\\')

    # 1) Find the element (top) header row (first line with \multicolumn ... and row end)
    i_top = None
    for i, ln in enumerate(lines):
        if r'\multicolumn' in ln and ends_row(ln):
            i_top = i
            break
    if i_top is None:
        return tex  # nothing to do (not a multiindex header)

    # 2) Find the next header row (EW/AOD)
    i_second = None
    for j in range(i_top + 1, len(lines)):
        if ends_row(lines[j]):
            i_second = j
            break
    if i_second is None:
        return tex  # no second row found

    # 3) Remove ANY rules between i_top and i_second
    k = i_top + 1
    while k < i_second:
        if is_rule(lines[k]) or not lines[k].strip():  # drop rules/blank lines
            lines.pop(k)
            i_second -= 1
        else:
            k += 1

    # 4) Ensure a rule right AFTER i_second (normalize to \hline unless you really want midrule)
    rule_line = r'\midrule' if use_midrule else r'\hline'
    insert_at = i_second + 1

    if insert_at < len(lines) and is_rule(lines[insert_at]):
        # replace whatever is there with our chosen rule
        lines[insert_at] = rule_line
    else:
        lines.insert(insert_at, rule_line)

    return '\n'.join(lines)


def latex_creation_(dataframes):
    from io import StringIO
    import re

    
    los_names = []
    detections = []

    for df in dataframes:
        # Extract LOS name from 'Object Name' column
        full_name = str(df.get('Object Name', ['Unknown']).iloc[0])
        object_name=full_name[0:full_name.rfind(' ')]
        los_name = full_name.split()[-1] 
        df['LOS']=los_name
        los_names.append(los_name)

        # Detection if there are components
        has_components = "Component" in df.columns and df["Component"].notna().any()
        df['detection?']=has_components
        detections.append(has_components)

    for i,detection in enumerate(detections):
        if detection==True:
            dataframes[i].rename(columns=column_name_fix, inplace=True)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv('combined_df.csv',index=False)

    # Define which columns hold EW measurements
    ew_columns = [col for col in combined_df.columns if "EW" in col]

    # Function to extract just the (<N) part and clean it
    def extract_upper_limit(val):
        if isinstance(val, str):
            match = re.search(r'\(<(\d+)\)', val)
            if match:
                return f"<{match.group(1)}"
        return val

    # Apply this to all EW columns in rows where detection is False
    for col in ew_columns:
        mask = combined_df["detection?"] == False
        combined_df.loc[mask, col] = combined_df.loc[mask, col].apply(extract_upper_limit)

    # Clear repeating LOS entries (only keep the first of each group)
    combined_df['LOS'] = combined_df['LOS'].mask(combined_df['LOS'].duplicated(), '')


    combined_df= combined_df.drop('Object Name', axis=1)
    combined_df= combined_df.drop('detection?', axis=1)

    cols = combined_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('LOS')))
    combined_df = combined_df[cols]

    def clean_val(val):
        if pd.isna(val):
            return ""
        val = str(val).strip()

        if val=='<0':
            return ""

        # Replace all forms of ± early
        val = val.replace('+-', r'\pm').replace('+/-', r'\pm').replace('±', r'\pm')

        # Skip "---" and blanks
        if val == "---" or val == "":
            return ""

        # Wrap in math mode only if relevant
        if any(op in val for op in ['<', '>', r'\pm']):
            return f"${val}$"
        
        return val
    
    df_for_latex = combined_df.copy().applymap(clean_val)

    #split the table into two now
    cols = df_for_latex.columns

    # ---- EW table ----
    ew_df_cols = [c for c in ['LOS', 'Galaxy Z', 'Integration Window',] if c in cols]
    for c in cols:
        #if (('EW' in c) and ('LogN' not in c) and c[5:6].isdigit()):
        if ('(mÅ)' in c) and ("LogN" not in c) and c[5:6].isdigit():
            print(c)
            ew_df_cols.append(c)

            op_dep=c[:-7] + "Optical Depth"

            ew_df_cols.append(op_dep)

    # de-duplicate while preserving order
    ew_df_cols = list(dict.fromkeys(ew_df_cols))

    # --- keep all your code above unchanged up to building ew_df/comp_df ---

    # keep only columns that exist, in your desired order
    existing = [c for c in ew_df_cols if c in df_for_latex.columns]
    missing  = [c for c in ew_df_cols if c not in df_for_latex.columns]
    if missing:
        print(f"Skipping missing columns: {missing}")

    ew_df = df_for_latex.loc[:, existing].copy()

    #ew_df = df_for_latex[ew_df_cols]

    ew_df = ew_df.loc[:, (ew_df != '').any(axis=0)]
    # If you DO want only the first LOS row per sightline, keep the next line; otherwise comment it out.
    ew_df = ew_df[ew_df["LOS"] != ""]

    # 🔧 Don't rename here until format_column_name is fixed
    # ew_df.rename(columns=format_column_name, inplace=True)

    # ---- Component table ----
    comp_df_cols = [c for c in ['LOS','Galaxy Z','Integration Window','Component','Velocity'] if c in cols]
    comp_df_cols += [c for c in cols if ('LogN' in c and 'from EW' not in c) or (' b' in c) or (c == 'b')]
    comp_df_cols = list(dict.fromkeys(comp_df_cols))
    comp_df = df_for_latex[comp_df_cols]

    # choose columns to transform (pre-rename, raw names)
    target_cols = [c for c in comp_df.columns
               if ('Velocity' in c) or
                  ('LogN' in c and 'from EW' not in c) or
                  (c.endswith(' b') or ' b' in c)]

    for c in target_cols:
        comp_df[c] = comp_df[c].apply(asymmify_math)


    # 🔧 Don't rename here either
    # comp_df.rename(columns=format_column_name, inplace=True)

    # Broader upper-limit cleaner (apply BEFORE math wrapping in clean_val in future)
    UL_RE = re.compile(r'\(?\s*<\s*(\d+)\s*\)?')
    def extract_upper_limit(val):
        if isinstance(val, str):
            m = UL_RE.search(val)
            if m: return f"<{m.group(1)}"
        return val

    # If you want to re-run the UL clean on the latex versions:
    for col in [c for c in ew_df.columns if 'EW' in c]:
        ew_df[col] = ew_df[col].apply(extract_upper_limit)

    ew_df.rename(columns=format_column_name, inplace=True)
    comp_df.rename(columns=format_column_name, inplace=True)

    ew_df_mi = build_ew_multiindex(ew_df)
    colfmt   = build_colfmt_for_groups(ew_df_mi.columns)

    ew_table_body = ew_df_mi.to_latex(
        index=False,
        escape=False,
        multicolumn=True,
        multicolumn_format='|c|',
        column_format=colfmt,
    )

    # 🔧 Force the header rules exactly how you want:
    ew_table_body = enforce_ew_header_rule(ew_table_body, use_midrule=False)



    comp_table_body = comp_df.to_latex(index=False, na_rep='', escape=False)

    # --- Safer tabular restyle (no stray "{") ---
    def restyle_tabular(tb: str) -> str:
        m = re.search(r'\\begin{tabular}{([^}]*)}', tb)
        if m:
            spec = m.group(1)
            new_spec = '|'.join(['c'] * len(spec))
            tb = re.sub(r'\\begin{tabular}{[^}]*}', rf'\\begin{{tabular}}{{|{new_spec}|}}', tb)

        tb = tb.replace(r'\toprule','').replace(r'\midrule','').replace(r'\bottomrule','')
        # Add \hline after header row only (first \\)
        tb = re.sub(r'\\\\\n', r'\\\\\n\\hline\n', tb, count=1)
        # Add \hline before \end{tabular}
        tb = re.sub(r'\n\\end{tabular}', r'\n\\hline\n\\end{tabular}', tb)
        return tb

    ew_table_body   = restyle_tabular(ew_table_body)
    comp_table_body = restyle_tabular(comp_table_body)

    def wrap_table(body: str, caption: str, label: str) -> str:
        # 🔧 IMPORTANT: keep the opening brace with \resizebox and DROP the '%' to avoid weirdness
        return (
            "\\begin{table*}\n"
            "\\centering\n"
            "\\renewcommand{\\arraystretch}{1.3}\n"
            "\\small\n"
            "\\begin{adjustbox}{max width=\linewidth}\n"
            f"{body}\n"
            "}\n"
            "\\end{adjustbox}"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            "\\end{table*}\n"
        )
    
    def preamble():
        # raw triple-quoted string = no need to escape backslashes
        return r"""
        \documentclass{article}
        \usepackage[utf8]{inputenc}  % keep if using pdfLaTeX; drop if XeLaTeX/LuaLaTeX
        \usepackage{multirow}
        \usepackage{booktabs}
        \usepackage{amsmath}
        \usepackage{graphicx}
        \usepackage{adjustbox}
        \usepackage[T1]{fontenc}
        \begin{document}
        """
    
    def enddoc():
        return r"\end{document}" + "\n"

    ew_full_latex   = wrap_table(ew_table_body,   "Equivalent-width summary per LOS.", "tab:ew")
    comp_full_latex = wrap_table(comp_table_body, "Component-level measurements.",      "tab:components")

    #full_latex=preamble() + ew_full_latex + "\n\n" + comp_full_latex + "\\end{document}\n"
    full_latex = "\n".join([preamble(), ew_full_latex, "", comp_full_latex, enddoc()])

    with open(f"latex_tables/{object_name}_full_latex.tex", "w") as f: f.write(full_latex)
    with open(f"latex_tables/{object_name}_full_latex.txt", "w") as f: f.write(full_latex)

    with open("equivalent_width_latex.txt", "w") as f: f.write(ew_full_latex)
    with open("component_analysis_latex.txt", "w") as f: f.write(comp_full_latex)

    latex_output=ew_full_latex + "\n\n" + comp_full_latex

    return latex_output

import os, re

def los_sort_key(path):
    """
    Key: (letter, number or -1 if none).
    Matches a trailing A-D with optional digits at the *end* of the stem.
    Examples: '..._A.csv' -> ('A', -1), '..._A1.csv' -> ('A', 1)
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r'([A-Da-d]\d*)$', stem)
    if m:
        tag = m.group(1).upper()
        letter = tag[0]
        num = int(tag[1:]) if len(tag) > 1 else -1
        return (letter, num)
    # If no LOS is found, push to the end (but keep a stable tie-breaker)
    return ('Z', float('inf'))

def do_all_latex_():

    blacklist=[]

    data_source = '/Users/jakereinheimer/Desktop/Fakhri/Best_fits/'

    for obj in os.listdir(data_source):

        if obj in blacklist:
            continue

        if obj.startswith('.'):
            continue

        obj_path = os.path.join(data_source, obj)   # <- correct join
        if not os.path.isdir(obj_path):
            continue

        # collect just the absorber CSVs (full paths)
        csv_files = []
        for fname in os.listdir(obj_path):
            if not fname.lower().endswith('.csv'):
                continue
            low = fname.lower()
            if ('absorber_summary' in low) or ('absorber_data' in low):
                csv_files.append(os.path.join(obj_path, fname))

        # sort *only* by LOS at end of stem (A, A1, A2, B, C, ...)
        ordered_csvs = sorted(csv_files, key=los_sort_key)

        print(obj)
        print(ordered_csvs)

        # pass the ordered list of full paths to your function
        latex_creation_(inp=ordered_csvs)

    #pdf them
    import shlex, subprocess

    output_dir='/Users/jakereinheimer/Desktop/Fakhri/VPFit/latex_tables'
    files=os.listdir(output_dir)
    for file in files:

        full_path=os.path.join(output_dir,file)

        if full_path.endswith('.tex'):

            cmd = f"latexmk -pdf {shlex.quote(str(full_path))} -outdir={output_dir}"
            subprocess.run(cmd, shell=True, check=True)

    return 1