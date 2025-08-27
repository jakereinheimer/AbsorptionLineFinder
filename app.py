from flask import Flask, render_template, request, url_for, redirect, session,send_from_directory,jsonify,json
import pickle
import base64
import os
from werkzeug.utils import secure_filename
from flask import flash
import numpy as np
import multiprocessing
from matplotlib.widgets import SpanSelector
import math
import matplotlib.pyplot as plt
import pandas as pd

from VPFit import VPFit
from TNG_trident import Sim_spectra
from essential_functions import clear_directory,get_data,floor_to_wave,read_atomDB
from mcmc import pre_mcmc,update_fit,mcmc, plot_fits
from AbsorptionLine import AbsorptionLineSystem

AtomDB=read_atomDB()

#helper functions
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Open the file in binary write mode
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)  # Pickle the object and write to file

def load_object(filename):
    with open(filename, 'rb') as inp:  # Open the file in binary read mode
        return pickle.load(inp)  # Return the unpickled object

def list_catalogs():
    catalogs=os.listdir('/Users/jakereinheimer/Desktop/Fakhri/data')

    try:
        catalogs.remove(".DS_Store")
    except:
        pass

    return catalogs

def list_spectra(catalog):
    spectra_directory = os.path.join('/Users/jakereinheimer/Desktop/Fakhri/data', catalog)
    spectra = os.listdir(spectra_directory)

    try:
        spectra.remove(".DS_Store")
    except:
        pass

    return sorted(spectra)

def list_velocity_plots():
    directory = os.path.join(app.static_folder, 'Data/velocity_plots')
    files = os.listdir(directory)
    png_files = [f for f in files if f.endswith('.png')]
    elements = set(file.split('_')[1] for file in png_files if '_' in file)  # Extract the element name part
    return elements

# Add to the top where other imports are
ALLOWED_EXTENSIONS = {'txt', 'fits', 'csv','xspec'}  # Adjust based on the file types you expect

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def quantize_file(folder_loc):
    dir=os.listdir(folder_loc)

    unique_lines=set()

    for file_name in dir:
        line_type = file_name.split(',')[0]
        unique_lines.add(line_type)

    return len(unique_lines)

def clean_house():
    try:
        clear_directory('static/Data/velocity_plots')
        clear_directory('static/Data/mcmc')
        os.remove('static/Data/FluxPlot.html')
        clear_directory('found_lines/')
        clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/Absorbers/csvs')
        clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/Absorbers/objs')
        clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/Absorbers/vel_plots')

    except:
        print('failed to delete everything')
        pass



app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

from multiprocessing import Manager


@app.route('/<filename>')
def download_file(filename):
    # Be sure to validate the filename to avoid security risks like path traversal
    return send_from_directory('static/Data/mcmc/', filename, as_attachment=True)

@app.route('/', methods=['GET'])
def index():
    selected_catalog = request.args.get('catalog', default=None)
    catalogs = list_catalogs()

    if selected_catalog is None or selected_catalog not in catalogs:
        selected_catalog = catalogs[0]

    session['selected_catalog'] = selected_catalog  # Store the selected catalog in session
    spectra_list = list_spectra(selected_catalog)
    return render_template('index.html', catalogs=catalogs, selected_catalog=selected_catalog, spectra=spectra_list)

@app.route('/upload_combined', methods=['POST'])
def upload_combined():
    clean_house()

    red_files = request.files.getlist('red_files')
    blue_files = request.files.getlist('blue_files')
    red_res = float(request.form.get('red_res', 7.5))
    blue_res = float(request.form.get('blue_res', 7.5))
    object_name=request.form.get('object_name',' ')
    save_object(object_name,'object_name.pkl')
    nmf_requested = 'nmf' in request.form

    upload_dir = '/Users/jakereinheimer/Desktop/Fakhri/data/custom/'

    def save_uploaded(files):
        paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(upload_dir, filename)
                file.save(filepath)
                paths.append(filepath)
        return paths

    red_paths = save_uploaded(red_files)
    if len(red_paths)==0:
        red_paths=None
    
    blue_paths = save_uploaded(blue_files)
    if len(blue_paths)==0:
        blue_paths=None


    if nmf_requested:
        name='nmf'
    else:
        name=''

    vp = VPFit(
        (red_paths, blue_paths, red_res, blue_res),
        'combined',
        name,
        custom=True,
    )

    vp.DoAll()

    save_object(vp, 'current_vp.pkl')
    save_object({}, 'custom_absorptions.pkl')

    return redirect(url_for('show_results'))

@app.route('/view_plot',methods=['POST'])
def view_plots():

    '''
    I want the user to select a folder and enter in a z value, and this will spit out all of the views and do MgII, FeII, and MgI
    '''

    clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/viewplot_upload/user_upload')

    uploaded_files = request.files.getlist("object_dir")
    galaxy_z = float(request.form.get("GalZ"))

    save_dir = os.path.join("viewplot_upload", "user_upload")
    os.makedirs(save_dir, exist_ok=True)

    #upload and save the files
    file_paths = []
    for file in uploaded_files:
        filename = file.filename[file.filename.find('/')+1:-1]  # includes the relative path from the selected folder
        if 'DS_Stor' in filename:
            continue
        filepath = os.path.join(save_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)
        file_paths.append(filepath)

    #go through the LOSs and creates vpfit objects
    vpfits=[]
    vpfits_names=[]
    for los in os.listdir(save_dir):

        los_dir=os.path.join(save_dir,los)
        
        blue_dir=os.path.join(los_dir,'b')
        red_dir=os.path.join(los_dir,'r')

        os.makedirs(blue_dir,exist_ok=True)
        os.makedirs(red_dir,exist_ok=True)

        red_paths_=os.listdir(red_dir)
        blue_paths_=os.listdir(blue_dir)

        red_paths=[os.path.join(red_dir,path) for path in red_paths_]
        blue_paths=[os.path.join(blue_dir,path) for path in blue_paths_]

        if len(red_paths)==0:
            red_paths=None

        if len(blue_paths)==0:
            blue_paths=None

        vp = VPFit(
        (red_paths, blue_paths, 10, 10),
        'combined',
        los,
        custom=True,
        )

        vpfits.append(vp)
        vpfits_names.append(los)


    #now to create the plot
    species=vpfits[0].view_plot_data(galaxy_z).keys()
    fig, axs = plt.subplots(len(species), len(vpfits), figsize=(15, 8 * len(vpfits)), squeeze=True,sharex=True,sharey=True)

    for i,los in enumerate(vpfits):

        view_data=los.view_plot_data(galaxy_z)

        axs[0, i].set_title(str(vpfits_names[i]))

        for j,species in enumerate(view_data.keys()):

            ax=axs[j,i]
            ax.set_ylim(0,2)

            data=view_data.get(species)
            
            #flux
            ax.step(data[0], data[1], where='mid', label=f"Flux", color="black")
            #error
            ax.step(data[0], data[2], where='mid', label=f"Flux", color="cyan")

            #big red line
            ax.axvline(0,color='red')

            if i==0:
                ax.text(0.05, 0.2, f"{species}", transform=ax.transAxes,fontsize=8)

    plt.subplots_adjust(hspace=0,wspace=0)
    plt.savefig(f"viewplot.png")

    return index()

'''
Chat gpt optimization
'''
'''
from joblib import Parallel, delayed
import corner

def gelman_rubin(chains):
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

def save_plots_and_tables(comp_idx, comp_chain, flattened, num_params_per_line, column_names, statuses, line_dict, mcmc_lines, summary_tables):
    """
    Saves trace plots, corner plots, and summary tables for each component.
    """
    # --- TRACE PLOTS ---
    fig, axs = plt.subplots(num_params_per_line, figsize=(10, 2 * num_params_per_line), sharex=True)
    for j in range(num_params_per_line):
        axs[j].plot(flattened[:, j], alpha=0.5, lw=0.3)

        param_chain = comp_chain[:, :, j]
        r_hat = gelman_rubin(param_chain)
        axs[j].text(0.02, 0.02, f"R = {r_hat:.3f}", transform=axs[j].transAxes)

        axs[j].set_ylabel(f'{column_names[j]} ({statuses[comp_idx, j]})')

    axs[-1].set_xlabel("Step")
    fig.suptitle(f"Component {comp_idx + 1} Trace")
    trace_path = os.path.join('static/chain_review/trace', f"trace_component_{comp_idx}.png")
    fig.tight_layout()
    fig.savefig(trace_path)
    plt.close(fig)

    # --- CORNER PLOT ---
    comp_chain_flat = comp_chain.reshape(-1, comp_chain.shape[-1])
    fig = corner.corner(
        comp_chain_flat,
        labels=[f'{column_names[j]} ({statuses[comp_idx, j]})' for j in range(num_params_per_line)],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    corner_path = os.path.join('static/chain_review/triangle', f"triangle_component_{comp_idx}.png")
    fig.savefig(corner_path)
    plt.close(fig)

    # --- Summary table ---
    summary_table = []
    for j in range(num_params_per_line):
        samples = comp_chain_flat[:, j]

        # Calculate EW for relevant line (avoiding redundant computations)
        parameter = column_names[j]
        if parameter != 'Velocity':
            line = get_line_from_dict(parameter, line_dict)

            if line:
                eq_width, logN_from_EW = calculate_ew_and_logN(line, mcmc_lines, comp_idx)
            else:
                eq_width, logN_from_EW = None, None
        else:
            eq_width, logN_from_EW = None, None

        median = np.percentile(samples, 50)
        low = np.percentile(samples, 16)
        high = np.percentile(samples, 84)
        p95 = np.percentile(samples, 95)
        p5 = np.percentile(samples, 5)

        summary_table.append({
            "param": column_names[j],
            "median": f"{median:.3f} ± {high - median:.3f}/{median - low:.3f}",
            "p95": f"{p95:.3f}",
            "p5": f"{p5:.3f}",
            "ew": eq_width,
            "logN": logN_from_EW
        })

    summary_tables.append(summary_table)
    return summary_tables

# Helper function to extract line based on parameter
def get_line_from_dict(parameter, line_dict):
    if "MgII" in parameter:
        return line_dict.get('MgII 2796.355099')
    elif "FeII" in parameter:
        return line_dict.get('FeII 2600.1720322', line_dict.get('FeII 2586.6492304', line_dict.get('FeII 2382.7639122')))
    elif "CaII" in parameter:
        return line_dict.get('CaII 3934.774716')
    elif "MgI" in parameter:
        return line_dict.get('MgI 2852.96342')
    return None

# Helper function to calculate EW and LogN
def calculate_ew_and_logN(line, mcmc_lines, comp_idx):
    vmin, vmax = mcmc_lines[comp_idx].vel_range
    mask = (line.velocity >= vmin) & (line.velocity <= vmax)
    if not np.any(mask):
        return None, None

    lam = line.wavelength[mask]
    f = line.flux[mask]
    f_err = line.errors[mask]  # Ensure this exists

    dlam = np.gradient(lam)  # Compute Δλ
    ew = np.sum((1 - f) * dlam)  # Compute equivalent width

    ew_err = np.sqrt(np.sum((dlam * f_err) ** 2))  # Compute error on EW

    wave_cm = line.suspected_line * 1e-8
    f = line.f
    m_e = 9.109 * 10 ** (-28)  # electron mass in grams
    c = 2.998 * 10 ** 10  # speed of light in cm/s
    e = 4.8 * 10 ** (-10)

    K = (wave_cm ** 2 / c ** 2) * (np.pi * e ** 2 / m_e) * f * 1e8  # Conversion constant

    N = ew / K
    N_err = ew_err / K

    logN = np.log10(N)
    logN_err = N_err / (N * np.log(10))

    return f"{ew:.3f} +/- {ew_err:.3f}", f"{logN:.3f} +/- {logN_err:.3f}"

# Run everything in parallel for each component
def run_parallel_processing(n_components, chain, flattened, num_params_per_line, column_names, statuses, line_dict, mcmc_lines):
    summary_tables = Parallel(n_jobs=-1)(
        delayed(save_plots_and_tables)(comp_idx, chain[:, :, comp_idx * num_params_per_line: (comp_idx + 1) * num_params_per_line], flattened, num_params_per_line, column_names, statuses, line_dict, mcmc_lines, [])
        for comp_idx in range(n_components)
    )
    return summary_tables

@app.route('/chain_upload', methods=['POST'])
def chain_upload():
    import os
    from flask import request

    try:
        clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/chain_upload/user_upload')
        clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/chain_review/trace')
        clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/chain_review/triangle')
    except:
        pass

    uploaded_files = request.files.getlist("object_dir")
    object_name = request.form.get('object_name', ' ')
    save_object(object_name, 'object_name.pkl')

    save_dir = os.path.join("chain_upload", "user_upload")
    os.makedirs(save_dir, exist_ok=True)

    # Upload and save the files
    file_paths = []
    for file in uploaded_files:
        filename = file.filename[file.filename.find('/') + 1:]
        if 'DS_Stor' in filename:
            continue
        filepath = os.path.join(save_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)
        file_paths.append(filepath)

    # Loading the data
    chain = np.load(os.path.join(save_dir, "chain.npy"))
    mcmc_lines = load_object(os.path.join(save_dir, 'final/mcmc_lines.pkl'))
    line_dict = load_object(os.path.join(save_dir, 'final/line_dict.pkl'))
    params = load_object(os.path.join(save_dir, 'final/initial_guesses.pkl'))
    elements = load_object(os.path.join(save_dir, 'initial/initial_element_list.pkl'))
    num_params_per_line = 1 + 2 * len(elements)
    params = np.array(params).reshape(-1, num_params_per_line)
    statuses = np.array(load_object(os.path.join(save_dir, 'initial/initial_statuses.pkl')))
    column_names = load_object(os.path.join(save_dir, 'initial/column_names.pkl'))

    # Initialize marginalized_flags as an empty dictionary or with any default values
    marginalized_flags = {i: False for i in range(chain.shape[-1] // num_params_per_line)}

    # Flatten the chain for the plot
    flattened = chain.reshape(-1, chain.shape[-1])  # shape: (10100*250, 10)
    median_params = np.median(flattened, axis=0)
    plot_fits(median_params, line_dict, elements, mcmc_lines, 'initial_fit_plot', chain_review=True, show_components=True)

    n_components = chain.shape[-1] // num_params_per_line
    summary_tables = run_parallel_processing(n_components, chain, flattened, num_params_per_line, column_names, statuses, line_dict, mcmc_lines)

    save_object(summary_tables, 'static/chain_review/summary_tables.pkl')

    return render_template(
        "chain_review.html",
        n_components=n_components,
        summary_tables=summary_tables,
        elements=elements,
        marginalized_flags=marginalized_flags
    )'''


"""
end chat gpt optimization
"""

@app.route('/chain_upload',methods=['POST'])
def chain_upload():
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


    try:
        clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/chain_upload/user_upload')
        clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/chain_review/trace')
        clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/chain_review/triangle')
    except:
        pass

    uploaded_files = request.files.getlist("object_dir")

    object_name=request.form.get('object_name',' ')
    save_object(object_name,'object_name.pkl')

    save_dir = os.path.join("chain_upload", "user_upload")
    os.makedirs(save_dir, exist_ok=True)

    #upload and save the files
    file_paths = []
    for file in uploaded_files:
        filename = file.filename[file.filename.find('/')+1:]  # includes the relative path from the selected folder
        if 'DS_Stor' in filename:
            continue
        filepath = os.path.join(save_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)
        file_paths.append(filepath)

    chain=np.load(os.path.join(save_dir,"chain.npy"))
    print('chain shape')
    print(chain.shape)

    burn = int(chain.shape[0] * 0.2)
    chain = chain[burn:, :, :]

    mcmc_lines=load_object(os.path.join(save_dir,'final/mcmc_lines.pkl'))
    line_dict=load_object(os.path.join(save_dir,'final/line_dict.pkl'))
    params = load_object(os.path.join(save_dir,'final/initial_guesses.pkl'))
    elements = load_object(os.path.join(save_dir,'initial/initial_element_list.pkl'))
    num_params_per_line = 1 + 2 * len(elements)
    params = np.array(params).reshape(-1, num_params_per_line)
    statuses = np.array(load_object(os.path.join(save_dir,'initial/initial_statuses.pkl')))
    column_names=load_object(os.path.join(save_dir,'initial/column_names.pkl'))

    # chain: shape (steps, walkers, parameters)
    flattened = chain.reshape(-1, chain.shape[-1])  # shape: (10100*250, 10)
    median_params = np.median(flattened, axis=0)
    #plot_fits(median_params,line_dict,elements,mcmc_lines,'initial_fit_plot',chain_review=True)

    # --- Map Values ---
    data_csv = pd.read_csv(os.path.join(save_dir,'absorber_data.csv'))

    components = data_csv['Component'].unique()
    components.sort()

    map_list = []
    for comp in components:
        if comp == 'Total':
            continue
        row = []
        comp_data = data_csv[data_csv['Component'] == comp]

        # MAP velocity
        vel = comp_data['MAP Velocity'].values[0]
        row.append(vel)

        for param in column_names:
            if param == 'Velocity':
                continue

            if param.split(' ')[0] == 'b':
                map_column_name = "MAP " + param.split(' ')[1] + ' ' + param.split(' ')[0] + ' (km/s)'
            else:
                map_column_name = "MAP " + param.split(' ')[1] + ' ' + param.split(' ')[0]

            row.append(comp_data[map_column_name].values[0])

        map_list.append(row)

    map_params = np.array(map_list)

    plot_fits(map_params.flatten(),line_dict,elements,mcmc_lines,'initial_fit_plot',chain_review=True)

    # --- End Map Values ___

    all_summary_tables=[]
    n_components = chain.shape[-1] // num_params_per_line

    marginalized_flags = {}

    detection_flags={}

    for comp_idx in range(n_components):
        start = comp_idx * num_params_per_line
        end = (comp_idx + 1) * num_params_per_line
        comp_chain = chain[:, :, start:end]

        comp_chain_flat = comp_chain.reshape(-1, comp_chain.shape[-1])

        # --- TRACE PLOTS ---
        fig, axs = plt.subplots(num_params_per_line, figsize=(10, 2 * num_params_per_line), sharex=True)
        for j in range(num_params_per_line):
            
            axs[j].plot(flattened[:, j], alpha=0.5, lw=0.3)

            param_chain = comp_chain[:, :, j]
            r_hat = gelman_rubin(param_chain)
            axs[j].text(0.02,0.02,f"R = {r_hat:.3f}", transform=axs[j].transAxes)

            axs[j].set_ylabel(f'{column_names[j]} ({statuses[comp_idx,j]})')

        axs[-1].set_xlabel("Step")
        fig.suptitle(f"Component {comp_idx + 1} Trace")
        trace_path = os.path.join('static/chain_review/trace', f"trace_component_{comp_idx}.png")
        fig.tight_layout()
        fig.savefig(trace_path)
        plt.close(fig)

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
                dlam = np.gradient(lam)

                # Compute equivalent width
                ew = np.sum((1 - f) * dlam)

                # Compute error on EW
                ew_err = np.sqrt(np.sum((dlam * f_err) ** 2))

                detection_flag=detection_flags.get(comp_idx).get(element)
                if detection_flag is None:

                    if ew<2*ew_err:
                        detection_flags.get(comp_idx)[element]="not_detected"

                    elif ew>.4:
                        detection_flags.get(comp_idx)[element]="saturated"

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

                eq_width = f"{ew:.3f} +/- {ew_err:.3f}"
                logN_from_EW = f"{logN:.3f} +/- {logN_err:.3f}"

            else:
                eq_width = None
                logN_from_EW = None

            if 'b' in column_names[j]:

                tol=1.0

                logn_samples=comp_chain[:, j-1]
                b_samples=comp_chain[:, j]

                mask = np.isfinite(logn_samples) & np.isfinite(b_samples) & (np.abs(b_samples - 10.0) <= tol)

                try:
                    logn_nondetection=float(np.percentile(logn_samples[mask], 5))
                except:
                    logn_nondetection=1.0
            else:
                logn_nondetection=1.0
            

            median = np.percentile(samples, 50)
            low = np.percentile(samples, 16)
            high = np.percentile(samples, 84)
            map_val = map_params[comp_idx,j]
            p95 = np.percentile(samples, 95)
            p5 = np.percentile(samples, 5)

            if 'LogN' in column_names[j]:

                if map_val<10:
                    detection_flags.get(comp_idx)[element]="not_detected"

                if map_val>13.8:
                    detection_flags.get(comp_idx)[element]="saturated"


            summary_table.append({
                "param": column_names[j],
                "median": f"{median:.3f} ± ({high - median:.3f},{median - low:.3f})",
                "map": f"{map_val:.3f}" + f" ± ({high:.3f},{low:.3f})",
                "p95": f"{p95:.3f}",
                "p5": f"{p5:.3f}",
                "ew": eq_width,
                "logN": logN_from_EW,
                "logn_nondetection":f'{logn_nondetection:.3f}'
            })
        all_summary_tables.append(summary_table)

    save_object(all_summary_tables,'static/chain_review/summary_tables.pkl')

    return render_template(
    "chain_review.html",
    n_components=n_components,
    summary_tables=all_summary_tables,
    detection_flags=detection_flags,
    elements=elements,
    marginalized_flags=marginalized_flags
    )


@app.route('/marginalize_component/<int:component_index>', methods=['POST'])
def marginalize_component(component_index):
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import corner

    selected_element = request.form.get("reference_element", None)

    summary_path = 'static/chain_review/summary_tables.pkl'
    chain_path = 'chain_upload/user_upload/chain.npy'
    column_path = 'chain_upload/user_upload/initial/column_names.pkl'
    status_path = 'chain_upload/user_upload/initial/initial_statuses.pkl'
    element_path = 'chain_upload/user_upload/initial/initial_element_list.pkl'
    marg_path = 'chain_upload/user_upload/marginalized_flags.pkl'

    chain = np.load(chain_path)
    summary_tables = load_object(summary_path)
    column_names = load_object(column_path)
    statuses = np.array(load_object(status_path))
    elements = load_object(element_path)

    element_masses = {
        'HI': 1.0079, 'MgII': 24.305, 'FeII': 55.845, 'MgI': 24.305,
        'CIV': 12.011, 'SiII': 28.0855, 'OVI': 15.999, 'CaII': 40.078
        # Add any others if neccesary 
    }

    num_params_per_line = (chain.shape[-1]) // len(summary_tables)
    start = component_index * num_params_per_line
    end = (component_index + 1) * num_params_per_line

    comp_chain = chain[:, :, start:end].reshape(-1, num_params_per_line)

    print(column_names)
    print(elements)

    element_b_indices = {
    el: j
    for el in elements
    for j, name in enumerate(column_names)
    if name.startswith("b ") and el in name
    }

    print('selected element')
    print(selected_element)
    print(element_b_indices)

    if selected_element and selected_element in element_b_indices:
        ref_el = selected_element
        ref_idx = element_b_indices[ref_el]
    else:
        ref_el, ref_idx = sorted(element_b_indices.items(), key=lambda x: element_masses.get(x[0], 999))[0]

    b_ref_vals = comp_chain[:, ref_idx]
    b_ref_min = np.percentile(b_ref_vals,14)

    b_floors = {}
    for el, idx in element_b_indices.items():
        if el == ref_el:
            b_floors[idx] = 0.0#b_ref_min
        else:
            m_ratio = np.sqrt(element_masses.get(ref_el, 1.0) / element_masses.get(el, 1.0))
            b_floors[idx] = b_ref_min * m_ratio

    # Step 3: Apply masking
    mask = np.ones(comp_chain.shape[0], dtype=bool)
    for idx, floor in b_floors.items():
        mask &= comp_chain[:, idx] >= floor

    comp_chain = comp_chain[mask]

    # Save updated full chain with this component replaced
    full_chain = np.load(chain_path)
    full_chain_flat = full_chain.reshape(-1, full_chain.shape[-1])

    # Replace only the masked rows in the component slice
    filtered_comp_chain = np.copy(full_chain_flat[:, start:end])
    for idx, floor in b_floors.items():
        below = filtered_comp_chain[:, idx] < floor
        filtered_comp_chain[below, idx] = np.nan  # or use floor, or 0 if preferred

    full_chain_flat[:, start:end] = filtered_comp_chain
    updated_chain = full_chain_flat.reshape(full_chain.shape)
    np.save(chain_path, updated_chain)

    # Save just the reference b_floor for plotting
    try:
        marginalized_flags = load_object(marg_path)
    except FileNotFoundError:
        marginalized_flags = {}

    marginalized_flags[component_index] = {
    "b_floor": float(b_ref_min),
    "reference_element": ref_el
    }

    save_object(marginalized_flags, marg_path)

    # Regenerate trace plot
    fig, axs = plt.subplots(num_params_per_line, figsize=(10, 2 * num_params_per_line), sharex=True)
    component_column_names = column_names
    component_statuses = statuses[component_index, :num_params_per_line]
    for j in range(num_params_per_line):
        axs[j].plot(comp_chain[:, j], alpha=0.5, lw=0.3)
        axs[j].set_ylabel(f'{component_column_names[j]} ({component_statuses[j]})')
        if j in b_floors:
            axs[j].axhline(b_floors[j], color='red', linestyle='--')
    axs[-1].set_xlabel("Step")
    fig.suptitle(f"Component {component_index + 1} Trace")
    trace_path = os.path.join('static/chain_review/trace', f"trace_component_{component_index}.png")
    fig.tight_layout()
    fig.savefig(trace_path)
    plt.close(fig)

    # Regenerate corner plot
    fig = corner.corner(
        comp_chain,
        labels=[f'{column_names[j]} ({statuses[component_index,j]})' for j in range(num_params_per_line)],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    axes = np.array(fig.axes).reshape((num_params_per_line, num_params_per_line))
    for i in range(num_params_per_line):
        if i in b_floors:
            ax = axes[i, i]
            ax.axvline(b_floors[i], color='red', linestyle='--', linewidth=1)
    corner_path = os.path.join('static/chain_review/triangle', f"triangle_component_{component_index}.png")
    fig.savefig(corner_path)
    plt.close(fig)

    # Update summary
    summary = []
    for j in range(num_params_per_line):
        samples = comp_chain[:, j]
        median = np.percentile(samples, 50)
        low = np.percentile(samples, 16)
        high = np.percentile(samples, 84)
        p95 = np.percentile(samples, 95)
        p5 = np.percentile(samples, 5)

        if 'b' in column_names[j]:

            tol=1.0

            logn_samples=comp_chain[:, j-1]
            b_samples=comp_chain[:, j]

            mask = np.isfinite(logn_samples) & np.isfinite(b_samples) & (np.abs(b_samples - 10.0) <= tol)

            logn_nondetection=float(np.percentile(logn_samples[mask], 95))
        else:
            logn_nondetection=None


        summary.append({
            "param": column_names[j],
            "median": f"{median:.3f} ± {high - median:.3f}/{median - low:.3f}",
            "p95": f"{p95:.3f}",
            "p5": f"{p5:.3f}",
            "ew": summary_tables[component_index][j].get("ew"),
            "logN": summary_tables[component_index][j].get("logN"),
            "logn_nondetection":f'{logn_nondetection}'
        })

    summary_tables[component_index] = summary
    save_object(summary_tables, summary_path)

    return render_template(
        "chain_review.html",
        n_components=len(summary_tables),
        summary_tables=summary_tables,
        elements=elements,
        marginalized_flags=marginalized_flags
    )



@app.route('/generate_csv', methods=['POST'])
def generate_csv():

    import os
    from flask import request, render_template
    import pandas as pd
    import numpy as np
    import re

    flags = request.form.to_dict()

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


    for key,value in line_dict.items():

        ew,ew_error=value.actual_ew_func()

        rows[0][f"{key.split(' ')[0]} {int(np.floor(float(key.split(' ')[1].strip())))} EW"]=f"{(1000*ew):.2f} +- {(1000*ew_error):.2f}"

        #N calc from ew
        wave_cm = value.suspected_line*1e-8
        f=value.f
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

        rows[0][f"{key.split(' ')[0]} {int(np.floor(float(key.split(' ')[1].strip())))} Total LogN from EW"]=f"{(logN):.2f} +- {(logN_err):.2f}"

    for i in range(n_components):
        if i==0:
            row=rows[0]
        else:
            row = {}

        row['Component'] = i + 1

        # Velocity
        velocity = {row_['param']: row_ for row_ in summary_tables[i] if row_['param'] == 'Velocity'}
        row['Velocity (km/s)'] = velocity.get('Velocity', {}).get('map', '---')

        for element in elements:
            flag = flags.get(f"flag_component_{i}_{element}", "detected")  # default to 'detected'
            print('flag')
            print(flag)
            params = {row_['param']: row_ for row_ in summary_tables[i] if element in row_['param']}

            logN = params.get(f"LogN {element}", {})
            b = params.get(f"b {element}", {})

            # Add EW and logN from EW
            ew_val = logN.get('ew', '---')
            logN_from_EW = logN.get('logN', '---')

            row[f"{element} EW"] = ew_val if ew_val is not None else '---'
            row[f"{element} LogN from EW"] = logN_from_EW if logN_from_EW is not None else '---'

            if flag == "saturated":
                row[f"{element} LogN"] = f">{logN.get('p5', '---')}"
                row[f"{element} b (km/s)"] = f"<{b.get('p95', '---')}"
            elif flag == "not_detected":
                row[f"{element} LogN"] = f"<{b.get('logn_nondetection', '---')}"
                row[f"{element} b (km/s)"] = "10 +- 1"
            else:  # detected
                row[f"{element} LogN"] = logN.get('map', '---')
                row[f"{element} b (km/s)"] = b.get('map', '---')
        if i>0:
            rows.append(row)

    # Optional: Total row
    if len(rows) > 1:
        total_row = {"Component": "Total", "Velocity (km/s)": None}
        for element in elements:
            
            N_values = []
            for r in rows:
                val = str(r.get(f"{element} LogN", "---"))
                if val.startswith("---"):
                    continue
                match = re.search(r"[\d.]+", val)
                if match:
                    try:
                        N_values.append(10**float(match.group()))
                    except ValueError:
                        continue
            
            if len(N_values) > 0:
                total_row[f"{element} LogN"] = f"{np.log10(np.sum(N_values)):.2f}"
            else:
                total_row[f"{element} LogN"] = "---"

            total_row[f"{element} b (km/s)"] = "---"

        rows.append(total_row)

    df = pd.DataFrame(rows)

    # Write CSV
    os.makedirs("static/csv_outputs", exist_ok=True)
    csv_path = os.path.join("static/csv_outputs", f"absorber_summary_{str(load_object('object_name.pkl'))}.csv")
    df.to_csv(csv_path, index=False)

    marg_path = '/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/chain_review/marginalized_flags.pkl'
    try:
        marginalized_flags = load_object(marg_path)
    except FileNotFoundError:
        marginalized_flags = {}

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

    return index()
    #return render_template("chain_review.html", n_components=n_components,summary_tables=summary_tables,elements=elements,marginalized_flags=marginalized_flags )

'''
@app.route('/generate_latex', methods=['POST'])
def generate_latex():
    import os
    from flask import request, render_template

    flags = request.form.to_dict()
    
    # Load summary tables and elements list
    summary_tables = load_object('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/chain_review/summary_tables.pkl')
    save_dir = os.path.join("chain_upload", "user_upload")
    elements = load_object(os.path.join(save_dir, 'initial/initial_element_list.pkl'))
    n_components = len(summary_tables)

    # Start formatting LaTeX table
    table_options = (
        r"\centering" "\n"
        r"\resizebox{\textwidth}{!}{%" "\n"
        r"\renewcommand{\arraystretch}{1.3}" "\n"
        r"\tiny" "\n"
    )

    # Dynamically generate column format string and header names
    columns = "|c"  # for velocity
    column_names = "Velocity"
    for element in elements:
        columns += "|c|c"
        column_names += f" & {element} LogN & {element} b (km/s)"

    columns += "|"  # Close off the column format string

    # Construct each row of the table
    latex_rows = []
    for i in range(n_components):
        row = []

        # Velocity value
        velocity = {row['param']: row for row in summary_tables[i] if row['param'] == 'Velocity'}
        row.append(f"{velocity.get('Velocity', {}).get('median', '---')}")

        for element in elements:
            # User-defined flag
            flag = flags.get(f"flag_component_{i}_{element}")

            # Fetch LogN and b values
            params = {row['param']: row for row in summary_tables[i] if element in row['param']}
            logN = params.get(f"LogN {element}", {})
            b = params.get(f"b {element}", {})

            if flag == "saturated":
                row.append(f">$ {logN.get('p5', '---')}$")
                row.append(f"<$ {b.get('p95', '---')}$")
            elif flag == "not_detected":
                row.append(f"<$ {logN.get('p95', '---')}$")
                row.append("---")
            else:  # detected
                row.append(f"${logN.get('median', '---')}$")
                row.append(f"${b.get('median', '---')}$")

        # Add formatted LaTeX row
        latex_rows.append(" & ".join(row) + r" \\" + "\n" + r"\hline" + "\n")

    # Final LaTeX string
    latex_output = (
        r"\begin{table*}" "\n" +
        table_options +
        f"\\begin{{tabular}}{{{columns}}}\n" +
        r"\hline" "\n" +
        column_names + r" \\" + "\n" +
        r"\hline" "\n" +
        "".join(latex_rows) +
        r"\end{tabular}" "\n" +
        r"}" "\n" +
        r"\caption{Summary of absorption line properties for each component.}" "\n" +
        r"\label{tab:absorption_summary}" "\n" +
        r"\end{table*}"
    )

    # Create a unique filename
    filename = f"latex_table.tex"
    filepath = os.path.join("static", "latex_tables", filename)

    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save the file
    with open(filepath, "w") as f:
        f.write(latex_output)

    return render_template("latex_result.html", latex_output=latex_output)'''


@app.route('/latex_creation', methods=['POST'])
def latex_creation():
    from io import StringIO
    import re

    dataframes = []

    for i in range(1, 5):  # Adjust if you expect more files
        file = request.files.get(f"csv{i}")
        if file and file.filename != "":
            try:
                # Read file into pandas DataFrame
                content = file.read().decode("utf-8")
                df = pd.read_csv(StringIO(content))
                dataframes.append(df)
            except Exception as e:
                return f"Error reading csv{i}: {e}", 400
    
    los_names = []
    detections = []

    for df in dataframes:
        # Extract LOS name from 'Object Name' column
        full_name = str(df.get('Object Name', ['Unknown']).iloc[0])
        los_name = full_name.split()[-1] 
        df['LOS']=los_name
        los_names.append(los_name)

        # Detection if there are components
        has_components = "Component" in df.columns and df["Component"].notna().any()
        df['detection?']=has_components
        detections.append(has_components)

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


    '''def wrap_math_if_needed(val):
        if pd.isna(val):
            return ""
        
        val = str(val).strip()
        
        # Skip rows that are just '---' or empty
        if val.strip() == "---" or val == "":
            return ""
        
        # Only wrap if it looks like a number with latex or a limit
        if re.search(r'[<>]|\\pm', val):
            # Don't double wrap if already in math mode
            if val.startswith("$") and val.endswith("$"):
                return val
            return f"${val}$"
        
        return val

    # Apply to all cells
    df_for_latex = combined_df.copy().applymap(wrap_math_if_needed)'''

    #split the table into two now
    cols = df_for_latex.columns

    # ---- EW table ----
    ew_df_cols = [c for c in ['Object Name', 'Galaxy Z', 'Integration Window'] if c in cols]
    ew_df_cols += [c for c in cols if ('EW' in c and any(ch.isdigit() for ch in c)) or (c == 'Total LogN from EW')]
    # de-duplicate while preserving order
    ew_df_cols = list(dict.fromkeys(ew_df_cols))

    ew_df = df_for_latex[ew_df_cols]

    # ---- Component table ----
    comp_df_cols = [c for c in ['Object Name','Galaxy Z','Integration Window','Component','Velocity (km/s)'] if c in cols]
    comp_df_cols += [c for c in cols if ('LogN' in c and 'from EW' not in c) or ('b (km/s)' in c)]
    comp_df_cols = list(dict.fromkeys(comp_df_cols))

    comp_df = df_for_latex[comp_df_cols]

    ew_table_body=ew_df.to_latex(index=False, na_rep='', escape=False)
    comp_table_body=comp_df.to_latex(index=False, na_rep='', escape=False)

    tables=[ew_table_body,comp_table_body]

    # Then call .to_latex()
    #table_body = df_for_latex.to_latex(index=False, na_rep='', escape=False)

    #table_body=combined_df.to_latex(index=False,na_rep='',escape=False)

    for i,table_body in enumerate(tables):
        match = re.search(r'\\begin{tabular}{([^}]*)}', table_body)
        if match:
            original_format = match.group(1)
            num_columns = len(original_format)
            new_format = '|'.join(['c'] * num_columns)
            new_tabular_line = rf'\\begin{{tabular}}{{|{new_format}|}}'
            table_body = re.sub(r'\\begin{tabular}{[^}]*}', new_tabular_line, table_body)

        table_body = table_body.replace('+-', r'\pm').replace('+/-', r'\pm').replace('±',r'\pm')

        table_body = table_body.replace(r'\toprule', r'')
        table_body = table_body.replace(r'\midrule', r'')
        table_body = table_body.replace(r'\bottomrule', r'')

        table_body = re.sub(r'\\\\\n', r'\\\\\n\\hline\n', table_body)

        full_latex=r"""\begin{table*}
            \centering
            \resizebox{\textwidth}{!}{%
            \renewcommand{\arraystretch}{1.3}
            \tiny
            """ + table_body + r"""
            }
            \caption{placeholder}
            \label{tab:absorbers}
            \end{table*}
            """
        
        if i==0:
            with open(f"equivalent_width_latex.txt", "a") as f:
                f.write(full_latex)
        if i==1:
            with open(f"component_analysis_latex.txt", "a") as f:
                f.write(full_latex)

    return render_template("latex_result.html", latex_output=full_latex)


    


@app.route('/download_latex/<file_id>')
def download_latex():
    from flask import send_file
    filename = f"latex_table.tex"


    if os.path.exists(filename):
        return send_file(filename, as_attachment=True)
    else:
        return "File not found", 404




@app.route('/upload_data', methods=['POST'])
def upload_data():

    clean_house()

    files = request.files.getlist('data_files')
    
    if not files or files[0].filename == '':
        flash('No selected files')
        return redirect(request.url)

    upload_dir = '/Users/jakereinheimer/Desktop/Fakhri/data/custom/'
    saved_filepaths = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)
            saved_filepaths.append(filepath)
        else:
            flash(f'File type not allowed: {file.filename}')
            return redirect(request.url)
    
    nmf_requested = 'nmf' in request.form
        
    if nmf_requested:
        name='nmf'
        vp = VPFit(saved_filepaths,
                'custom',
                name)
        vp.DoAll()
    else:
        name=''
        vp = VPFit(saved_filepaths,
                'custom',
                name)
        vp.DoAll()
    
    filename = f'/Users/jakereinheimer/Desktop/Fakhri/VPFit/saved_objects/custom/{name}_vpfit.pkl'

    save_object(vp, filename) 
    save_object(vp,'current_vp.pkl')
    save_object({},'custom_absorptions.pkl')

    return redirect(url_for('show_results'))  # Or however you want to handle the next step



@app.route('/analyze', methods=['POST'])
def analyze():

    clean_house()

    selected_spectrum = request.form['spectrum']
    session['selected_spectrum'] = selected_spectrum
    selected_catalog = session.get('selected_catalog', None)  # Retrieve the selected catalog from session
    
    filename = f'/Users/jakereinheimer/Desktop/Fakhri/VPFit/saved_objects/{selected_catalog}/{selected_spectrum}_vpfit.pkl'

    # Check if the object already exists
    #if os.path.exists(filename):
    #    vp = load_object(filename)
    #    vp.vel_plots()
    #    vp.PlotFlux()

    #else:
    #    Initialize and process VPFit object
    vp = VPFit(f'/Users/jakereinheimer/Desktop/Fakhri/data/{selected_catalog}/{selected_spectrum}/',
                selected_catalog,
                selected_spectrum)
    vp.DoAll()

    save_object(vp, filename)  # Save the processed object
    save_object(vp, 'current_vp.pkl')
    save_object({},'custom_absorptions.pkl')

    return redirect(url_for('show_results'))

@app.route('/velocity_plots/<filename>')
def velocity_plots(filename):
    return url_for('Absorbers', filename=f'vel_plots/{filename}')

@app.route('/show_results', methods=['GET', 'POST'])
def show_results():
    absorbers = [load_object(os.path.join('Absorbers/objs/', item)) for item in os.listdir('Absorbers/objs/') if item.endswith('.pkl')]

    # Generate default MgII plots
    for absorber in absorbers:
        absorber.make_vel_plot('MgII')

    return render_template('results.html', absorbers=absorbers, custom_absorptions=custom_absorptions)


@app.route('/update_plot', methods=['POST'])
def update_plot():
    try:
        data = request.json
        absorber_index = int(data['absorber_index'])
        element = data['element']

        absorbers = [load_object(os.path.join('Absorbers/objs/', item)) for item in os.listdir('Absorbers/objs/') if item.endswith('.pkl')]

        if absorber_index >= len(absorbers):
            return jsonify({'error': 'Invalid absorber index'}), 400

        absorber = absorbers[absorber_index]
        plot_base64 = absorber.get_plot_base64(element)

        return jsonify({'plot_base64': plot_base64})

    except Exception as e:
        print(f"Error updating plot: {e}")
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/add_manual_absorption', methods=['POST'])
def add_manual_absorption():

    data = request.json
    xmin = float(data['xmin'])
    xmax = float(data['xmax'])
    element = data['element']
    transition = float(data['transition'])

    # Load the current VPFit object
    vp = load_object('current_vp.pkl')
    custom_absorptions=load_object('custom_absorptions.pkl')

    # Create a new absorption line object using VPFit's method
    absorption_line = vp.make_new_absorption(xmin, xmax, element, transition)

    absorption_line.plot()

    # Add to shared custom_absorptions dict (element + transition as key)
    custom_absorptions[absorption_line.name] = absorption_line

    # Prepare the list of absorptions to send back
    absorptions_summary = {}
    for k, v in custom_absorptions.items():
        absorptions_summary[k] = {
            'range': [v.wavelength[0], v.wavelength[-1]],
            'name' : v.name,
            'plot_base64': v.plot_base64
        }

    save_object(custom_absorptions,'custom_absorptions.pkl')

    return jsonify({'success': True, 'absorptions': absorptions_summary})

@app.route('/no_detection', methods=['POST'])
def no_detection():

    gal_z = float(request.form.get('manual_z'))

    vp = load_object('current_vp.pkl')

    line_dict=vp.no_detection(gal_z)

    #create csv and bail
    from astropy.cosmology import Planck18 as cosmo

    # Convert to velocity relative to galaxy
    vmin = -10
    vmax = 10
    #TODO COME BACK TO THIS


    # Create a list of rows, starting with metadata
    rows = [{
        "Object Name" : str(load_object('object_name.pkl')),
        "Galaxy Z": gal_z,
        "Integration Window": (vmin, vmax),
    }]

    #rows[0]['Projected Distance']=np.round(abs(cosmo.comoving_distance(ref_z).to('kpc')-cosmo.comoving_distance(our_2796.suspected_z).to('kpc')),2)
    #rows[0]['Absorber Redshift']=f"{our_2796.suspected_z:.5f}"
    keys_to_remove = []
    for key,value in line_dict.items():

        ew,ew_error=value.actual_ew_func()

        rows[0][f"{key.split(' ')[0]} {int(np.floor(float(key.split(' ')[1].strip())))} EW"]=f"{int(1000*ew)} +/- {int(1000*ew_error)} (<{int(3*1000*ew_error)})"

        #rows[0][f"{key} EW"]=f"{(1000*ew):.2f} +- {(1000*ew_error):.2f}"
        try:
            value.make_velocity(gal_z)
        except:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del line_dict[key]

    # Create a dataframe
    df = pd.DataFrame(rows)

    df.to_csv('absorber_data.csv',index=False)

    #__________________________________________________________________________________________
    #move all the stuff to new folder
    import shutil 

    object_folder=f"mcmc_outputs/{str(load_object('object_name.pkl'))}"
    os.makedirs(object_folder,exist_ok=True)

    #copy absorption csv file over too
    shutil.copy('absorber_data.csv',object_folder)

    #copy the line dict over to it too
    os.makedirs(os.path.join(object_folder,'final'),exist_ok=True)
    save_object(line_dict,os.path.join(object_folder,'final/line_dict.pkl'))

    return index()

@app.route('/multi_mcmc', methods=['POST'])
def multi_mcmc(no_pre=False, fixed=False):

    from random import random

    clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/final')
    clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/initial')


    #try:
    absorber_index = int(request.form['absorber_index'])
    element_list = request.form.getlist('multi_mcmc_elements')

    if absorber_index==100:
        absorber=load_object('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/custom_absorber/custom_absorber.pkl')
        lines=request.form.getlist('multi_mcmc_elements')
        absorber=absorber.return_specific_lines(lines)
        element_list_dict={}
        for line in lines:
            element_list_dict[line.split(' ')[0]]=' '

        element_list=list(element_list_dict.keys())
    
    elif absorber_index==101:
        absorber=load_object('custom_absorptions.pkl')
        element_list_dict={}
        for key,value in absorber.items():
            element_list_dict[key.split(' ')[0]]=' '

        element_list=list(element_list_dict.keys())

    else:

        absorbers = [load_object(os.path.join('Absorbers/objs/', item)) 
                        for item in os.listdir('Absorbers/objs/') if item.endswith('.pkl')]

        if absorber_index >= len(absorbers):
            return jsonify({'error': 'Invalid absorber index'}), 400

        absorber = absorbers[absorber_index]

    ref_z = request.form.get('ref_z')
    if ref_z is None:
        ref_z=absorber.z
    else:
        ref_z=float(ref_z)
    save_object(ref_z,'static/Data/multi_mcmc/initial/ref_z.pkl')

    initial_guesses,line_dict=pre_mcmc(absorber,element_list)

    #mcmc lines bounds
    mcmc_lines = load_object('static/Data/multi_mcmc/initial/initial_mcmc_lines.pkl')
    vmax_list = [line.vel_range[1] for line in mcmc_lines]
    vmin_list = [line.vel_range[0] for line in mcmc_lines]
    save_object(vmax_list,'static/Data/multi_mcmc/initial/vmax_list.pkl')
    save_object(vmin_list,'static/Data/multi_mcmc/initial/vmin_list.pkl')

    #statuses
    num_params_per_line = 1 + 2 * len(element_list)
    param_list_2d = np.array(initial_guesses).reshape(-1, num_params_per_line)
    
    display_params=param_list_2d.copy()
    display_params[:, 0] = np.round(param_list_2d[:, 0], 1)

    for i,e in enumerate(element_list):
        display_params[:,1+(i*2)] = np.round(display_params[:,1+(i*2)],2)
        display_params[:,2+(i*2)] = np.round(display_params[:,2+(i*2)],2)
    

    column_names=['Velocity']
    for e in element_list:
        column_names.append(f'LogN {e}')
        column_names.append(f'b {e}')

    num_rows=len(param_list_2d)
    num_cols=len(param_list_2d[0])
    statuses = []

    for i in range(num_rows):
        status_row = []
        for j in range(num_cols):
            status = 'free'
            status_row.append(status)
        statuses.append(status_row)

    #end statuses

    velocity_data = {line.name: line.to_dict() for line in line_dict.values()}

    velocity_data = {
        f"{line.name.split()[0]} {int(math.floor(float(line.name.split()[1])))}": line.to_dict()
        for line in line_dict.values()
    }



    real_masked_regions={}
    save_object(real_masked_regions,"static/Data/multi_mcmc/initial/masked_regions.pkl")

    save_object(velocity_data,'static/Data/multi_mcmc/initial/velocity_data.pkl')
    save_object(column_names,'static/Data/multi_mcmc/initial/column_names.pkl')
    save_object(param_list_2d,'static/Data/multi_mcmc/initial/initial_guesses.pkl')
    save_object(param_list_2d,'static/Data/multi_mcmc/initial/algo_guesses.pkl')



    return render_template('pre_mcmc.html',
                           parameters=display_params,
                           statuses=statuses,
                           line_dict=line_dict,
                           velocity_data=velocity_data,
                           column_names=column_names,
                           elements=element_list,
                           vmin_list=vmin_list,
                           vmax_list=vmax_list,
                           )

@app.route('/mcmc_param_update', methods=['POST'])
def mcmc_param_update():

    from random import random


    param_list_2d=np.array(load_object('static/Data/multi_mcmc/initial/initial_guesses.pkl'))
    elements=load_object('static/Data/multi_mcmc/initial/initial_element_list.pkl')

    num_params_per_line = 1 + 2 * len(elements)
    param_list_2d = param_list_2d.reshape(-1, num_params_per_line)
    print('param list')
    print(param_list_2d)
    num_rows=len(param_list_2d)
    num_cols=len(param_list_2d[0])

    #updating mcmc_line bounds
    mcmc_lines = load_object('static/Data/multi_mcmc/initial/initial_mcmc_lines.pkl')
    vmin_list=[]
    vmax_list=[]
    for i, line in enumerate(mcmc_lines):
        line.vel_range=(float(request.form.get(f'vmin_{i}', line.vel_range[0])),float(request.form.get(f'vmax_{i}', line.vel_range[1])))
        vmin_list.append(float(request.form.get(f'vmin_{i}', line.vel_range[0])))
        vmax_list.append(float(request.form.get(f'vmax_{i}', line.vel_range[1])))
    save_object(mcmc_lines,'static/Data/multi_mcmc/initial/initial_mcmc_lines.pkl')

    #statuses
    params = []
    statuses = []

    for i in range(num_rows):
        param_row = []
        status_row = []
        for j in range(num_cols):
            val = float(request.form[f'param_{i}_{j}'])
            status = request.form[f'status_{i}_{j}']

            param_row.append(val)
            status_row.append(status)
        params.append(param_row)
        statuses.append(status_row)


    element_list=load_object('static/Data/multi_mcmc/initial/initial_element_list.pkl')
    line_dict=load_object('static/Data/multi_mcmc/initial/initial_line_dict.pkl')
    column_names=load_object('static/Data/multi_mcmc/initial/column_names.pkl')

    velocity_data = {line.name: line.to_dict() for line in line_dict.values()}

    velocity_data = {
        f"{line.name.split()[0]} {int(math.floor(float(line.name.split()[1])))}": line.to_dict()
        for line in line_dict.values()
    }
    
    save_object(statuses,'static/Data/multi_mcmc/initial/initial_statuses.pkl')

    #masking
    real_masked_regions=load_object("static/Data/multi_mcmc/initial/masked_regions.pkl")

    masked_json = request.form.get("masked_regions")
    if masked_json:
        masked_regions = json.loads(masked_json)

        for key,value in masked_regions.items():

            new_name=floor_to_wave(key)
            bounds=(value.get('xmin'),value.get('xmax'))
            line_dict.get(new_name).add_masked_region(bounds)
            if real_masked_regions.get(new_name) is None:
                real_masked_regions[new_name]=[bounds]
            else:
                real_masked_regions.get(new_name).append(bounds)

            
        save_object(real_masked_regions,"static/Data/multi_mcmc/initial/masked_regions.pkl")
        save_object(line_dict,'static/Data/multi_mcmc/initial/initial_line_dict.pkl')
        print(real_masked_regions)

    lmfit_iterations = int(request.form["lmfit_iterations"])

    params=update_fit(params,element_list,lmfit_iterations)

    params = np.array(params).reshape(-1, num_params_per_line)

    params_np = np.array(params)

    print('status')
    print(statuses)

    display_params=params_np.copy()
    display_params[:, 0] = np.round(params_np[:, 0], 1)

    for i,e in enumerate(element_list):
        display_params[:,1+(i*2)] = np.round(display_params[:,1+(i*2)],2)
        display_params[:,2+(i*2)] = np.round(display_params[:,2+(i*2)],2)

    return render_template('pre_mcmc.html',
                           parameters=display_params,
                           statuses=statuses,
                           line_dict=line_dict,
                           velocity_data=velocity_data,
                           column_names=column_names,
                           elements=element_list,
                           vmin_list=vmin_list,
                           vmax_list=vmax_list,
                           random=random()
                           )


@app.route('/actual_mcmc', methods=['POST'])
def actual_mcmc():

    params = load_object('static/Data/multi_mcmc/initial/initial_guesses.pkl')
    elements=load_object('static/Data/multi_mcmc/initial/initial_element_list.pkl')
    num_params_per_line = 1 + 2 * len(elements)
    params = np.array(params).reshape(-1, num_params_per_line)
    statuses = load_object('static/Data/multi_mcmc/initial/initial_statuses.pkl')

    mcmc_steps = int(request.form.get('mcmc_steps', 1000))
    mcmc_walkers = int(request.form.get('mcmc_walkers', 250))

    mcmc(params,statuses,mcmc_steps,mcmc_walkers)

    return render_template('mcmc_results.html',
                            fit_plot_url = url_for('static', filename='Data/multi_mcmc/final/final_models.png'),
                            plot_url=url_for('static', filename='Data/multi_mcmc/final/mcmc_results.csv'),
                            trace_plot_url=url_for('static', filename='Data/multi_mcmc/final/mcmc_trace.png'),
                            corner_plot_url=url_for('static', filename='Data/multi_mcmc/final/mcmc_corner.png'),
                            mcmc_steps=mcmc_steps,
                            mcmc_walkers=mcmc_walkers)

@app.route('/manual_mcmc_lines', methods=['POST'])
def manual_mcmc_lines():

    vp = load_object('current_vp.pkl')
    mcmc_lines = load_object('static/Data/multi_mcmc/initial/initial_mcmc_lines.pkl')

    data = request.json
    xmin = data['xmin']
    xmax = data['xmax']

    #try:
    vp.smart_find(float(xmin), float(xmax))  # Process the selection
    return jsonify({'success': True})

@app.route('/add_component', methods=['POST'])
def add_component():
    from random import random

    # Load current guesses and statuses
    parameters = load_object('static/Data/multi_mcmc/initial/initial_guesses.pkl')
    statuses = load_object('static/Data/multi_mcmc/initial/initial_statuses.pkl')
    column_names = load_object('static/Data/multi_mcmc/initial/column_names.pkl')
    line_dict = load_object('static/Data/multi_mcmc/initial/initial_line_dict.pkl')
    mcmc_lines = load_object('static/Data/multi_mcmc/initial/initial_mcmc_lines.pkl')
    velocity_data=load_object('static/Data/multi_mcmc/initial/velocity_data.pkl')
    element_list=load_object('static/Data/multi_mcmc/initial/initial_element_list.pkl')
    vmin_list=load_object('static/Data/multi_mcmc/initial/vmin_list.pkl')
    vmax_list=load_object('static/Data/multi_mcmc/initial/vmax_list.pkl')

    print('parameters')
    print(parameters)
    print(len(parameters))

    print(element_list)


    num_params_per_line = 1 + 2 * len(element_list)
    params = np.array(parameters).reshape(-1, num_params_per_line)

    print('params')
    print(params)

    # Create a new free row with reasonable values (copy from last row or use defaults)
    new_row = list(params[-1])  # or [0.0]*len(parameters[0])
    new_status_row = ['free'] * len(column_names)

    print('new row')
    print(new_row)

    parameters.extend(new_row)
    params = np.array(parameters).reshape(-1, num_params_per_line)
    statuses.append(new_status_row)
    import copy
    mcmc_lines.append(copy.deepcopy(mcmc_lines[-1]))

    save_object(parameters, 'static/Data/multi_mcmc/initial/initial_guesses.pkl')
    save_object(statuses, 'static/Data/multi_mcmc/initial/initial_statuses.pkl')
    save_object(mcmc_lines,'static/Data/multi_mcmc/initial/initial_mcmc_lines.pkl')

    return render_template('pre_mcmc.html',
                           parameters=params,
                           statuses=statuses,
                           line_dict=line_dict,
                           velocity_data=velocity_data,
                           column_names=column_names,
                           elements=element_list,
                           vmin_list=vmin_list,
                           vmax_list=vmax_list,
                           random=random()
                           )

@app.route('/delete_component', methods=['POST'])
def delete_component():
    from random import random

    parameters = load_object('static/Data/multi_mcmc/initial/initial_guesses.pkl')
    statuses = load_object('static/Data/multi_mcmc/initial/initial_statuses.pkl')
    column_names = load_object('static/Data/multi_mcmc/initial/column_names.pkl')
    line_dict = load_object('static/Data/multi_mcmc/initial/initial_line_dict.pkl')
    mcmc_lines = load_object('static/Data/multi_mcmc/initial/initial_mcmc_lines.pkl')
    velocity_data=load_object('static/Data/multi_mcmc/initial/velocity_data.pkl')
    element_list=load_object('static/Data/multi_mcmc/initial/initial_element_list.pkl')
    vmin_list=load_object('static/Data/multi_mcmc/initial/vmin_list.pkl')
    vmax_list=load_object('static/Data/multi_mcmc/initial/vmax_list.pkl')

    num_params_per_line = 1 + 2 * len(element_list)
    parameters = np.array(parameters).reshape(-1, num_params_per_line)
    parameters=list(parameters)

    try:
        index_to_remove = int(request.form['component_index'])
    except (KeyError, ValueError):
        index_to_remove = -1

    if 0 <= index_to_remove < len(parameters) and len(parameters) > 1:
        parameters.pop(index_to_remove)
        statuses.pop(index_to_remove)
        mcmc_lines.pop(index_to_remove)
        vmin_list.pop(index_to_remove)
        vmax_list.pop(index_to_remove)

    save_object(parameters, 'static/Data/multi_mcmc/initial/initial_guesses.pkl')
    save_object(statuses, 'static/Data/multi_mcmc/initial/initial_statuses.pkl')
    save_object(mcmc_lines,'static/Data/multi_mcmc/initial/initial_mcmc_lines.pkl')
    save_object(vmin_list,'static/Data/multi_mcmc/initial/vmin_list.pkl')
    save_object(vmax_list,'static/Data/multi_mcmc/initial/vmax_list.pkl')

    #mcmc_param_update()

    return render_template('pre_mcmc.html',
                           parameters=parameters,
                           statuses=statuses,
                           line_dict=line_dict,
                           velocity_data=velocity_data,
                           column_names=column_names,
                           elements=element_list,
                           vmin_list=vmin_list,
                           vmax_list=vmax_list,
                           random=random()
                           )


@app.route('/chain_to_fixed_mcmc', methods=["POST"])
def chain_to_fixed_mcmc():
    
    clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/final')
    clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/initial')

    save_dir = os.path.join("chain_upload", "user_upload")

    chain=np.load(os.path.join(save_dir,"chain.npy"))
    mcmc_lines=load_object(os.path.join(save_dir,'final/mcmc_lines.pkl'))
    line_dict=load_object(os.path.join(save_dir,'final/line_dict.pkl'))
    params = load_object(os.path.join(save_dir,'final/initial_guesses.pkl'))
    element_list = load_object(os.path.join(save_dir,'initial/initial_element_list.pkl'))
    num_params_per_line = 1 + 2 * len(element_list)
    params = np.array(params).reshape(-1, num_params_per_line)
    statuses = np.array(load_object(os.path.join(save_dir,'initial/initial_statuses.pkl')))
    column_names=load_object(os.path.join(save_dir,'initial/column_names.pkl'))
    ref_z=load_object(os.path.join(save_dir,'initial/ref_z.pkl'))

    flattened = chain.reshape(-1, chain.shape[-1])  # shape: (10100*250, 10)
    median_params = np.median(flattened, axis=0)
    median_params_2d = np.reshape(median_params,(-1,num_params_per_line))
    print('median_params')
    print(median_params_2d)

    for i in range(len(statuses)):
        for j in range(len(statuses[i])):
            statuses[i][j]='fixed'

    velocity_data = {line.name: line.to_dict() for line in line_dict.values()}

    velocity_data = {
        f"{line.name.split()[0]} {int(math.floor(float(line.name.split()[1])))}": line.to_dict()
        for line in line_dict.values()
    }

    vmax_list = [line.vel_range[1] for line in mcmc_lines]
    vmin_list = [line.vel_range[0] for line in mcmc_lines]
    save_object(vmax_list,'static/Data/multi_mcmc/initial/vmax_list.pkl')
    save_object(vmin_list,'static/Data/multi_mcmc/initial/vmin_list.pkl')

    real_masked_regions={}
    save_object(real_masked_regions,"static/Data/multi_mcmc/initial/masked_regions.pkl")

    save_object(line_dict,'static/Data/multi_mcmc/initial/initial_line_dict.pkl')
    save_object(mcmc_lines,'static/Data/multi_mcmc/initial/initial_mcmc_lines.pkl')
    save_object(element_list,'static/Data/multi_mcmc/initial/initial_element_list.pkl')

    save_object(ref_z,'static/Data/multi_mcmc/initial/ref_z.pkl')
    save_object(velocity_data,'static/Data/multi_mcmc/initial/velocity_data.pkl')
    save_object(column_names,'static/Data/multi_mcmc/initial/column_names.pkl')
    save_object(median_params_2d,'static/Data/multi_mcmc/initial/initial_guesses.pkl')
    save_object(median_params_2d,'static/Data/multi_mcmc/initial/algo_guesses.pkl')

    plot_fits(median_params,line_dict,element_list,mcmc_lines,'initial/initial_guesses')

    column_names=['Velocity']
    for e in element_list:
        column_names.append(f'LogN {e}')
        column_names.append(f'b {e}')

    return render_template('pre_mcmc.html',
                        parameters=median_params_2d,
                        statuses=statuses,
                        line_dict=line_dict,
                        velocity_data=velocity_data,
                        column_names=column_names,
                        elements=element_list,
                        vmin_list=vmin_list,
                        vmax_list=vmax_list,
                        )


@app.route('/chain_to_mcmc', methods=['POST'])
def chain_to_mcmc():
    
    clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/final')
    clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/initial')

    save_dir = os.path.join("chain_upload", "user_upload")

    chain=np.load(os.path.join(save_dir,"chain.npy"))
    mcmc_lines=load_object(os.path.join(save_dir,'final/mcmc_lines.pkl'))
    line_dict=load_object(os.path.join(save_dir,'final/line_dict.pkl'))
    params = load_object(os.path.join(save_dir,'final/initial_guesses.pkl'))
    element_list = load_object(os.path.join(save_dir,'initial/initial_element_list.pkl'))
    num_params_per_line = 1 + 2 * len(element_list)
    params = np.array(params).reshape(-1, num_params_per_line)
    statuses = np.array(load_object(os.path.join(save_dir,'initial/initial_statuses.pkl')))
    column_names=load_object(os.path.join(save_dir,'initial/column_names.pkl'))
    ref_z=load_object(save_dir,'initial/ref_z.pkl')

    flattened = chain.reshape(-1, chain.shape[-1])  # shape: (10100*250, 10)
    median_params = np.median(flattened, axis=0)
    median_params_2d = np.reshape(median_params(-1,num_params_per_line))

    velocity_data = {line.name: line.to_dict() for line in line_dict.values()}

    velocity_data = {
        f"{line.name.split()[0]} {int(math.floor(float(line.name.split()[1])))}": line.to_dict()
        for line in line_dict.values()
    }

    vmax_list = [line.vel_range[1] for line in mcmc_lines]
    vmin_list = [line.vel_range[0] for line in mcmc_lines]
    save_object(vmax_list,'static/Data/multi_mcmc/initial/vmax_list.pkl')
    save_object(vmin_list,'static/Data/multi_mcmc/initial/vmin_list.pkl')

    real_masked_regions={}
    save_object(real_masked_regions,"static/Data/multi_mcmc/initial/masked_regions.pkl")

    save_object(line_dict,'static/Data/multi_mcmc/initial/initial_line_dict.pkl')
    save_object(mcmc_lines,'static/Data/multi_mcmc/initial/initial_mcmc_lines.pkl')
    save_object(element_list,'static/Data/multi_mcmc/initial/initial_element_list.pkl')

    save_object(ref_z,'static/Data/multi_mcmc/initial/ref_z.pkl')
    save_object(velocity_data,'static/Data/multi_mcmc/initial/velocity_data.pkl')
    save_object(column_names,'static/Data/multi_mcmc/initial/column_names.pkl')
    save_object(median_params_2d,'static/Data/multi_mcmc/initial/initial_guesses.pkl')
    save_object(median_params_2d,'static/Data/multi_mcmc/initial/algo_guesses.pkl')

    plot_fits(median_params,line_dict,element_list,mcmc_lines,'initial/initial_guesses')

    column_names=['Velocity']
    for e in element_list:
        column_names.append(f'LogN {e}')
        column_names.append(f'b {e}')

    return render_template('pre_mcmc.html',
                        parameters=median_params_2d,
                        statuses=statuses,
                        line_dict=line_dict,
                        velocity_data=velocity_data,
                        column_names=column_names,
                        elements=element_list,
                        vmin_list=vmin_list,
                        vmax_list=vmax_list,
                        )

    

@app.route('/data')
def data():

    vp = load_object(f'current_vp.pkl')

    wavelength = vp.wavelength
    flux = vp.flux
    error = vp.error

    return jsonify(wavelength=wavelength.tolist(), flux=flux.tolist(), error=error.tolist())



@app.route('/save_selection', methods=['POST'])
def save_selection():

    vp = load_object('current_vp.pkl')

    data = request.json
    xmin = data['xmin']
    xmax = data['xmax']

    #try:
    vp.smart_find(float(xmin), float(xmax))  # Process the selection
    return jsonify({'success': True})

@app.route('/manual_smart_find', methods=['POST'])
def manual_smart_find():
        
        vp = load_object('current_vp.pkl')

        z = float(request.form['manual_z'])
        vmin = float(request.form['manual_vmin'])
        vmax = float(request.form['manual_vmax'])

        #vp.smart_find_velocity(z,vmin,vmax)

        # Convert velocities to wavelength range for the smart find
        c_kms = 299792.458
        lambda_2796 = 2796.354269  # example rest wavelength of MgII

        lambda_min = lambda_2796 * (1 + z) * (1 + vmin / c_kms)
        lambda_max = lambda_2796 * (1 + z) * (1 + vmax / c_kms)


        vp.smart_find(lambda_min,lambda_max)

        custom_absorber=load_object('static/Data/custom_absorber/custom_absorber.pkl')

        custom_absorber_fluxplot=url_for('static', filename='Data/custom_absorber/FluxPlot.html')

        return render_template('custom_absorber_page.html',
                            absorber=custom_absorber,
                            fluxplot=custom_absorber_fluxplot
                                )


@app.route('/show_custom_results', methods=['GET'])
def show_custom_results():
    
    custom_absorber=load_object('static/Data/custom_absorber/custom_absorber.pkl')

    custom_absorber_fluxplot=url_for('static', filename='Data/custom_absorber/FluxPlot.html')

    return render_template('custom_absorber_page.html',
                           absorber=custom_absorber,
                           fluxplot=custom_absorber_fluxplot
                            )

@app.route('/continue_mcmc', methods=['POST'])
def continued_mcmc():
    display_params = load_object("static/Data/multi_mcmc/initial/initial_guesses.pkl")
    statuses = load_object("static/Data/multi_mcmc/initial/initial_statuses.pkl")
    line_dict = load_object("static/Data/multi_mcmc/initial/initial_line_dict.pkl")
    velocity_data=load_object('static/Data/multi_mcmc/initial/velocity_data.pkl')
    column_names = load_object('static/Data/multi_mcmc/initial/column_names.pkl')
    elements = load_object("static/Data/multi_mcmc/initial/initial_element_list.pkl")

    return render_template('pre_mcmc.html',
                           parameters=display_params,
                           statuses=statuses,
                           line_dict=line_dict,
                           velocity_data=velocity_data,
                           column_names=column_names,
                           elements=elements,
                           )



#Trident::::

@app.route('/trident', methods=['POST'])
def process_number():

    ID = request.form['ID']

    sim=Sim_spectra(ID)
    session['sim'] = base64.b64encode(pickle.dumps(sim)).decode('utf-8')
    #sim.plot('static/Trident/Trident_spectrum/spec_0.txt','static/Trident/Trident_spectrum/FluxPlot_0.html')
    #sim.plot('static/Trident/Trident_spectrum/spec_1.txt','static/Trident/Trident_spectrum/FluxPlot_1.html')

    x_plot_url = url_for('static', filename=f'Trident/Trident_plots/plot_x.png')
    y_plot_url = url_for('static', filename=f'Trident/Trident_plots/plot_y.png')
    z_plot_url = url_for('static', filename=f'Trident/Trident_plots/plot_z.png')

    plot_url = url_for('static', filename=f'Trident/Trident_plots/plot_ang_momentum.png')
    
    return render_template('trident.html',
                           x_plot_url = x_plot_url,
                           y_plot_url = y_plot_url,
                           z_plot_url = z_plot_url,
                           plot_url = plot_url,
                            )

@app.route('/trident_results1', methods=['GET','POST'])
def prepare_rays():
    
    if 'sim' in session:
        sim = pickle.loads(base64.b64decode(session['sim'].encode('utf-8')))
    else:
        return "Sim object not found in session", 400

    # Use request.form.get() to avoid KeyError if missing
    d_ray1 = request.form.get('d_ray1')
    theta_ray1 = request.form.get('theta_ray1')
    d_ray2 = request.form.get('d_ray2')
    theta_ray2 = request.form.get('theta_ray2')

    sim.do_Trident(float(d_ray1), float(theta_ray1), float(d_ray2), float(theta_ray2))

    sim.plot('static/Trident/Trident_spectrum/spec_0.txt','static/Trident/Trident_spectrum/FluxPlot_0.html')
    sim.plot('static/Trident/Trident_spectrum/spec_1.txt','static/Trident/Trident_spectrum/FluxPlot_1.html')

    plot_url = url_for('static', filename='Trident/Trident_plots/ray_plot.png')

    return render_template('trident_results1.html',
                        plot_url = plot_url,
                        )

@app.route('/trident_results_random', methods=['GET','POST'])
def random_rays():
    
    if 'sim' in session:
        sim = pickle.loads(base64.b64decode(session['sim'].encode('utf-8')))
    else:
        return "Sim object not found in session", 400

    sim.do_Trident('random',0,0,0)

    sim.plot('static/Trident/Trident_spectrum/spec_0.txt','static/Trident/Trident_spectrum/FluxPlot_0.html')
    sim.plot('static/Trident/Trident_spectrum/spec_1.txt','static/Trident/Trident_spectrum/FluxPlot_1.html')

    plot_url = url_for('static', filename='Trident/Trident_plots/ray_plot.png')

    return render_template('trident_results_random.html',
                        plot_url = plot_url,
                        )




@app.route('/trident_results', methods=['GET', 'POST'])
def trident_analysis():

    vp1 = VPFit(f'/Users/jakereinheimer/Desktop/Fakhri/VPFit/Trident_spectrum/spec_0.txt',
                   "TNG",
                   "blank")
    vp1.ApertureMethod()
    vp1.match_mgII()
    vp1.MgMatch()
    vp1.vel_plots(0)
    vp1.PlotFlux(0)

    vp2 = VPFit(f'/Users/jakereinheimer/Desktop/Fakhri/VPFit/Trident_spectrum/spec_1.txt',
                   "TNG",
                   "blank")
    vp2.ApertureMethod()
    vp2.match_mgII()
    vp2.MgMatch()
    vp2.vel_plots(1)
    vp2.PlotFlux(1)

    if request.method == 'POST':
        selected_doublet = request.form['doublet']
        session['selected_doublet'] = selected_doublet
    else:
        selected_doublet = session.get('selected_doublet', 'MgII')

    plot_url_1 = url_for('static', filename=f'Trident/velocity_plots/velocityPlot_{selected_doublet}_0')
    plot_url_2 = url_for('static', filename=f'Trident/velocity_plots/velocityPlot_{selected_doublet}_1')

    elements = list_velocity_plots()

    return render_template('results.html', 
                           plot_url_1 = plot_url_1,
                           plot_url_2 = plot_url_2,
                           elements=elements)



if __name__ == '__main__':

    from multiprocessing import Manager

    manager = Manager()
    custom_absorptions = manager.dict()

    app.run(debug=True, host='0.0.0.0', port=8080)
