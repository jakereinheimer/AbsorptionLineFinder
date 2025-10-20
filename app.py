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
from reviewer import chain_upload_,marginalize_component_,generate_csv_,latex_creation_,do_all_latex_

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


@app.route('/chain_upload',methods=['POST'])
def chain_upload():


    try:
        clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/chain_upload/user_upload')
        clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/chain_review/trace')
        clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/chain_review/triangle')
    except:
        pass

    uploaded_files = request.files.getlist("object_dir")

    object_name=request.form.get('object_name',' ')
    save_object(object_name,'object_name.pkl')

    class_flags=request.files.get('class_flags')

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

    n_components,all_summary_tables,detection_flags,elements=chain_upload_(save_dir,class_flags)

    return render_template(
    "chain_review.html",
    n_components=n_components,
    summary_tables=all_summary_tables,
    detection_flags=detection_flags,
    elements=elements,
    )


@app.route('/marginalize_component', methods=['POST'])
def marginalize_component():

    # --- Inputs / paths ---
    component_index   = int(request.form.get("component"))-1
    reference_element = request.form.get("reference_element","FeII")  # optional
    target_element    = request.form.get("target_element","MgII")     # optional
    percentile        = int(request.form.get("percentile",16)) #optional

    n_components,summary_tables,detection_flags,elements=marginalize_component_(component_index,reference_element,target_element,percentile)

    # --- Render page with updated state ---
    return render_template(
        "chain_review.html",
        n_components=len(summary_tables),
        summary_tables=summary_tables,
        detection_flags=detection_flags,
        elements=elements,
    )





@app.route('/generate_csv', methods=['POST'])
def generate_csv():

    import os
    from flask import request, render_template
    import pandas as pd
    import numpy as np
    import re

    flags = request.form.to_dict()

    generate_csv_(flags)

    return index()


@app.route('/latex_creation', methods=['POST'])
def latex_creation(inp=None):
    from io import StringIO
    import re

    if inp==None:
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
                
    else:
        dataframes=[]

        for file in inp:
            try:
                df = pd.read_csv(file)
                dataframes.append(df)
            except Exception as e:
                return f"Error reading csv{i}: {e}", 400
            
    full_latex=latex_creation_(dataframes)

    return render_template("latex_result.html", latex_output=full_latex)

@app.route('/do_all_latex', methods=['POST'])
def do_all_latex():

    clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/latex_tables')

    do_all_latex_()

    return index()

#TEMP
from flask import request, jsonify

@app.route('/mask_region', methods=['POST'])
def mask_region():
    payload = request.get_json(force=True)
    line_name = payload['line_name']
    vmin, vmax = payload['region']  # [vmin, vmax]

    # ensure ordering
    vmin, vmax = (vmin, vmax) if vmin <= vmax else (vmax, vmin)

    # store on your object; example:
    # absorption_lines[line_name].add_mask((vmin, vmax))
    # or however your project keeps state:
    # current_user_session.masks[line_name].append((vmin, vmax))

    return jsonify(ok=True, region=[vmin, vmax])

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

        rows[0][f"{key.split(' ')[0]} {int(np.floor(float(key.split(' ')[1].strip())))} EW (mÅ)"]=f"{int(1000*ew)} +/- {int(1000*ew_error)} (<{int(2*1000*ew_error)})"

        wave_cm = value.suspected_line*1e-8
        f=value.f
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

        #add it to df
        rows[0][f"{key.split(' ')[0]} {int(np.floor(float(key.split(' ')[1].strip())))} Optical Depth"]=f"<{logN:.2f}"

        try:
            value.make_velocity(gal_z)
        except:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del line_dict[key]

    # Create a dataframe
    df = pd.DataFrame(rows)

    df.to_csv(f"static/csv_outputs/absorber_data_{str(load_object('object_name.pkl'))}.csv",index=False)

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

        #jump in, we need to keep the whole absorber object
        ref_z = float(request.form.get('ref_z'))
        '''
        object_name=load_object("object_name.pkl")
        full_lines=absorber.return_full_dict()
        actual_dict={}
        for key,value in full_lines.items():
            if len(value.MgII_wavelength)>3:
                value.make_velocity(ref_z)
                actual_dict[key]=value

        save_object(actual_dict,f"{object_name}_full_linelist.pkl")
        #end jump in
        '''
        
        #for future:
        full_lines=absorber.return_full_dict()
        actual_dict={}
        for key,value in full_lines.items():
            if len(value.MgII_wavelength)>3:
                value.make_velocity(ref_z)
                actual_dict[key]=value

        save_object(actual_dict,f"static/Data/multi_mcmc/initial/full_line_dict.pkl")
        #end future
        
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
    ref_z=load_object(os.path.join(save_dir,'initial/ref_z.pkl'))

    flattened = chain.reshape(-1, chain.shape[-1])  # shape: (10100*250, 10)
    median_params = np.median(flattened, axis=0)
    median_params_2d = np.reshape(median_params,(-1,num_params_per_line))

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
