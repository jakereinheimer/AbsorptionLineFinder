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

from VPFit import VPFit
from TNG_trident import Sim_spectra
from essential_functions import clear_directory,get_data,floor_to_wave
from mcmc import run_mcmc,run_multi_mcmc,continue_mcmc,pre_mcmc,update_fit,mcmc
from AbsorptionLine import AbsorptionLine,AbsorptionLineSystem

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
    blue_paths = save_uploaded(blue_files)

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

    #except Exception as e:
    #    print(f"Error adding manual absorption: {e}")
    #    return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/multi_mcmc', methods=['POST'])
def multi_mcmc():

    from random import random

    clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/final')
    clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/initial')

    #try:
    absorber_index = int(request.form['absorber_index'])
    element_list = request.form.getlist('multi_mcmc_elements')

    if absorber_index==100:
        absorber=load_object('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/custom_absorber/custom_absorber.pkl')
    
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

    #statuses
    num_params_per_line = 1 + 2 * len(element_list)
    param_list_2d = np.array(initial_guesses).reshape(-1, num_params_per_line)

    display_params=param_list_2d.copy()
    base_velocity = param_list_2d[0, 0]
    display_params[:, 0] = np.round(param_list_2d[:, 0] - base_velocity, 1)

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
                           random=random()
                           )

@app.route('/mcmc_param_update', methods=['POST'])
def mcmc_param_update():

    from random import random


    param_list_2d=np.array(load_object('static/Data/multi_mcmc/initial/initial_guesses.pkl'))
    base_velocity = param_list_2d[0,0]
    num_rows=len(param_list_2d)
    num_cols=len(param_list_2d[0])

    #statuses
    params = []
    statuses = []

    for i in range(num_rows):
        param_row = []
        status_row = []
        for j in range(num_cols):
            val = float(request.form[f'param_{i}_{j}'])
            status = request.form[f'status_{i}_{j}']

            if j==0:
                val += base_velocity

            param_row.append(val)
            status_row.append(status)
        params.append(param_row)
        statuses.append(status_row)

    print('statuses')
    print(statuses)


    element_list=load_object('static/Data/multi_mcmc/initial/initial_element_list.pkl')
    line_dict=load_object('static/Data/multi_mcmc/initial/initial_line_dict.pkl')
    column_names=load_object('static/Data/multi_mcmc/initial/column_names.pkl')

    velocity_data=load_object('static/Data/multi_mcmc/initial/velocity_data.pkl')
    
    save_object(statuses,'static/Data/multi_mcmc/initial/initial_statuses.pkl')

    masked_json = request.form.get("masked_regions")
    if masked_json:
        masked_regions = json.loads(masked_json)

        real_masked_regions={}
        for key,value in masked_regions.items():
            new_name=floor_to_wave(key)
            real_masked_regions[new_name]=value

        print(real_masked_regions)
            
        save_object(real_masked_regions,"static/Data/multi_mcmc/initial/masked_regions.pkl")


    update_fit(params,element_list)

    params_np = np.array(params)

    display_params=params_np.copy()
    base_velocity = params_np[0, 0]
    display_params[:, 0] = np.round(params_np[:, 0] - base_velocity, 1)

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
                           random=random()
                           )


@app.route('/actual_mcmc', methods=['POST'])
def actual_mcmc():

    params = load_object('static/Data/multi_mcmc/initial/initial_guesses.pkl')
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

@app.route('/add_component', methods=['POST'])
def add_component():
    from random import random

    # Load current guesses and statuses
    parameters = load_object('static/Data/multi_mcmc/initial/initial_guesses.pkl')
    statuses = load_object('static/Data/multi_mcmc/initial/initial_statuses.pkl')
    column_names = load_object('static/Data/multi_mcmc/initial/column_names.pkl')
    line_dict = load_object('static/Data/multi_mcmc/initial/initial_line_dict.pkl')

    # Create a new free row with reasonable values (copy from last row or use defaults)
    new_row = list(parameters[-1])  # or [0.0]*len(parameters[0])
    new_status_row = ['free'] * len(column_names)

    parameters.append(new_row)
    statuses.append(new_status_row)

    save_object(parameters, 'static/Data/multi_mcmc/initial/initial_guesses.pkl')
    save_object(statuses, 'static/Data/multi_mcmc/initial/initial_statuses.pkl')

    return render_template('pre_mcmc.html',
                           parameters=parameters,
                           statuses=statuses,
                           line_dict=line_dict,
                           column_names=column_names,
                           random=random())

@app.route('/delete_component', methods=['POST'])
def delete_component():
    from random import random

    parameters = load_object('static/Data/multi_mcmc/initial/initial_guesses.pkl')
    statuses = load_object('static/Data/multi_mcmc/initial/initial_statuses.pkl')
    column_names = load_object('static/Data/multi_mcmc/initial/column_names.pkl')
    line_dict = load_object('static/Data/multi_mcmc/initial/initial_line_dict.pkl')

    try:
        index_to_remove = int(request.form['component_index'])
    except (KeyError, ValueError):
        index_to_remove = -1

    if 0 <= index_to_remove < len(parameters) and len(parameters) > 1:
        parameters.pop(index_to_remove)
        statuses.pop(index_to_remove)

    save_object(parameters, 'static/Data/multi_mcmc/initial/initial_guesses.pkl')
    save_object(statuses, 'static/Data/multi_mcmc/initial/initial_statuses.pkl')

    return render_template('pre_mcmc.html',
                           parameters=parameters,
                           statuses=statuses,
                           line_dict=line_dict,
                           column_names=column_names,
                           random=random())


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
    #except Exception as e:
    #    return jsonify({'success': False, 'error': str(e)})

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

'''
@app.route('/continue_mcmc', methods=['POST'])
def continued_mcmc():

    mcmc_steps = int(request.form.get('continued_multi_mcmc_steps', 1000))
    mcmc_walkers = int(request.form.get('continued_multi_mcmc_walkers', 250))

    map_params = load_object('static/Data/multi_mcmc/final/map_params.pkl')
    statuses = load_object('static/Data/multi_mcmc/initial/initial_statuses.pkl')
    elements=load_object('static/Data/multi_mcmc/initial/initial_element_list.pkl')

    params_per_line = 1 + 2 * len(elements)
    if map_params.ndim == 1:
        initial_guesses = map_params.reshape(-1, params_per_line)


    mcmc(initial_guesses,statuses,mcmc_steps,mcmc_walkers)

    #clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/final')

    # Run the continued MCMC
    #continue_mcmc(mcmc_steps,mcmc_walkers)

    return render_template('mcmc_results.html',
                            fit_plot_url = url_for('static', filename='Data/multi_mcmc/final/final_models.png'),
                            plot_url=url_for('static', filename='Data/multi_mcmc/final/mcmc_results.csv'),
                            trace_plot_url=url_for('static', filename='Data/multi_mcmc/final/mcmc_trace.png'),
                            corner_plot_url=url_for('static', filename='Data/multi_mcmc/final/mcmc_corner.png'),
                            mcmc_steps=mcmc_steps,
                            mcmc_walkers=mcmc_walkers)'''

    

#pop up tkinter window
from tkinter import Tk, Entry, Label, Button, StringVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
import matplotlib.pyplot as plt

def launch_tk(selected_catalog, selected_spectrum, shared_custom_absorptions):
    from tkinter import Tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    from matplotlib.widgets import SpanSelector

    print('Creating tkinter window')

    root = Tk()
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root.title("Select Absorption Range")

    # Load Spectrum Data
    vp = load_object('current_vp.pkl')
    wavelength = vp.wavelength
    flux = vp.flux
    error = vp.error

    plt.clf()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(wavelength, flux,where='mid', label="Flux", color="black",linewidth=.5)
    ax.step(wavelength,error,where='mid', label="Error", color="purple",linewidth=.5)
    ax.set_xlim(min(wavelength),max(wavelength))
    ax.set_ylim(0,2)
    ax.set_title('Select Absorption Line')

    selected_range = {'start': None, 'end': None}

    def onselect(xmin, xmax):
        selected_range['start'] = xmin
        selected_range['end'] = xmax
        print(f"Selected range: {xmin} - {xmax}")

    span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                        props=dict(alpha=0.5, facecolor='red'))

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Add toolbar for zoom/pan/save
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    toolbar.pack()

    element_var = StringVar()
    transition_var = StringVar()

    Label(root, text="Element:").pack()
    element_entry = Entry(root, textvariable=element_var)
    element_entry.pack()

    Label(root, text="Transition:").pack()
    transition_entry = Entry(root, textvariable=transition_var)
    transition_entry.pack()

    # Submit function
    def submit_absorption():
        element = element_var.get().strip()
        transition = float(transition_var.get().strip())
        start = selected_range['start']
        end = selected_range['end']

        if not element or not transition:
            print("Please enter both element and transition!")
            return

        if start is None or end is None:
            print("Please select a range first!")
            return

        # Create new absorption line
        vp = load_object('current_vp.pkl')
        shared_custom_absorptions[f"{element} {transition}"] = vp.make_new_absorption(start, end,element,transition)
        print(f"Added absorption: {element} {transition}, Range: {start:.2f} - {end:.2f}")

        print('custom absorptions')
        print(shared_custom_absorptions)

        print('wavelength test')
        print(shared_custom_absorptions.values()[0].wavelength)

        # Save a marker file to indicate update
        with open('update_flag.txt', 'w') as f:
            f.write('updated')

        root.quit()  # Close window after submission

    Button(root, text="Submit Absorption", command=submit_absorption).pack(pady=10)

    root.mainloop()


import multiprocessing

@app.route('/hand_select_absorption', methods=['POST'])
def hand_select_absorption():

    selected_catalog = session.get("selected_catalog")
    selected_spectrum = session.get("selected_spectrum")

    # Pass session variables to process
    p = multiprocessing.Process(target=launch_tk, args=(selected_catalog, selected_spectrum, custom_absorptions))
    p.start()

    return jsonify({'status': 'success', 'message': 'Tkinter window launched. Please select range locally.'})


@app.route('/custom_multi_mcmc',methods=['POST'])
def custom_multi_mcmc():

    mcmc_steps = int(request.form.get('custom_multi_mcmc_steps', 1000))
    mcmc_walkers = int(request.form.get('custom_multi_mcmc_walkers', 250))

    actual_custom_absorptions={}
    elements_dict={}
    for key,value in custom_absorptions.items():
        actual_custom_absorptions[key]=value
        elements_dict[key.split(' ')[0]]=value

    elements=elements_dict.keys()

    run_multi_mcmc(actual_custom_absorptions,elements,mcmc_steps,mcmc_walkers)

    return render_template('mcmc_results.html',
                            fit_plot_url = url_for('static', filename='Data/multi_mcmc/final/final_models.png'),
                            plot_url=url_for('static', filename='Data/multi_mcmc/final/mcmc_results.csv'),
                            trace_plot_url=url_for('static', filename='Data/multi_mcmc/final/mcmc_trace.png'),
                            corner_plot_url=url_for('static', filename='Data/multi_mcmc/final/mcmc_corner.png'),
                            mcmc_steps=mcmc_steps,
                            mcmc_walkers=mcmc_walkers)



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
