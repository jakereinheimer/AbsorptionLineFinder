from flask import Flask, render_template, request, url_for, redirect, session,send_from_directory,jsonify
import pickle
import base64
import os
from werkzeug.utils import secure_filename
from flask import flash
import numpy as np
import multiprocessing
from matplotlib.widgets import SpanSelector
import mpld3

from VPFit import VPFit
from TNG_trident import Sim_spectra
from essential_functions import clear_directory,get_data
from mcmc import run_mcmc,run_multi_mcmc,continue_mcmc
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

@app.route('/upload_data', methods=['POST'])
def upload_data():

    clean_house()

    file = request.files['data_file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('/Users/jakereinheimer/Desktop/Fakhri/data/custom/', filename)  # Specify your path for saving files
        file.save(filepath)

        nmf_requested = 'nmf' in request.form
        
        if nmf_requested:
            vp = VPFit(filepath,
                    'custom',
                    'nmf')
            vp.DoAll()
        else:
            vp = VPFit(filepath,
                    'custom',
                    '')
            vp.DoAll()

        

        return redirect(url_for('show_results'))  # Or however you want to handle the next step
    else:
        flash('File type not allowed')
        return redirect(request.url)



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

    return redirect(url_for('show_results'))

@app.route('/velocity_plots/<filename>')
def velocity_plots(filename):
    return url_for('Absorbers', filename=f'vel_plots/{filename}')

'''
@app.route('/show_results', methods=['GET', 'POST'])
def show_results():
    absorbers = [load_object(os.path.join('Absorbers/objs/', item)) for item in os.listdir('Absorbers/objs/') if item.endswith('.pkl')]

    for i,absorber in enumerate(absorbers):
        absorber.make_vel_plot('MgII',i)

    # Prepare default plot URLs for each absorber
    plot_urls = [url_for('static', filename=f'Data/velocity_plots/velocityPlot_MgII_{i}.png') for i in range(len(absorbers))]

    selected_absorber_index = None
    selected_element = None

    if request.method == 'POST':
        try:
            selected_absorber_index = int(request.form.get('absorber_index'))
            selected_element = request.form.get('element')

            if selected_absorber_index is not None and selected_element:
                absorber = absorbers[selected_absorber_index]
                absorber.make_vel_plot(selected_element)

                print(selected_absorber_index)
                print(selected_element)

                plot_urls[selected_absorber_index] = url_for('static', filename=f'Data/velocity_plots/velocityPlot_{selected_element}_{selected_absorber_index}.png')

        except (ValueError, IndexError) as e:
            print(f"Error processing form data: {e}")

    return render_template(
        'results.html',
        absorbers=absorbers,
        plot_urls=plot_urls,
        selected_element=selected_element,
        selected_absorber_index=selected_absorber_index
    )'''

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





'''
@app.route('/show_results', methods=['GET', 'POST'])
def show_results():

    if request.method == 'POST':
        selected_doublet = request.form['doublet']
        session['selected_doublet'] = selected_doublet
    else:
        selected_doublet = session.get('selected_doublet', 'MgII.png')

    plot_url = url_for('static', filename=f'Data/velocity_plots/velocityPlot_{selected_doublet}')
    print("Generated plot URL:", plot_url)  # Debugging to ensure the URL is correct
    #num_of_pairs=os.listdir(quantize_file('found_lines/'+selected_doublet.split('.')[0]+'/')/2) /2
    #num_of_pairs=quantize_file('found_lines/'+selected_doublet.split('.')[0]+'/')
    num_of_pairs=1

    # Create a list of indices for the number of buttons
    button_indices = list(range(num_of_pairs))

    absorbers = [load_object(os.path.join('Absorbers/objs/', item)) for item in os.listdir('Absorbers/objs/') if item.endswith('.pkl')]

    return render_template('results.html', plot_url=plot_url, selected_doublet=selected_doublet,absorbers=absorbers)'''

'''
@app.route('/mcmc', methods=['GET', 'POST'])
def mcmc_for_lines():
    if request.method == 'POST':
        line_index = int(request.form['line_index']) # Obtain the index of the line pair
        doublet=session['selected_doublet'].split('.')[0]

        mcmc_steps = int(request.form.get('mcmc_steps', 1000))
        mcmc_walkers= int(request.form.get('mcmc_walkers', 250))

        results = run_mcmc(doublet,line_index,nsteps=mcmc_steps,nwalkers=mcmc_walkers)

        plot_url = url_for('static', filename="Data/mcmc/mcmc_result.png")
        trace_plot_url = url_for('static', filename="Data/mcmc/mcmc_trace.png")
        corner_plot_url = url_for('static', filename="Data/mcmc/mcmc_corner.png")
        #chain something

        # Redirect or render a template to show the results
        return render_template('mcmc_results.html', results=results, line_index=line_index,plot_url=plot_url,trace_plot_url=trace_plot_url,corner_plot_url=corner_plot_url,mcmc_steps=mcmc_steps,mcmc_walkers=mcmc_walkers)
    else:
        # If not a POST request, redirect to a default page or handle accordingly
        return redirect(url_for('index'))
'''

@app.route('/mcmc', methods=['POST'])
def mcmc_for_lines():
    absorber_index = int(request.form['absorber_index'])
    element = request.form['mcmc_element']
    mcmc_steps = int(request.form.get('mcmc_steps', 1000))
    mcmc_walkers = int(request.form.get('mcmc_walkers', 250))

    absorbers = [load_object(os.path.join('Absorbers/objs/', item)) for item in os.listdir('Absorbers/objs/') if item.endswith('.pkl')]

    if absorber_index >= len(absorbers):
        return jsonify({'error': 'Invalid absorber index'}), 400

    absorber = absorbers[absorber_index]

    # Run MCMC for the selected element
    print('running mcmc')
    absorber.mcmc(element, nsteps=mcmc_steps, nwalkers=mcmc_walkers)
    print('Succeded')

    # Return results
    return render_template('mcmc_results.html',
                            plot_url=url_for('static', filename='Data/mcmc/mcmc_result.png'),
                            trace_plot_url=url_for('static', filename='Data/mcmc/mcmc_trace.png'),
                            corner_plot_url=url_for('static', filename='Data/mcmc/mcmc_corner.png'),
                            mcmc_steps=mcmc_steps,
                            mcmc_walkers=mcmc_walkers)


@app.route('/multi_mcmc', methods=['POST'])
def multi_mcmc():

    clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/final')
    clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/initial')

    #try:
    absorber_index = int(request.form['absorber_index'])
    element_list = request.form.getlist('multi_mcmc_elements')
    mcmc_steps = int(request.form.get('multi_mcmc_steps', 1000))
    mcmc_walkers = int(request.form.get('multi_mcmc_walkers', 250))

    if absorber_index==100:
        absorber=load_object('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/custom_absorber/custom_absorber.pkl')

    else:

        absorbers = [load_object(os.path.join('Absorbers/objs/', item)) 
                        for item in os.listdir('Absorbers/objs/') if item.endswith('.pkl')]

        if absorber_index >= len(absorbers):
            return jsonify({'error': 'Invalid absorber index'}), 400

        absorber = absorbers[absorber_index]

    # Run the multi_mcmc function with the selected elements
    absorber.multi_mcmc(element_list, nsteps=mcmc_steps, nwalkers=mcmc_walkers)

    print(f'Multi MCMC succeeded for elements: {element_list}')

    return render_template('mcmc_results.html',
                            fit_plot_url = url_for('static', filename='Data/multi_mcmc/final/final_models.png'),
                            plot_url=url_for('static', filename='Data/multi_mcmc/final/mcmc_results.csv'),
                            trace_plot_url=url_for('static', filename='Data/multi_mcmc/final/mcmc_trace.png'),
                            corner_plot_url=url_for('static', filename='Data/multi_mcmc/final/mcmc_corner.png'),
                            mcmc_steps=mcmc_steps,
                            mcmc_walkers=mcmc_walkers)
    #except Exception as e:
    #    print(f"Error running multi MCMC: {e}")
    #    return f"Error: {e}", 500

@app.route('/data')
def data():

    vp = load_object(f'/Users/jakereinheimer/Desktop/Fakhri/VPFit/saved_objects/{session.get("selected_catalog")}/{session.get("selected_spectrum")}_vpfit.pkl')

    wavelength = vp.wavelength
    flux = vp.flux
    error = vp.error

    return jsonify(wavelength=wavelength.tolist(), flux=flux.tolist(), error=error.tolist())



@app.route('/save_selection', methods=['POST'])
def save_selection():

    vp = load_object(f'/Users/jakereinheimer/Desktop/Fakhri/VPFit/saved_objects/{session.get("selected_catalog")}/{session.get("selected_spectrum")}_vpfit.pkl')

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
    vp = load_object(f'/Users/jakereinheimer/Desktop/Fakhri/VPFit/saved_objects/{selected_catalog}/{selected_spectrum}_vpfit.pkl')
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
        vp = load_object(f'/Users/jakereinheimer/Desktop/Fakhri/VPFit/saved_objects/{selected_catalog}/{selected_spectrum}_vpfit.pkl')
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


@app.route('/continue_mcmc', methods=['POST'])
def continued_mcmc():

    mcmc_steps = int(request.form.get('continued_multi_mcmc_steps', 1000))
    mcmc_walkers = int(request.form.get('continued_multi_mcmc_walkers', 250))

    #clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/multi_mcmc/final')

    # Run the continued MCMC
    continue_mcmc(mcmc_steps,mcmc_walkers)

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
