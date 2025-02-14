from flask import Flask, render_template, request, url_for, redirect, session,send_from_directory
from VPFit import VPFit
from TNG_trident import Sim_spectra
import pickle
import base64
import os
from werkzeug.utils import secure_filename
from flask import flash

from essential_functions import clear_directory
from mcmc import run_mcmc

#helper functions
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Open the file in binary write mode
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)  # Pickle the object and write to file

def load_object(filename):
    with open(filename, 'rb') as inp:  # Open the file in binary read mode
        return pickle.load(inp)  # Return the unpickled object

def list_catalogs():
    catalogs=os.listdir('/Users/jakereinheimer/Desktop/Fakhri/data')
    catalogs.remove(".DS_Store")
    return catalogs

def list_spectra(catalog):
    spectra_directory = os.path.join('/Users/jakereinheimer/Desktop/Fakhri/data', catalog)
    spectra = os.listdir(spectra_directory)
    return sorted(spectra)

def list_velocity_plots():
    directory = os.path.join(app.static_folder, 'Data/velocity_plots')
    files = os.listdir(directory)
    png_files = [f for f in files if f.endswith('.png')]
    print(png_files)
    elements = set(file.split('_')[1] for file in png_files if '_' in file)  # Extract the element name part
    return elements

# Add to the top where other imports are
ALLOWED_EXTENSIONS = {'txt', 'fits', 'csv','xspec'}  # Adjust based on the file types you expect

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def quantize_file(folder_loc):
    num=len(os.listdir(folder_loc))
    return num

def clean_house():
    try:
        clear_directory('static/Data/velocity_plots')
        clear_directory('static/Data/mcmc')
        os.remove('static/Data/FluxPlot.html')
        clear_directory('found_lines/')
    except:
        print('failed to delete everything')
        pass



app = Flask(__name__)
app.secret_key = 'supersecretkey'

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

def process_file(filepath):
    # Placeholder function to process the file
    pass  # Implement processing similar to your other routes



@app.route('/analyze', methods=['POST'])
def analyze():

    clean_house()

    selected_spectrum = request.form['spectrum']
    session['selected_spectrum'] = selected_spectrum
    selected_catalog = session.get('selected_catalog', None)  # Retrieve the selected catalog from session
    
    filename = f'/Users/jakereinheimer/Desktop/Fakhri/VPFit/saved_objects/{selected_catalog}/{selected_spectrum}_vpfit.pkl'

    # Check if the object already exists
    if os.path.exists(filename):
        vp = load_object(filename)
        vp.vel_plots()
        vp.PlotFlux()

    else:
        # Initialize and process VPFit object
        vp = VPFit(f'/Users/jakereinheimer/Desktop/Fakhri/data/{selected_catalog}/{selected_spectrum}/',
                   selected_catalog,
                   selected_spectrum)
        vp.DoAll()
        #save_object(vp, filename)  # Save the processed object

    return redirect(url_for('show_results'))

@app.route('/velocity_plots/<filename>')
def velocity_plots(filename):
    return url_for('static', filename=f'velocity_plots/{filename}')


@app.route('/show_results', methods=['GET', 'POST'])
def show_results():

    if request.method == 'POST':
        selected_doublet = request.form['doublet']
        session['selected_doublet'] = selected_doublet
    else:
        selected_doublet = session.get('selected_doublet', 'MgII')

    elements = list_velocity_plots()
    plot_url = url_for('static', filename=f'Data/velocity_plots/velocityPlot_{selected_doublet}')
    print("Generated plot URL:", plot_url)  # Debugging to ensure the URL is correct
    #num_of_pairs=os.listdir(quantize_file('found_lines/'+selected_doublet.split('.')[0]+'/')/2) /2
    num_of_pairs=int(quantize_file('found_lines/'+selected_doublet.split('.')[0]+'/')/2)

    # Create a list of indices for the number of buttons
    button_indices = list(range(num_of_pairs))

    return render_template('results.html', plot_url=plot_url, selected_doublet=selected_doublet, elements=elements, button_indices=button_indices)


@app.route('/mcmc', methods=['GET', 'POST'])
def mcmc_for_lines():
    if request.method == 'POST':
        line_index = int(request.form['line_index']) # Obtain the index of the line pair
        doublet=session['selected_doublet'].split('.')[0]

        mcmc_steps = int(request.form.get('mcmc_steps', 1000))
        mcmc_walkers= int(request.form.get('mcmc_walkers', 250))

        results = run_mcmc(doublet,line_index-1,nsteps=mcmc_steps,nwalkers=mcmc_walkers)

        plot_url = url_for('static', filename="Data/mcmc/mcmc_result.png")
        trace_plot_url = url_for('static', filename="Data/mcmc/mcmc_trace.png")
        corner_plot_url = url_for('static', filename="Data/mcmc/mcmc_corner.png")
        #chain something

        # Redirect or render a template to show the results
        return render_template('mcmc_results.html', results=results, line_index=line_index,plot_url=plot_url,trace_plot_url=trace_plot_url,corner_plot_url=corner_plot_url,mcmc_steps=mcmc_steps,mcmc_walkers=mcmc_walkers)
    else:
        # If not a POST request, redirect to a default page or handle accordingly
        return redirect(url_for('index'))





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
    app.run(debug=True, host='0.0.0.0', port=8080)
