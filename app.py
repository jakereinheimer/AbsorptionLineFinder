from flask import Flask, render_template, request, url_for, redirect, session
from VPFit import VPFit
import pickle
import os

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
    directory = os.path.join(app.static_folder, 'velocity_plots')
    files = os.listdir(directory)
    png_files = [f for f in files if f.endswith('.png')]
    print(png_files)
    elements = set(file.split('_')[1] for file in png_files if '_' in file)  # Extract the element name part
    return elements



app = Flask(__name__)
app.secret_key = 'supersecretkey'

@app.route('/', methods=['GET'])
def index():
    selected_catalog = request.args.get('catalog', default=None)
    catalogs = list_catalogs()

    if selected_catalog is None or selected_catalog not in catalogs:
        selected_catalog = catalogs[0]

    session['selected_catalog'] = selected_catalog  # Store the selected catalog in session
    print(session['selected_catalog'])
    spectra_list = list_spectra(selected_catalog)
    return render_template('index.html', catalogs=catalogs, selected_catalog=selected_catalog, spectra=spectra_list)


@app.route('/analyze', methods=['POST'])
def analyze():
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
    plot_url = url_for('static', filename=f'velocity_plots/velocityPlot_{selected_doublet}')
    print("Generated plot URL:", plot_url)  # Debugging to ensure the URL is correct

    return render_template('results.html', plot_url=plot_url, selected_doublet=selected_doublet, elements=elements)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
