from astropy.io import fits
import os
import pandas as pd
import numpy as np
import shutil
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.ndimage

def run_nmf(wavelength,flux,error):

    mask = flux>0

    flux=flux[mask]
    wavelength=wavelength[mask]
    error=error[mask]

    flux_smoothed= median_filter(flux, size=100)

    flux_reshaped = flux_smoothed.reshape(1, -1)  # Reshape for NMF

    # Adjust NMF initialization and regularization
    nmf = NMF(n_components=12, random_state=0)

    # Fit NMF to the scaled flux data
    W = nmf.fit_transform(flux_reshaped)
    H = nmf.components_

    # Reconstruct the continuum from the components
    reconstructed_continuum = np.dot(W, H).flatten()

    flux_data=flux/reconstructed_continuum
    error_data=error/reconstructed_continuum

    return wavelength,flux_data,error_data


def get_data(folder,catalog,name):

    if catalog == "Kodiaq":

        with fits.open(folder+name+"_f.fits") as hdul:

            flux = hdul[0].data  # Change the index if your data is in a different extension

        with fits.open(folder+name+"_e.fits") as hdul:

            error = hdul[0].data 

        #wavelength
        redshifts= pd.read_csv('/Users/jakereinheimer/Desktop/Fakhri/VPFit/logging/Data_table.txt', delimiter=',',header=0,dtype=str)
            
        filtered_row = redshifts[redshifts['Object identifier'] == name]
        
        found_redshift= float(filtered_row['Emission line derived redshift'].values[0])
        
        wavelength_start = int(filtered_row['Lower wavelength of spectrum'].values[0])
        
        wavelength_end = int(filtered_row['Upper wavelength of spectrum'].values[0])
        
        wavelength_data = np.linspace(wavelength_start,wavelength_end,len(flux))

        #wavelength_data,flux_data,error_data=run_nmf(wavelength_data,flux,error)

        flux_data=flux
        error_data=error
    
    elif catalog=="SDSS7":
        with fits.open(folder + "data.fit") as hdul:
            data = hdul[0].data  # Data from the primary HDU
            flux = data[0]  # Original flux
            error = data[2]

            header = hdul[0].header
            # Retrieve the necessary header information to calculate the wavelength
            crval1 = header['CRVAL1']  # Starting wavelength
            cdelt1 = header['CD1_1']  # Wavelength increment per pixel
            npixels = header['NAXIS1'] 
            found_redshift = header['Z']

            wavelength_data = 10**(crval1 + cdelt1 * np.arange(npixels))

        wavelength_data,flux_data,error_data=run_nmf(wavelength_data,flux,error)



    elif catalog=="test":

        if name=='test3':

            file_path = f'/Users/jakereinheimer/Desktop/Fakhri/data/test/{name}/'  # Replace with the actual path to your FITS file

            #flux
            with fits.open(file_path+"flx.fits") as hdul:
                flux_data = np.array(hdul[0].data)

            #error
            with fits.open(file_path+"erf.fits") as hdul:
                error_data = np.array(hdul[0].data)

            #wave
            with fits.open(file_path+"wav.fits") as hdul:
                wavelength_data = np.array(hdul[0].data)

            found_redshift=2.1 #idk

            #wavelength_data,flux_data,error_data=run_nmf(wavelength_data,flux_data,error_data)

        else:

            file_path = f'/Users/jakereinheimer/Desktop/Fakhri/data/test/{name}/'  # Replace with the actual path to your FITS file

            #flux
            with fits.open(file_path+"test_f.fits") as hdul:
                flux_data = np.array(hdul[0].data)

            #error
            with fits.open(file_path+"test_e.fits") as hdul:
                error_data = np.array(hdul[0].data)

            #wave
            with fits.open(file_path+"test_wav.fits") as hdul:
                wavelength_data = np.array(hdul[0].data)

            found_redshift=2.1 #idk

            wavelength_data,flux_data,error_data=run_nmf(wavelength_data,flux_data,error_data)

    elif catalog=="DESI":

        pass

    elif catalog=="Secret_data":

        file_path = f'/Users/jakereinheimer/Desktop/Fakhri/data/{catalog}/{name}/'  # Replace with the actual path to your FITS file

        files=os.listdir(file_path)

        print(files)

        for file in files:

            print(file)

            if 'flx' in file:
                with fits.open(os.path.join(file_path,file)) as hdul:
                    flux_data = np.array(hdul[0].data)

            elif 'erf' in file:
                with fits.open(os.path.join(file_path,file)) as hdul:
                    error_data = np.array(hdul[0].data)

            elif 'wav' in file:
                with fits.open(os.path.join(file_path,file)) as hdul:
                    wavelength_data = np.array(hdul[0].data)

            else:
                raise "SOMETHING IS WRONG"

        found_redshift=2.1 #idk



    elif catalog == "TNG":
    
        data = pd.read_csv(folder, delimiter=" ", header=1, names=['wavelength', 'tau', 'flux', 'error'])

        wavelength=data['wavelength'].to_numpy()
        flux=data['flux'].to_numpy()
        error=data['error'].to_numpy()

        flux_smoothed= median_filter(flux, size=40)

        flux_reshaped = flux_smoothed.reshape(1, -1)  # Reshape for NMF

        # Adjust NMF initialization and regularization
        nmf = NMF(n_components=1, random_state=0)

        # Fit NMF to the scaled flux data
        W = nmf.fit_transform(flux_reshaped)
        H = nmf.components_

        # Reconstruct the continuum from the components
        reconstructed_continuum = np.dot(W, H).flatten()

        flux_data=flux/reconstructed_continuum
        error_data=error/reconstructed_continuum
        wavelength_data=wavelength

        found_redshift=2


    else:
        return (None,None,None,None)

    return (flux_data,error_data,wavelength_data,found_redshift)


import os
import pandas as pd
from astropy.io import fits

def get_custom_data(file_list, nmf=False):
    # Handle if loc is a directory

    print('file list')
    print(file_list)

    wavelength = flux = error = None

    if len(file_list)>1:

        for fname in file_list:
            if (fname.endswith('.fits') or fname.endswith('.fit')):
                if 'wav' in fname:
                    with fits.open(fname) as hdul:
                        wavelength = np.array(hdul[0].data)
                        print('wave')
                        print(wavelength)

                if 'flx' in fname:
                    with fits.open(fname) as hdul:
                        flux = np.array(hdul[0].data)

                if ('erf' in fname) or ('var' in fname):
                    with fits.open(fname) as hdul:
                        error = np.array(hdul[0].data)

                continue
    
    else:

        file=file_list[0]

        # Handle FITS files
        if file.endswith('.fits'):
            with fits.open(file) as hdul:
                data = hdul[1].data  # Assuming 1st extension contains the table
                wavelength = data['wavelength']
                flux = data['flux']
                error = data['error']

        # Handle XSPEC files (assuming it's ASCII with columns)
        elif file.endswith('.xspec'):
            df = pd.read_csv(file, delim_whitespace=True, comment='#')
            wavelength = df['wavelength'].to_numpy()
            flux = df['flux'].to_numpy()
            error = df['error'].to_numpy()

        # Handle CSV files (your original logic)
        elif file.endswith('.csv'):
            df = pd.read_csv(file)
            wavelength = df.get('Wavelength', df.get('wavelength')).to_numpy()
            flux = df.get('Flux', df.get('flux', df.get('NMF_flux'))).to_numpy()
            error = df.get('Error', df.get('error', df.get('NMF_error'))).to_numpy()

        else:
            raise ValueError(f"Unsupported file type: {file}")

    # Default redshift (maybe detect in the future)
    found_redshift = 2

    if wavelength is None:
        raise ValueError("Missing wavelength file")
    if flux is None:
        raise ValueError("Missing flux file")
    if error is None:
        raise ValueError("Missing error file")

    # Apply NMF if requested
    if nmf:
        wavelength, flux, error = run_nmf(wavelength, flux, error)

    return flux, error, wavelength, found_redshift


def gaussian_isf(resolution, width=5):
    from scipy.stats import norm

    """Generate a Gaussian ISF based on the resolution of the spectrograph."""
    sigma_isf = resolution / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
    x = np.linspace(-width * sigma_isf, width * sigma_isf, 2 * width + 1)
    isf = norm.pdf(x, scale=sigma_isf)
    isf /= isf.sum()  # Normalize the ISF
    return isf



def read_atomDB():
        
        #atom_loc='/Users/jakereinheimer/Desktop/Fakhri/VPFit/filtered_atom_db.dat'
        atom_loc='/Users/jakereinheimer/Desktop/Fakhri/VPFit/test_atom_db.dat'

        AtomDB = pd.read_csv(atom_loc, sep=',', engine='python', header=None, names=['Transition', 'Wavelength', 'Strength', 'Tau'])

        # Strip whitespace from Transition column
        AtomDB['Transition'] = AtomDB['Transition'].astype(str).str.strip()

        # Convert Wavelength to float (if not already)
        AtomDB['Wavelength'] = AtomDB['Wavelength'].astype(float)

        AtomDB['Floor'] = np.floor(AtomDB['Wavelength'].astype(float)).astype(int)

        return AtomDB

AtomDB=read_atomDB()

def floor_to_wave(name):
    floor = int(name.split(' ')[1].strip())
    transition = name.split(' ')[0].strip()

    filtered_row = AtomDB[(AtomDB['Floor'] == floor) & (AtomDB['Transition'] == transition)]

    if filtered_row.empty:
        return None  # or raise an error, depending on what you want

    # Get the first row's values
    new_name = f"{filtered_row.iloc[0]['Transition']} {filtered_row.iloc[0]['Wavelength']}"
    
    return new_name


def read_parameter(parameter_name):
    filename='parameters.txt'
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Check if the line contains the parameter name
                if parameter_name in line:
                    # Assume the format is 'name = value'
                    parts = line.split('=')
                    if len(parts) == 2:
                        # Return the value, stripping any whitespace and converting to float if possible
                        value = parts[1].strip()
                        try:
                            return float(value)
                        except ValueError:
                            return value
    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    print(f"Parameter '{parameter_name}' not found in the file.")
    return None



def clear_directory(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # List all files and directories in the folder
    items = os.listdir(folder_path)

    for item in items:
        item_path = os.path.join(folder_path, item)
        
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)  # Remove the file
                print(f"Deleted file: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove the directory and all its contents
                print(f"Deleted directory: {item_path}")
        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")


def record_pair_csv(pair_list, name, actual=False):
    peak1 = []
    peak2 = []
    EW1 = []
    EW2 = []
    zs=[]

    # Loop through each pair in the pair_list
    if actual==False:

        for pair in pair_list:
            peak1.append(pair[0].peak)
            peak2.append(pair[1].peak)
            #EW1.append(pair[0].EW)
            #EW2.append(pair[1].EW)

        # Create a DataFrame from the collected data
        df = pd.DataFrame({
            'peak1': peak1,
            'peak2': peak2,
            #'EW1': EW1,
            #'EW2': EW2
        })

    else:
        for pair in pair_list:
            peak1.append(pair[0].peak)
            peak2.append(pair[1].peak)
            #EW1.append(pair[0].actual_ew)
            #EW2.append(pair[1].actual_ew)
            zs.append(pair[2])

        # Create a DataFrame from the collected data
        df = pd.DataFrame({
            'peak1': peak1,
            'peak2': peak2,
            #'EW1': EW1,
            #'EW2': EW2,
            'z': zs
        })

    # Save the DataFrame to a CSV file
    df.to_csv(f"logging/{name}.csv", index=False)





