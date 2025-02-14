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


def get_data(folder,catalog,name):

    if catalog == "Kodiaq":

        flux_loc=folder+name+"_f.fits"
        error_loc=folder+name+"_e.fits"

        with fits.open(flux_loc) as hdul:

            flux_data = hdul[0].data  # Change the index if your data is in a different extension

        with fits.open(error_loc) as hdul:

            error_data = hdul[0].data 

        #wavelength
        redshifts= pd.read_csv('/Users/jakereinheimer/Desktop/Fakhri/VPFit/logging/Data_table.txt', delimiter=',',header=0,dtype=str)
            
        filtered_row = redshifts[redshifts['Object identifier'] == name]
        
        found_redshift= float(filtered_row['Emission line derived redshift'].values[0])
        
        wavelength_start = int(filtered_row['Lower wavelength of spectrum'].values[0])
        
        wavelength_end = int(filtered_row['Upper wavelength of spectrum'].values[0])
        
        wavelength_data = np.linspace(wavelength_start,wavelength_end,len(flux_data))
    
    elif catalog=="SDSS7":
        with fits.open(folder + "data.fit") as hdul:
            data = hdul[0].data  # Data from the primary HDU
            flux = data[0]  # Original flux
            flux_data=np.copy(flux)
            #flux_data = data[1]  # Continuum subtracted flux
            error_data = data[2]

            header = hdul[0].header
            # Retrieve the necessary header information to calculate the wavelength
            crval1 = header['CRVAL1']  # Starting wavelength
            cdelt1 = header['CD1_1']  # Wavelength increment per pixel
            npixels = header['NAXIS1'] 
            found_redshift = header['Z']

            wavelength_data = 10**(crval1 + cdelt1 * np.arange(npixels))

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
        error_data=error_data/reconstructed_continuum

        #NMF plot
        plt.plot(wavelength_data,flux,label="Original Flux",c="black",linewidth=.5)
        plt.plot(wavelength_data,reconstructed_continuum,label="NMF + Median",c="red")
        plt.xlabel("Wavelength")
        plt.ylabel("Flux")
        plt.legend()
        plt.savefig("NMF_test.png")

        #smoothing
        sigma = .5  # Adjust sigma based on your data and requirements
        smoothed_flux = gaussian_filter(flux_data, sigma=sigma)
        smoothed_errors = gaussian_filter(error_data, sigma=sigma)  # Smooth relative errors

        from scipy.interpolate import interp1d
        # Current velocity per pixel (given)
        v_pix = 69
        # Desired new velocity per pixel
        v_new = 60  # Change this to your desired velocity per pixel

        # Calculate new wavelength grid
        wavelength_start = 10**(crval1)
        wavelength_end = 10**(crval1 + cdelt1 * (npixels - 1))
        current_wavelength_grid = np.linspace(wavelength_start, wavelength_end, npixels)
        new_wavelength_increment = (wavelength_end - wavelength_start) * (v_new / v_pix) / npixels
        new_wavelength_grid = np.arange(wavelength_start, wavelength_end, new_wavelength_increment)

        # Interpolate the old data to the new wavelength grid
        flux_interpolator = interp1d(current_wavelength_grid, smoothed_flux, kind='linear', fill_value="extrapolate")
        error_interpolator = interp1d(current_wavelength_grid, smoothed_errors, kind='linear', fill_value="extrapolate")

        # New binned data
        flux_data = flux_interpolator(new_wavelength_grid)
        error_data = error_interpolator(new_wavelength_grid)
        wavelength_data=new_wavelength_grid



    elif catalog=="test":

        file_path = '/Users/jakereinheimer/Desktop/Fakhri/data/test/test1/'  # Replace with the actual path to your FITS file

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


def get_custom_data(loc):
    
    df=pd.read_csv(loc)

    try:
        wavelength=df['Wavelength'].to_numpy()
    except:
        wavelength=df['wavelength'].to_numpy()
    try:
        flux=df['Flux'].to_numpy()
    except:
        flux=df['flux'].to_numpy()
    try:
        error=df['Error'].to_numpy()
    except:
        error=df['error'].to_numpy()

    found_redshift=2

    return (flux,error,wavelength,found_redshift)

def nmf(inp_wavelength,inp_flux,inp_error):

    wavelength=inp_wavelength
    flux=inp_flux
    error=inp_error
    
    #Masking
    mask = flux>0

    flux=flux[mask]
    wavelength=wavelength[mask]
    error=error[mask]
    
    #median filter to smooth data        
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

    return wavelength,flux_data,error_data




def read_atomDB():
        
        atom_loc='/Users/jakereinheimer/Desktop/Fakhri/VPFit/filtered_atom_db.dat'

        AtomDB = pd.read_csv(atom_loc, sep=',', engine='python', header=None, names=['Transition', 'Wavelength', 'Strength', 'Tau'])

        return AtomDB

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

def double_gaussian(x, amp1, cen1, sigma1, amp2, cen2, sigma2):
    return (amp1 * np.exp(-(x - cen1)**2 / (2 * sigma1**2)) +
            amp2 * np.exp(-(x - cen2)**2 / (2 * sigma2**2))) +1

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
            EW1.append(pair[0].EW)
            EW2.append(pair[1].EW)

        # Create a DataFrame from the collected data
        df = pd.DataFrame({
            'peak1': peak1,
            'peak2': peak2,
            'EW1': EW1,
            'EW2': EW2
        })

    else:
        for pair in pair_list:
            peak1.append(pair[0].peak)
            peak2.append(pair[1].peak)
            EW1.append(pair[0].actual_ew)
            EW2.append(pair[1].actual_ew)
            zs.append(pair[2])

        # Create a DataFrame from the collected data
        df = pd.DataFrame({
            'peak1': peak1,
            'peak2': peak2,
            'EW1': EW1,
            'EW2': EW2,
            'z': zs
        })

    # Save the DataFrame to a CSV file
    df.to_csv(f"logging/{name}.csv", index=False)





