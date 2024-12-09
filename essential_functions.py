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

def rebin(wave,old_vel,new_vel):
    from scipy.interpolate import interp1d

    new_wavelength_increment = (wave[-1] - wave[0]) * (new_vel / old_vel) / len(wave)
    new_wavelength_grid = np.arange(wave[0], wave[-1], new_wavelength_increment)


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

        #smoothing
        sigma = .5  # Adjust sigma based on your data and requirements
        smoothed_flux = gaussian_filter(flux_data, sigma=sigma)
        smoothed_errors = gaussian_filter(error_data, sigma=sigma)  # Smooth relative errors

        from scipy.interpolate import interp1d
        # Current velocity per pixel (given)
        v_pix = 4
        # Desired new velocity per pixel
        v_new = 60  # Change this to your desired velocity per pixel

        # Calculate new wavelength grid
        current_wavelength_grid = np.linspace(wavelength_data[0], wavelength_data[-1], len(wavelength_data))
        new_wavelength_increment = (wavelength_data[-1] - wavelength_data[0]) * (v_new / v_pix) / len(wavelength_data)
        new_wavelength_grid = np.arange(wavelength_data[0], wavelength_data[-1], new_wavelength_increment)

        # Interpolate the old data to the new wavelength grid
        flux_interpolator = interp1d(current_wavelength_grid, smoothed_flux, kind='linear', fill_value="extrapolate")
        error_interpolator = interp1d(current_wavelength_grid, smoothed_errors, kind='linear', fill_value="extrapolate")

        # New binned data
        flux_data = flux_interpolator(new_wavelength_grid)
        error_data = error_interpolator(new_wavelength_grid)
        wavelength_data=new_wavelength_grid

    else:
        return (None,None,None,None)

    return (flux_data,error_data,wavelength_data,found_redshift)



def read_atomDB():
        
        atom_loc='/Users/jakereinheimer/Desktop/Fakhri/VPFit/filtered_atom_db.dat'

        AtomDB = pd.read_csv(atom_loc, sep=',', engine='python', header=None, names=['Transition', 'Wavelength', 'Strength', 'Tau'])

        return AtomDB


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





