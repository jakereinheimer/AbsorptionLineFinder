import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objs as go
import plotly.io as pio
import math
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter1d


matplotlib.use('Agg')

#my functions
from essential_functions import get_data,get_custom_data,read_atomDB,clear_directory,record_pair_csv,nmf
from AbsorptionLine import AbsorptionLine



class VPFit:
    
    def __init__(self,data_loc,catalog,name,custom=False):
        
        self.catalog=catalog
        self.object_name=name
        print(f"Catalog : {self.catalog}")

        if catalog=='custom':
            self.flux,self.error,self.wavelength,self.found_redshift=get_custom_data(data_loc)
            self.p=4
            self.N=3

            if self.object_name=='nmf':

                self.wavelength,self.flux,self.error = nmf(self.wavelength,self.flux,self.error)

        else:
            self.flux,self.error,self.wavelength,self.found_redshift=get_data(data_loc,catalog,name)

        if catalog=="Kodiaq":
            self.p=2
            self.N=4
        elif catalog == "TNG":
            self.p=4
            self.N=3
        elif catalog=="test":
            self.p=4
            self.N=3

        elif catalog=="SDSS7":
            self.p=8
            self.N=2

            self.plate=int(self.object_name.split("-")[0])
            self.fiber=int(self.object_name.split("-")[1])
            #self.fiber = f"{int(self.fiber):04d}"

            redshifts=pd.read_csv('/Users/jakereinheimer/Desktop/Fakhri/redshifts.csv')
            #print(redshifts.head())

            
            # Use boolean indexing to find rows
            result_df = redshifts[(redshifts['PLATE'] == self.plate) & (redshifts['FIBER'] == self.fiber)]

            # Extract 'ZABS' values as a list
            self.zhu_abs = result_df['ZABS'].tolist()
            self.zhu_abs_err = result_df['ERR_ZABS'].tolist()


        #correct sky lines
        for i,e in enumerate(self.error):
            if e>(np.mean(self.error)+(5*np.std(self.error))):
                self.flux[i]=1.0
                self.error[i]=np.mean(self.error)

        
        #clean data
        for i,f in enumerate(self.flux):
            if f<0:
                self.flux[i]=0

        for i,f in enumerate(self.flux):
            if f>3:
                self.flux[i]=3
        
        
        for i,e in enumerate(self.error):
            if e<0:
                self.error[i]=0

        for i,e in enumerate(self.error):
            if e>3:
                self.error[i]=3
        

        #get atom database
        self.atomDB=read_atomDB()
        
            
        self.lyman_alpha=1215.67*(1+self.found_redshift)
        
        self.no_forrest_flux=np.array(self.flux[self.wavelength>self.lyman_alpha])
        self.no_forrest_wavelength=np.array(self.wavelength[self.wavelength>self.lyman_alpha])
        self.no_forrest_error=np.array(self.error[self.wavelength>self.lyman_alpha])


        '''
        if smoothing:
            std=np.std(self.no_forrest_flux)
            for i,point in enumerate(self.no_forrest_flux):

                #catch first and last so no error
                if i==0 or i==len(self.no_forrest_flux)-1:
                    continue

                if self.no_forrest_flux[i] < .5 and self.no_forrest_flux[i-1] >= (1-std) and self.no_forrest_flux[i+1] >= (1-std):

                    self.no_forrest_flux[i] = (self.no_forrest_flux[i-1]+self.no_forrest_flux[i+1])/2
                '''


        
        print(f"Examining: {self.object_name}")
        print(f"Redshift is {self.found_redshift}")
        print(f"Making the end of the lyman alpha forrest as:{self.lyman_alpha}")



    def PlotFlux(self,iteration=0):
        
                # Create the plot with Plotly
        trace_flux = go.Scatter(
        x=self.wavelength, 
        y=self.flux, 
        mode='lines', 
        name='Flux',
        line=dict(width=1),
        line_shape='hv'  # This makes it a horizontal-vertical step plot
        )

        trace_error = go.Scatter(
        x=self.wavelength, 
        y=self.error, 
        mode='lines', 
        name='Error', 
        line=dict(color='red', width=1),
        line_shape='hv'  # This makes it a horizontal-vertical step plot
        )


            # Initialize the list of shapes (for vertical lines)
        shapes = [
            # End of Lyman alpha forest
            dict(type='line', x0=self.lyman_alpha, y0=min(self.flux), x1=self.lyman_alpha, y1=3, line=dict(color='Red', dash='dash')),
        ]

        if self.catalog=="SDSS7":
            lines=[2796.355099,2803.5322972,2600.1720322]
            for i,zabs in enumerate(self.zhu_abs):
                for j,line in enumerate(lines):
                    shapes.append(dict(type='line', x0=line*(1+zabs), y0=min(self.flux), x1=line*(1+zabs), y1=3, line=dict(color='purple', dash='dash')))
                    shapes.append(dict(type='line', x0=line*(1+(zabs+self.zhu_abs_err[i])), y0=min(self.flux), x1=line*(1+(zabs+self.zhu_abs_err[i])), y1=3, line=dict(color='cyan', dash='dash')))
                    shapes.append(dict(type='line', x0=line*(1+(zabs-self.zhu_abs_err[i])), y0=min(self.flux), x1=line*(1+(zabs-self.zhu_abs_err[i])), y1=3, line=dict(color='cyan', dash='dash')))
                    





        '''
        #show MgII lines
        c="blue"
        for i in self.zs.keys():
            pair=self.zs.get(i).get("MgII")

            if len(pair)==0:
                continue

            shapes.append(dict(type='line', x0=pair[0].peak, y0=min(self.flux), x1=pair[0].peak, y1=3, line=dict(color=c, dash='dash')))
            shapes.append(dict(type='line', x0=pair[1].peak, y0=min(self.flux), x1=pair[1].peak, y1=3, line=dict(color=c, dash='dash')))

        #show FeII lines
        c="orange"
        for i in self.zs.keys():
            pairs=self.zs.get(i).get("FeII")

            for pair in pairs:
                shapes.append(dict(type='line', x0=pair.peak, y0=min(self.flux), x1=pair.peak, y1=3, line=dict(color=c, dash='dash')))

        '''

        #Add vertical lines for all found absorptions
        #for i,absorption in enumerate(self.absorptions):
        #    shapes.append(dict(type='line', x0=absorption.peak, y0=min(self.flux), x1=absorption.peak, y1=3, line=dict(color='blue', dash='dash')))
        #    shapes.append(dict(type='line', x0=absorption.start, y0=min(self.flux), x1=absorption.start, y1=3, line=dict(color='orange', dash='dash')))
        #    shapes.append(dict(type='line', x0=absorption.stop, y0=min(self.flux), x1=absorption.stop, y1=3, line=dict(color='orange', dash='dash')))
        
        layout = go.Layout(
            title="Spectral Line Analysis",
            xaxis=dict(title='Wavelength'),
            yaxis=dict(title='Flux'),
            shapes=shapes
        )
        
        fig = go.Figure(data=[trace_flux, trace_error], layout=layout)
        
        # Save the interactive plot as an HTML file
        if self.catalog=="TNG":
            pio.write_html(fig, file=f'static/Trident/FluxPlot_{iteration}.html', auto_open=False)
        else:
            pio.write_html(fig, file='static/Data/FluxPlot.html', auto_open=False)



    def ApertureMethod(self, clean=True):
        """
        Implements the Aperture Method for detecting absorption features.
        
        Parameters:
        p: Number of pixels per resolution element.
        N_sigma: Significance threshold for detection.
        """
        
        # Define the aperture width as 2p + 1
        p=self.p
        N_sigma=self.N
        
        # Create arrays to store equivalent widths and their uncertainties
        equivalent_widths = np.zeros(len(self.flux))
        uncertainties = np.zeros(len(self.flux))
        
        # List to hold AbsorptionLine objects
        absorption_lines = []
        
        # Loop over each pixel in the spectrum
        for i in range(p, len(self.flux) - p):
            # Compute the equivalent width for the current aperture
            Dn = 1 - self.flux[i - p:i + p + 1]
            d_lambda_n = np.diff(self.wavelength[i - p:i + p + 2])
            
            # Calculate the equivalent width
            equivalent_widths[i] = -np.sum(Dn * d_lambda_n[0])

            # Get corresponding errors for the region
            flux_errors = self.error[i - p:i + p + 1]
            
            # Calculate the uncertainty for the equivalent width
            uncertainties[i] = np.sqrt(np.sum((flux_errors * d_lambda_n[0]) ** 2))
            if uncertainties[i] == 0:
                uncertainties[i] = 10**(-5)
        
        # Calculate the significance level
        try:
            significance = equivalent_widths / uncertainties
        except RuntimeWarning:
            significance = 0.
        
        # Find regions where the significance level exceeds the threshold
        detected_regions = significance <= -N_sigma

        i = 0
        while i < len(detected_regions):
            if detected_regions[i]:
                # Start of an absorption feature
                start_idx = i
                
                # Scan below (left) to find the start point where significance reaches ~1
                while start_idx > 0 and significance[start_idx] <= -1:
                    start_idx -= 1
                
                # Now scan above (right) to find the stop point where significance reaches ~1
                stop_idx = i
                while stop_idx < len(significance) - 1 and significance[stop_idx] <= -1:
                    stop_idx += 1
                
                # Keep extending if it dips again after the stop_idx
                
                extended = False
                while stop_idx < len(significance) - 1:
                    if significance[stop_idx] <= -N_sigma:  # Check for another dip after the initial stop
                        extended = True
                        stop_idx += 1
                    else:
                        if extended:
                            extended = False  # If it dipped and rose again, stop further extension
                            break
                        else:
                            break
                
                # Extract wavelength, flux, and error values for the extended region
                wave = self.wavelength[start_idx:stop_idx]
                flux = self.flux[start_idx:stop_idx]
                errors = self.error[start_idx:stop_idx]
                
                # Create an AbsorptionLine object and append to the list
                new_absorption_line = AbsorptionLine(wave, flux, errors)
                absorption_lines.append(new_absorption_line)
                
                # Move the index to just past the current stop index
                i = stop_idx
            else:
                i += 1
        
        # Print the number of absorption lines found
        print(f"Found {len(absorption_lines)} absorption features.")

        self.whole_absorptions=absorption_lines

        self.absorptions=[]
        for i in self.whole_absorptions:
            if i.peak>self.lyman_alpha:
                self.absorptions.append(i)

        if clean:
            # Clean the absorptions by removing those with zero errors or small equivalent widths
            for i in range(len(self.absorptions) - 1, -1, -1):  # Iterate backwards
                absorption = self.absorptions[i]

                # Get rid of places where error is zero
                if absorption.number_of_zeros_error>2:
                    self.absorptions.pop(i)

                # Get rid of thin absorptions
                elif absorption.EW < .1:
                    self.absorptions.pop(i)

        #for i in self.absorptions:
        #    i.plot(f"somethingat{i.peak}")

        #do the fitting for the surviving absorptions
        #for i in self.absorptions:
        #    i.get_center()
        
        # Return the absorption lines for further analysis
        return self.absorptions
    

    def optimizedMethod(self, p=3, N_sigma=3):

        pass


    def Nearest(self, wavelength, tol=1., whole=False):
        # Determine which list of absorption lines to use
        absorptions_list = self.whole_absorptions if whole else self.absorptions
        
        # Sort absorption lines by how close their peaks are to the input wavelength
        sorted_absorptions = sorted(absorptions_list, key=lambda line: abs(line.peak - wavelength))
        
        # Iterate over the sorted list and check for a close match within the given tolerance
        for line in sorted_absorptions:
            if math.isclose(wavelength, line.peak, abs_tol=tol):
                return line
        
        # If no exact match is found within the tolerance
        return False

    
    def isNear(self, wavelength):
        # Loop through all absorption lines to find an exact match
        for line in self.absorptions:
            if line.peak == wavelength:
                return line
        
        # If no exact match is found
        return False
    
    def calculate_peak_error(self, wavelength, resolution):
        """Calculate the error based on spectral resolution."""
        return wavelength / resolution


    def match_mgII(self):

        absorptions=self.ApertureMethod()

        filtered_rows = self.atomDB[self.atomDB['Transition'] == 'MgII']
        MgII_doublet=tuple(filtered_rows['Wavelength'])
        MgII_stregnths=tuple(filtered_rows['Strength'])
        EW_ratio=MgII_stregnths[0]/MgII_stregnths[1]

        # List to store the matched doublets or absorption groups
        potential_doublets = []

        rest_dist = abs(MgII_doublet[1] - MgII_doublet[0])
        maximum_dist=(1+self.found_redshift)*rest_dist

        for i, absorption in enumerate(absorptions):
            # Define the range of wavelengths you are interested in
            min_wavelength = absorption.peak + rest_dist
            max_wavelength = absorption.peak + maximum_dist

            for j, other_absorption in enumerate(absorptions):

                if other_absorption.peak>min_wavelength and other_absorption.peak<max_wavelength:
                    potential_doublets.append((absorption,other_absorption))

        #show potentials
        record_pair_csv(potential_doublets,"potential_doublets")

        # saturation check
        self.sat_check=[]
        for i, pair in enumerate(potential_doublets):

            if pair[0].is_saturated or pair[1].is_saturated:
                continue

            self.sat_check.append(pair)

        record_pair_csv(self.sat_check,"satuaraion_check")

            



        #z_check
        spectral_resolution = 5000  # This is just an example; adjust according to actual value

        self.z_check=[]
        for i, pair in enumerate(self.sat_check):

            #calc
            diff_obs = abs(pair[0].peak - pair[1].peak)
            diff_rest = abs(MgII_doublet[0] - MgII_doublet[1])

            pair[0].suspected_line_loc=MgII_doublet[0]
            pair[1].suspected_line_loc=MgII_doublet[1]

            #calc error for each peak
            peak_error1= pair[0].peak / spectral_resolution
            peak_error2= pair[1].peak / spectral_resolution

            suspected_z = (diff_obs / diff_rest) - 1
            suspected_loc1 = (suspected_z + 1) * MgII_doublet[0]
            suspected_loc2 = (suspected_z + 1) * MgII_doublet[1]

            propagated_error = np.sqrt((peak_error1 / diff_rest)**2 + (peak_error2 / diff_rest)**2)
            tolerance = propagated_error * MgII_doublet[0]  # Example of how to use the error to set tolerance

            if math.isclose(suspected_loc1, pair[0].peak, abs_tol=tolerance):
                suspected_z=(pair[0].peak-pair[0].suspected_line_loc)/pair[0].suspected_line_loc
                self.z_check.append((pair[0], pair[1], suspected_z))
            elif math.isclose(suspected_loc2, pair[1].peak, abs_tol=tolerance):
                suspected_z=(pair[0].peak-pair[0].suspected_line_loc)/pair[0].suspected_line_loc
                self.z_check.append((pair[0], pair[1], suspected_z))


        #for the survivors, calc z and then EW
        for i,pair in enumerate(self.z_check):

            pair[0].actual_equivalent_width()
            pair[1].actual_equivalent_width()

        record_pair_csv(self.z_check,"z_check",actual=True)


        #ew check
        self.ew_check=[]
        for i, pair in enumerate(self.z_check):
            if pair[0].actual_ew<pair[1].actual_ew:
                continue

            self.ew_check.append(pair)

        record_pair_csv(self.ew_check,"ew_check")


        #Column density check
        self.N_check=[]
        for i, pair in enumerate(self.ew_check):
            # Calculate N and sigma_N for both lines
            N1, sigma_N1 = pair[0].find_N(pair[2], MgII_stregnths[0], MgII_doublet[0])
            N2, sigma_N2 = pair[1].find_N(pair[2], MgII_stregnths[1], MgII_doublet[1])

            # Calculate the difference in N and the combined uncertainty
            #TODO why does this need factor of 10 to work?
            combined_sigma_N = 10*np.sqrt(sigma_N1**2 + sigma_N2**2)

            # Check if the difference in N is within the combined uncertainty
            if math.isclose(N1, N2, abs_tol=combined_sigma_N):
                self.N_check.append(pair)

        # Sort based on the ratio of the actual equivalent widths to a target EW ratio
        self.N_check.sort(key=lambda x: abs((x[0].actual_ew / x[1].actual_ew) - EW_ratio))

        # Record the pairs that pass the check
        record_pair_csv(self.N_check, "N_check", actual=True)


        #ML check?

        return self.N_check
        
        '''
        #fit gaussians 
        self.gaussian_check=[]
        zs=[]
        for i,pair in enumerate(self.EW_check):

            tested_wavelength = self.wavelength[np.where(np.logical_and(self.wavelength > (pair[0].start - 10), self.wavelength < (pair[1].stop + 10)))]
            tested_flux = self.flux[np.where(np.logical_and(self.wavelength > (pair[0].start - 10), self.wavelength < (pair[1].stop + 10)))]


            # Initial guess for the parameters: [amp1, cen1, sigma1, amp2, cen2, sigma2]
            #initial_guess = [-1, pair[0].peak, 1, -1, pair[1].peak, 1]

            initial_guess = [-.5, pair[0].peak, -.5, pair[1].peak]

            # Fit the double Gaussian model
            #popt, pcov = curve_fit(double_gaussian, tested_wavelength, tested_flux, p0=initial_guess, maxfev=10000000)
            popt, pcov = curve_fit(test_double_gaussian, tested_wavelength, tested_flux, p0=initial_guess, maxfev=10000000)

            #if pair[0].peak-1>popt[1]>pair[0].peak+1:
            #    continue

            #if pair[1].peak-1>popt[4]>pair[1].peak+1:
            #    continue

            z = (popt[3] - MgII_doublet[1]) / MgII_doublet[1]
            zs.append(z)

            EW1=2*np.sqrt(2*np.pi)*popt[0]*1
            #EW1=2*np.sqrt(2*np.pi)*popt[0]*popt[2]
            #EW2=2*np.sqrt(2*np.pi)*popt[3]*popt[5]
            EW2=2*np.sqrt(2*np.pi)*popt[3]*1

            if 0.9 <= EW1/EW2 <= 2.2:
                self.gaussian_check.append(pair)
            
            if pair[0].is_saturated and pair[1].is_saturated:
                #something about comparing their intensity 1:1
                if math.isclose(EW1/EW2,1,abs_tol=.3):
                    gaussian_check.append(pair)

            else:
                #intensity comparison but 2:1
                if math.isclose(EW1/EW2,2,abs_tol=.3):
                    gaussian_check.append(pair)
            

            
            
            fit_curve = test_double_gaussian(tested_wavelength, *popt)

            # Plot the original data and the fit
            plt.figure(figsize=(10, 5))
            plt.plot(tested_wavelength, tested_flux, 'b', label='Data')
            plt.plot(tested_wavelength, fit_curve, 'r--', label='Fit: Double Gaussian')
            plt.xlabel('Wavelength (Å)')
            plt.ylabel('Flux')
            plt.title(f'Double Gaussian Fit to MgII Lines, \n z:{z}')
            plt.legend()
            plt.savefig(f"gaussians/pairAt{pair[0].peak:.2f}.png")
            '''

        #record_pair_csv(self.gaussian_check,"Gaussian_check",actual=True)

        #FeII check
        '''
        #there should theoretically be FeII lines at 
        iron_confidence=[]
        self.iron_check=[]

        iron_doublets=self.doublets.get("FeII")

        for i,pair in enumerate(self.N_check):

            z=(pair[0].peak-2976)/2796

            check=[]
            for j,doub in enumerate(iron_doublets):

                suspected_loc=(z*doub)+doub

                suspected_line=self.Nearest(suspected_loc,tol=2.,whole=True)

                #suspected_EW=self.doublet_stregnth.get("MgII")[0]/self.doublet_stregnth.get("FeII")[j]

                #actual_EW=pair[0].EW/suspected_EW

                if isinstance(suspected_line,AbsorptionLine): #and math.isclose(actual_EW,suspected_EW,abs_tol=.5):
                    check.append(True)

            true_count = sum(check)
            print(check)

            if true_count>=3:
                self.iron_check.append(pair)
                iron_confidence.append(true_count)

        record_pair_csv(self.iron_check,"iron_check",actual=True)

        '''
        return self.iron_check

    
    def MgMatch(self):
        elements_to_check = ["FeII","MgI","AlII","CIV","SiII","SiIV","CII","OI"]
        spectral_resolution = 2800  # Example value, adjust as needed

        self.zs = {}
        for i, pair in enumerate(self.N_check):
            self.zs[pair[2]] = {"MgII": (pair[0], pair[1])}

        for z in self.zs.keys():
            current_dict = self.zs.get(z)

            for element in elements_to_check:
                doublet = tuple(self.atomDB[self.atomDB['Transition'] == element]['Wavelength'])
                strength = tuple(self.atomDB[self.atomDB['Transition'] == element]['Strength'])

                found_doublets = []
                N_check=[]
                for loc in doublet:
                    suspected_loc = (1 + z) * loc
                    error = self.calculate_peak_error(suspected_loc, spectral_resolution)
                    found_line = self.Nearest(suspected_loc, tol=5*error, whole=True)
                    if isinstance(found_line, AbsorptionLine):
                        #found_line.suspected_line_loc=loc
                        found_doublets.append(found_line)

                N_values = [line.find_N(z, strg, dblt) for line, strg, dblt in zip(found_doublets, strength, doublet)]
                ew = [line.actual_equivalent_width() for line in found_doublets]
                std = np.std(N_values)
                mean = np.mean(N_values)
                
                for j, line in enumerate(found_doublets):
                    #TODO consider changing tolerance
                    if math.isclose(line.N, mean, abs_tol=3*std):
                        #line.suspected_line_loc=doublet[j]
                        N_check.append(line)

                current_dict[element] = tuple(N_check)
                
                #current_dict[element] = tuple(found_doublets)
        print(self.zs)


    def make_blank_plot(self,name):

        fig, axs = plt.subplots(2, 1, figsize=(4, 6),sharex=True)

        plt.tight_layout(rect=[0.05, 0, 1, 1])  # Adjust to make room for the super y-label
        plt.savefig(f"static/velocityPlot_{name}.png")
        plt.clf()


    def vel_plots(self,iterations=0):

        clear_directory('/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Data/velocity_plots/')

        c = 3e5  # Speed of light in km/s
        velocity_window = 200  # km/s, adjust as necessary

        # Gather all elements and prepare figures based on redshifts and number of lines per redshift
        elements = {element for z in self.zs.values() for element in z if z[element]}
        for element in elements:
            # Prepare to track max number of lines at any redshift for this element
            max_lines_per_z = max(len(self.zs[z].get(element, [])) for z in self.zs)

            # Set up figure and axes grid
            fig, axs = plt.subplots(max_lines_per_z, len(self.zs), figsize=(20, 5 * max_lines_per_z), squeeze=False,sharex=True,sharey=True)
            
            z_sorted = sorted(self.zs.keys())  # Sort redshifts to maintain consistent order

            for col, z in enumerate(z_sorted):
                lines = self.zs[z].get(element, [])
                for row, line in enumerate(lines):
                    ax = axs[row, col]  # Select subplot
                    center_wavelength = line.peak
                    wavelength_window = center_wavelength * (1 + np.array([-velocity_window, velocity_window]) / c)
                    idx_start = np.searchsorted(self.wavelength, wavelength_window[0])
                    idx_stop = np.searchsorted(self.wavelength, wavelength_window[1])

                    full_velocity = (self.wavelength[idx_start:idx_stop] - center_wavelength) / center_wavelength * c
                    full_flux = self.flux[idx_start:idx_stop]
                    full_error = self.error[idx_start:idx_stop]

                    ax.step(full_velocity, full_flux, where='mid', label=f"Spectrum of {line.peak:.2f} Å", color="blue")
                    ax.step(full_velocity, full_error, where='mid', label="Error", color="purple", linestyle='--')
                    ax.axvline(0, color='red', linestyle='--', label="Center at 0 km/s")

                    #ew = line.actual_equivalent_width(z)
                    #N = line.find_N(z, line.strength, line.peak)
                    ax.set_title(f"line: {line.suspected_line_loc:.2f} Å at {line.peak:.2f} Å \nEW: {line.actual_ew:.2f} Å ± {line.actual_ew_error:.4f} Å \nlog(N): {line.log_N:.2f} cm⁻² ± {line.log_sigma_N:.2f} cm⁻²", fontsize=10)
                    #ax.legend()

                # Hide empty subplots if any
                for row in range(len(lines), max_lines_per_z):
                    fig.delaxes(axs[row, col])

            # Label axes and configure layout
            for ax in axs.flatten():
                ax.set_xlabel('Relative Velocity (km/s)', fontsize=12)
            for ax in axs[:, 0]:  # Set y-label for the first column
                ax.set_ylabel('Flux', fontsize=12)

            plt.suptitle(f"Velocity Profiles for {element}", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for title

            if self.catalog == "TNG":
                plt.savefig(f"static/Trident/velocity_plots/velocityPlot_{element}_{iterations}.png")
            else:
                plt.savefig(f"static/Data/velocity_plots/velocityPlot_{element}.png")
            plt.clf()

            print(f"Done with {element}")





    def DoAll(self):

        self.ApertureMethod()

        self.match_mgII()

        self.MgMatch()

        self.vel_plots()

        self.PlotFlux()

        j=0
        for outer_key, inner_dict in self.zs.items():
            
            for inner_key, tup in inner_dict.items():
                i=0
                for line in tup:

                    name=f"found_lines/{inner_key}/line_{j},{i}"

                    line.save(name)

                    i+=1

            j+=1
