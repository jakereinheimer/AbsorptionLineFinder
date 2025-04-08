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
from essential_functions import get_data,get_custom_data,read_atomDB,clear_directory,record_pair_csv,gaussian_isf
from AbsorptionLine import AbsorptionLine, MicroAbsorptionLine, AbsorptionLineSystem,Absorber,Custom_absorption_line



class VPFit:
    
    def __init__(self,data_loc,catalog,name,custom=False):
        
        self.catalog=catalog
        self.object_name=name
        print(f"Catalog : {self.catalog}")
        print(f"Name : {name}")

        
        if catalog=='custom':
            self.flux,self.error,self.wavelength,self.emitter_redshift=get_custom_data(data_loc)
            self.p=4
            self.N=3

        else:
            self.flux,self.error,self.wavelength,self.emitter_redshift=get_data(data_loc,catalog,name)


        if catalog=="Kodiaq":
            self.p=2
            self.N=5
            self.fwhm=7.5
            self.tolerance_absorption_systems=30
        elif catalog == "TNG":
            self.p=4
            self.N=3
        elif catalog=="test":
            self.p=4
            self.N=3
            self.fwhm=7.5
            self.tolerance_absorption_systems=50

        elif catalog=="test2":
            self.p=4
            self.N=3
            self.fwhm=7.5
            self.tolerance_absorption_systems=50

        elif catalog=="SDSS7":
            self.p=8
            self.N=2
            self.fwhm=20

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
        
            
        self.lyman_alpha=1215.67*(1+self.emitter_redshift)
        
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
        print(f'z: {self.emitter_redshift}')


    def get_spectrum(self):

        return self.wavelength,self.flux,self.error
    
    def make_new_absorption(self,start,stop,element,transition):

        start_ind = np.argmin(np.abs(self.wavelength - start))
        stop_ind = np.argmin(np.abs(self.wavelength - stop))

        # Slice relevant part of the arrays
        sub_wavelength = self.wavelength[start_ind:stop_ind]
        sub_flux = self.flux[start_ind:stop_ind]
        sub_error = self.error[start_ind:stop_ind]

        print('sub wavelength')
        print(sub_wavelength)

        # Find min flux index within the selected region
        max_flux_ind = np.argmin(sub_flux)

        # Initialize left and right boundaries at max flux point
        left = max_flux_ind
        right = max_flux_ind

        # Scan left until you get 2 consecutive flux > 1
        while left > 1:
            if sub_flux[left - 1] > 1 and sub_flux[left - 2] > 1:
                break
            left -= 1

        # Scan right until you get 2 consecutive flux > 1
        while right < len(sub_flux) - 2:
            if sub_flux[right + 1] > 1 and sub_flux[right + 2] > 1:
                break
            right += 1

        # Get final indices relative to full spectrum
        final_start_ind = start_ind + left
        final_stop_ind = start_ind + right + 1  # +1 to include the last point

        # Slice the full wavelength, flux, error arrays
        final_wavelength = self.wavelength[final_start_ind:final_stop_ind]
        final_flux = self.flux[final_start_ind:final_stop_ind]
        final_error = self.error[final_start_ind:final_stop_ind]

        # Create the absorption line object
        absorption = Custom_absorption_line(final_wavelength, final_flux, final_error, element, transition)

        padding=300
        absorption.bonus(self.wavelength[final_start_ind-padding:final_stop_ind+padding],self.flux[final_start_ind-padding:final_stop_ind+padding],self.error[final_start_ind-padding:final_stop_ind+padding])

        return absorption


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
        #print(f"Found {len(absorption_lines)} absorption features.")

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
    

    def optimizedMethod(self, wavelength=None, flux=None, error=None, N=5):
            
            resolution=self.fwhm
            
            if wavelength is None:

                default_mode=True

                '''
                #mask = (self.wavelength > (1215.67 * (1 + self.emitter_redshift))) & (self.wavelength < 6800)

                #valid_indices = np.where(mask)[0]
                
                wavelength = self.wavelength[mask]
                flux = self.flux[mask]
                error = self.error[mask]'''

                wavelength=self.wavelength
                flux=self.flux
                error=self.error


            else:
                default_mode=False

            isf = gaussian_isf(resolution)
            num_pixels = len(flux)
            equivalent_widths = np.zeros(num_pixels)
            uncertainties = np.zeros(num_pixels)

            # Loop through each pixel in the spectrum
            for i in range(len(isf)//2, num_pixels - len(isf)//2):
                # Apply ISF weighting
                window_flux = flux[i - len(isf)//2 : i + len(isf)//2 + 1]
                window_wavelength = wavelength[i - len(isf)//2 : i + len(isf)//2 + 1]
                Dn = 1 - window_flux
                d_lambda = np.diff(window_wavelength)

                # Ensure Dn and d_lambda have the same length by taking Dn[:-1]
                if len(Dn) - 1 == len(d_lambda):
                    # Calculate weighted equivalent width
                    equivalent_widths[i] = -np.sum(isf[:-1]**2 * Dn[:-1] * d_lambda)

                    # Calculate uncertainty
                    window_error = error[i - len(isf)//2 : i + len(isf)//2 + 1]
                    uncertainties[i] = np.sqrt(np.sum((window_error[:-1] * isf[:-1] * d_lambda)**2))

            # Calculate significance
            significance = equivalent_widths / uncertainties
            detected = significance <= -N

            # Pad the detected array with False values to match the original spectrum length
            padding_length = len(self.wavelength) - len(detected)
            detected = np.pad(detected, (padding_length, 0), constant_values=False)

            #list of micro absorption lines
            micro_absorption_lines=[]
            start_index=None

            for i,result in enumerate(detected):

                if result:
                    if start_index == None:

                        start_index=i

                else:
                    if start_index is not None:

                        end_index=i

                        if (self.wavelength[i] > (1215.67 * (1 + self.emitter_redshift))):

                            micro_absorption_lines.append(MicroAbsorptionLine(self.wavelength[start_index:end_index],self.flux[start_index:end_index],self.error[start_index:end_index],start_index,end_index))

                        start_index=None
            
            #remove micro lines with error of zero
            #for i,microline in enumerate(self.micro_absorption_lines):
            #    print(microline.number_of_zeros_error)
            #    if microline.number_of_zeros_error<=3:
            #        self.micro_absorption_lines.pop(i)

            print(f"Found {len(micro_absorption_lines)} microlines")

            if default_mode==False:
                return micro_absorption_lines
            else:
                self.micro_absorption_lines=micro_absorption_lines


            #absorption line systems
            i = 0

            # List to hold groups of systems
            absorption_systems = []

            while i < len(self.micro_absorption_lines):
                current_system = [self.micro_absorption_lines[i]]
                current_end = self.micro_absorption_lines[i].global_end_ind

                # Look ahead and append lines within the tolerance range
                j = i + 1
                while j < len(self.micro_absorption_lines) and self.micro_absorption_lines[j].global_start_ind <= current_end + self.tolerance_absorption_systems:
                    current_system.append(self.micro_absorption_lines[j])
                    current_end = max(current_end, self.micro_absorption_lines[j].global_end_ind)
                    j += 1

                # Store the current system
                absorption_systems.append(current_system)

                # Move to the next unprocessed line
                i = j

            for system in absorption_systems:
                # Adjust global_start_ind by scanning left until flux > 1
                start_ind = system[0].global_start_ind
                while start_ind > 0 and self.flux[start_ind] <= 1.0:
                    start_ind -= 1
                system[0].global_start_ind = start_ind

                # Adjust global_end_ind by scanning right until flux > 1
                end_ind = system[-1].global_end_ind
                while end_ind < len(self.flux) - 1 and self.flux[end_ind] <= 1.0:
                    end_ind += 2
                system[-1].global_end_ind = end_ind


            absorption_systems_obj=[]

            for i,system in enumerate(absorption_systems):

                abs_object=AbsorptionLineSystem(self,system)

                if abs_object.equivalent_width() > 0.1 and abs_object.number_of_zeros_error < 3:
                    #if abs_object.equivalent_width() > 0.1:

                    absorption_systems_obj.append(abs_object)

            if default_mode:

                self.absorptions=absorption_systems_obj

            else:
                return absorption_systems_obj

            '''
            #plot to check
            plt.figure(figsize=(14, 8))
            plt.plot(self.wavelength, self.flux, label='Flux', color='black')

            # Highlight each absorption system
            for system in absorption_systems:
                start_wavelength = self.wavelength[system[0].global_start_ind]
                end_wavelength = self.wavelength[system[-1].global_end_ind]

                plt.axvspan(start_wavelength, end_wavelength, color='red', alpha=0.3, label='Absorption System' if system == absorption_systems[0] else "")

            # Plot details
            plt.xlabel('Wavelength')
            plt.ylabel('Flux')
            plt.title('Detected Absorption Systems')
            plt.legend()
            plt.ylim((0,2))
            #test
            plt.xlim((4250,4300))
            #kodiaq
            #plt.xlim((4400,4500))
            plt.grid(True)
            plt.savefig('Optimized_test.png')
            '''


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
    
    def MgII_search(self):

        filtered_rows = self.atomDB[self.atomDB['Transition'] == 'MgII']
        MgII_doublet=tuple(filtered_rows['Wavelength'])
        MgII_stregnths=tuple(filtered_rows['Strength'])
        EW_ratio=MgII_stregnths[0]/MgII_stregnths[1]

        potential_doublets = []

        rest_dist = abs(MgII_doublet[1] - MgII_doublet[0])
        maximum_dist=(1+self.emitter_redshift)*rest_dist

        for i, absorption in enumerate(self.absorptions):
            # Define the range of wavelengths you are interested in
            min_wavelength = absorption.peak + rest_dist
            max_wavelength = absorption.peak + maximum_dist

            for j, other_absorption in enumerate(self.absorptions):

                if other_absorption.peak>min_wavelength and other_absorption.peak<max_wavelength:
                    potential_doublets.append((absorption,other_absorption))

        #show potentials
        record_pair_csv(potential_doublets,"potential_doublets")

        #new z check
        z_check=[]

        peak_error = self.fwhm / 2.355

        for pair in potential_doublets:

            z_1=(pair[0].peak-MgII_doublet[0])/MgII_doublet[0]
            z_2=(pair[1].peak-MgII_doublet[1])/MgII_doublet[1]

            if math.isclose(z_1,z_2,abs_tol=.05):

                final_suspected_z = (z_1+z_2)/2

                # Propagate errors to estimate z error
                z_error = np.sqrt((peak_error / MgII_doublet[0])**2 + (peak_error / MgII_doublet[1])**2) * final_suspected_z

                z_check.append((pair[0], pair[1], final_suspected_z))

                pair[0].suspected_line=MgII_doublet[0]
                pair[0].f=MgII_stregnths[0]
                pair[0].z=final_suspected_z
                pair[0].z_err=z_error
                pair[1].suspected_line=MgII_doublet[1]
                pair[1].f=MgII_stregnths[1]
                pair[1].z=final_suspected_z
                pair[1].z_err=z_error


        '''
        #z check
        z_check=[]

        for pair in potential_doublets:

            peak1,peak2 = pair[0].peak,pair[1].peak

            suspected_z = (abs(peak1-peak2)/abs(MgII_doublet[0] - MgII_doublet[1])) - 1

            # Calculate suspected peak locations based on the redshift
            suspected_loc1 = (1 + suspected_z) * MgII_doublet[0]
            suspected_loc2 = (1 + suspected_z) * MgII_doublet[1]

            
            

            peak_error = self.fwhm / 2.355
            
            final_suspected_z = (((peak1 - MgII_doublet[0]) / MgII_doublet[0])+((peak1 - MgII_doublet[1]) / MgII_doublet[1]))/2

            z_error = np.sqrt((peak_error / MgII_doublet[0])**2 + (peak_error / MgII_doublet[1])**2) * final_suspected_z

            tolerance = 5 * z_error * MgII_doublet[0]

            # Propagate the error for tolerance calculation
            #propagated_error = np.sqrt((peak_error / abs(MgII_doublet[0] - MgII_doublet[1]))**2 + (peak_error / abs(MgII_doublet[0] - MgII_doublet[1]))**2)
            #tolerance = propagated_error * MgII_doublet[0]
            tolerance = 1.1*100

            print('tolerance')
            print(tolerance)

            if all([
                math.isclose(suspected_loc1, peak1, abs_tol=tolerance),
                math.isclose(suspected_loc2, peak2, abs_tol=tolerance)
            ]):
                final_suspected_z = (((peak1 - MgII_doublet[0]) / MgII_doublet[0])+((peak1 - MgII_doublet[1]) / MgII_doublet[1]))/2
                # Propagate errors to estimate z error
                z_error = np.sqrt((peak_error / MgII_doublet[0])**2 + (peak_error / MgII_doublet[1])**2) * final_suspected_z
                z_check.append((pair[0], pair[1], final_suspected_z))

                pair[0].suspected_line=MgII_doublet[0]
                pair[0].f=MgII_stregnths[0]
                pair[0].z=final_suspected_z
                pair[0].z_err=z_error
                pair[1].suspected_line=MgII_doublet[1]
                pair[1].f=MgII_stregnths[1]
                pair[1].z=final_suspected_z
                pair[1].z_err=z_error'''

        record_pair_csv(z_check,"z_check",actual=True)

        #width check
        width_check=[]
        for i, pair in enumerate(z_check):
            if math.isclose(len(pair[0].wavelength),len(pair[1].wavelength),abs_tol=10):

                width_check.append(pair)

        record_pair_csv(width_check,"width_check")

        #go through the surviving pairs and calc EW and N

        for pair in width_check:

            pair[0].find_N()
            pair[0].equivalent_width()

            pair[1].find_N()
            pair[1].equivalent_width()

        

        
        #ew check
        ew_check=[]
        for i, pair in enumerate(width_check):
            if (1.1<=pair[0].ew/pair[1].ew<=2):
                
                ew_check.append(pair)

        record_pair_csv(ew_check,"ew_check")
        

        #N check
        N_check=[]
        sig=10
        for i,pair in enumerate(ew_check):

            if ((pair[1].N-(pair[1].sigma_N_minus*sig))<=pair[0].N<=(pair[1].N+(sig*pair[1].sigma_N_plus))):

                N_check.append(pair)


        # Sort based on the ratio of the actual equivalent widths to a target EW ratio
        N_check.sort(key=lambda x: abs((x[0].ew / x[1].ew) - EW_ratio))

        # Record the pairs that pass the check
        record_pair_csv(N_check, "N_check", actual=True)


        #fill absorbers list
        self.absorbers=[]

        for i,pair in enumerate(N_check):

            self.absorbers.append(Absorber(pair[0].z,pair[0].z_err,pair))

        
    def MgMatch(self):

        lines_to_check = self.atomDB["Wavelength"].to_numpy()
        elements_to_check = self.atomDB['Transition'].to_numpy()

        for i,absorber in enumerate(self.absorbers):

            z=absorber.z

            MgII_1=absorber.lines.get('MgII 2796.355099')
            MgII_2=absorber.lines.get('MgII 2803.5322972')

            MgII_1.name='MgII 2796.355099'
            MgII_1.z=z

            MgII_2.name='MgII 2803.5322972'
            MgII_2.z=z

            z_low=(MgII_1.wavelength[0]-2796.355099)/2796.355099
            z_high=(MgII_1.wavelength[-1]-2796.355099)/2796.355099


            if len(MgII_1.wavelength) >= len(MgII_2.wavelength):
                MgII_1.MgII_dimensions(MgII_1.start_ind,MgII_1.end_ind)
                MgII_2.MgII_dimensions(MgII_1.start_ind,MgII_1.end_ind)
            else:
                MgII_1.MgII_dimensions(MgII_2.start_ind,MgII_2.end_ind)
                MgII_2.MgII_dimensions(MgII_2.start_ind,MgII_2.end_ind)

            for line in self.absorptions:
                
                if line.name is not None:
                    continue

                potential_lines = [
                (z, element, potential_line) for element, potential_line in zip(elements_to_check, lines_to_check)
                if ((z_low - (.01 * z)) <= (line.peak - potential_line) / potential_line <= (z_high + (.01 * z)))
                ]
            
                line.extend_possible_lines(potential_lines)

            
        for line in self.absorptions:

            if line.name is not None:
                if line.name.split(' ')[0]=='MgII':
                    line.sorted_absorptions=[]
                    line.MgII_dimensions(z_low,z_high,MgII=True)
                    line.update_line_attributes()
                    line.find_mcmc_microlines()
                    continue

            best=line.chose_best_line()

            if best==None:
                continue

            for i in self.absorbers:

                if math.isclose(i.z,best[0],abs_tol=.01):

                    MgII_1=i.lines.get('MgII 2796.355099')

                    z_low=(MgII_1.wavelength[0]-2796.355099)/2796.355099
                    z_high=(MgII_1.wavelength[-1]-2796.355099)/2796.355099

                    line.MgII_dimensions(z_low,z_high)

                    i.add_line(f"{best[1]} {best[2]}",line)

                    line.find_mcmc_microlines()

                    #line.mcmc_microlines=self.optimizedMethod(line.MgII_wavelength,line.MgII_flux,line.MgII_errors,N=.5)
            
                    


            '''


            for k,line in enumerate(lines_to_check):

                if any(line == value for value in (2796.355099, 2803.5322972)):
                    continue

                good_absorptions = [
                absorption for absorption in self.absorptions
                if ((z_low-(.01*z)) <= (absorption.peak - line) / line <= (z_high+(.01*z)) and absorption.name==None)]

                

                suspected_loc = (1 + z) * line

                sorted_absorptions = sorted(good_absorptions, key=lambda line: abs(line.peak - suspected_loc))

                try:
                    suspected_line=sorted_absorptions[0]
                except:
                    continue

                for absorp in sorted_absorptions:
                    if absorp.name==None:
                        continue

                if math.isclose(suspected_loc,suspected_line.peak,abs_tol=20):#abs_tol=50*absorber.z_err):

                    absorber.add_line(f"{elements_to_check[k]} {line}",suspected_line)

                    suspected_line.name=f"{elements_to_check[k]} {line}"

                    suspected_line.z=z

                    suspected_line.suspected_line=line
                    suspected_line.update_microlines()
            '''
                


        '''

    
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
                        found_line.suspected_line_loc=loc
                        found_doublets.append(found_line)

                N_values = [line.find_N(z, strg, dblt) for line, strg, dblt in zip(found_doublets, strength, doublet)]
                ew = [line.actual_equivalent_width() for line in found_doublets]
                std = np.std(N_values)
                mean = np.mean(N_values)
                
                for j, line in enumerate(found_doublets):
                    #TODO consider changing tolerance
                    if math.isclose(line.N, mean, abs_tol=3*std):
                        line.suspected_line_loc=doublet[j]
                        N_check.append(line)

                current_dict[element] = tuple(N_check)
                
                #current_dict[element] = tuple(found_doublets)
        print(self.zs)'''


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

            #print(f"Done with {element}")

    def velocity_plot(self,element):

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

            #print(f"Done with {element}")

    def PlotFlux(self,iteration=0):
        
                # Create the plot with Plotly
        trace_flux = go.Scatter(
        x=self.wavelength, 
        y=self.flux, 
        mode='lines', 
        name='Flux',
        line=dict(color='black',width=1),
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

        # Initialize the list of annotations
        annotations = []

        # Highlight absorption line regions and add labels
        for system in self.absorptions:
            start_wavelength = self.wavelength[system.start_ind]
            end_wavelength = self.wavelength[system.end_ind]

            # Check if the absorption is an MgII doublet
            if system.name is not None:
                if system.name.split(' ')[0]=='MgII':
                    color = 'rgba(0, 0, 0, 0.2)' #mgII is black

                else:
                    color = 'rgba(255, 165, 0, 0.3)' # other absorptions are orange

            else:
                color = 'rgba(0, 255, 0, 0.3)'  # Green for others

            shapes.append(dict(
                type='rect',
                x0=start_wavelength,
                x1=end_wavelength,
                y0=min(self.flux),
                y1=max(self.flux),
                fillcolor=color,
                line=dict(width=0)
            ))

            # Add annotation if the system has a known absorber
            if hasattr(system, 'z') and hasattr(system, 'suspected_line'):
                if system.name is None:
                    label_text = f"Unknown"
                else:
                    label_text = f"z={system.z:.4f}, {system.name.split(' ')[0]}:{int(np.floor(float(system.name.split(' ')[1])))}"
                annotations.append(dict(
                    x=(start_wavelength + end_wavelength) / 2,  # Position annotation at the center
                    y=3,  # Adjust y position slightly below the max flux
                    text=label_text,
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="black",
                    borderwidth=1
                ))

            if len(system.sorted_absorptions)>=2:
                label_text = f"z={system.z:.4f}, {system.sorted_absorptions[1][1]}:{int(np.floor(float(system.sorted_absorptions[1][2])))}"
                annotations.append(dict(
                    x=(start_wavelength + end_wavelength) / 2,  # Position annotation at the center
                    y=2.85,  # Adjust y position slightly below the max flux
                    text=label_text,
                    showarrow=False,
                    font=dict(size=8, color="black"),
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="black",
                    borderwidth=1
                ))


            # Add annotation if the system has a known absorber
            if hasattr(system, 'z') and hasattr(system, 'suspected_line'):
                if system.name is None:
                    label_text = f"Unknown"
                else:
                    label_text = f"z={system.z:.4f}, {system.name.split(' ')[0]}:{int(np.floor(float(system.name.split(' ')[1])))}"
                annotations.append(dict(
                    x=(start_wavelength + end_wavelength) / 2,  # Position annotation at the center
                    y=3,  # Adjust y position slightly below the max flux
                    text=label_text,
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="black",
                    borderwidth=1
                ))
                        
            
            layout = go.Layout(
                title="Spectral Line Analysis",
                xaxis=dict(title='Wavelength'),
                yaxis=dict(title='Flux'),
                shapes=shapes,
                annotations=annotations
            )
        
        try:
            fig = go.Figure(data=[trace_flux, trace_error], layout=layout)
        except:
            fig = go.Figure(data=[trace_flux, trace_error])

        
        # Save the interactive plot as an HTML file
        if self.catalog=="TNG":
            pio.write_html(fig, file=f'static/Trident/FluxPlot_{iteration}.html', auto_open=False)
        else:
            pio.write_html(fig, file='static/Data/FluxPlot.html', auto_open=False)

    def export_absorbers(self):
        for absorber in self.absorbers:
            absorber.export_data()
            absorber.make_vel_plot('MgII')


    def hand_select_absorption(self):
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.patches import Rectangle
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.wavelength, self.flux, label='Flux', color='blue')
        ax.set_title('Select Absorption Range')
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Flux')

        selection = []

        def onselect(eclick, erelease):
            x1, x2 = eclick.xdata, erelease.xdata
            selection.clear()
            selection.append((min(x1, x2), max(x1, x2)))
            ax.add_patch(Rectangle((min(x1, x2), 0), max(x1, x2) - min(x1, x2), 3,
                                edgecolor='red', facecolor='none', linewidth=2))
            fig.canvas.draw()

        from matplotlib.widgets import SpanSelector
        span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                            rectprops=dict(alpha=0.5, facecolor='red'))

        plt.legend()
        plt.show()

        if selection:
            start, end = selection[0]
            return start, end
        else:
            return None




    def DoAll(self):

        #self.ApertureMethod()
        self.optimizedMethod(N=2)

        self.MgII_search()

        self.MgMatch()

        #self.vel_plots()

        self.PlotFlux()

        self.export_absorbers()