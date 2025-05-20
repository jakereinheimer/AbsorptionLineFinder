import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objs as go
import plotly.io as pio
import math
import pickle

c_kms = 2.99792458e5  # speed of light in km/s


matplotlib.use('Agg')

#my functions
from essential_functions import get_data,get_custom_data,read_atomDB,clear_directory,record_pair_csv,gaussian_isf
from AbsorptionLine import AbsorptionLine, MicroAbsorptionLine, AbsorptionLineSystem,Absorber,Custom_absorption_line


#helper functions
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Open the file in binary write mode
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)  # Pickle the object and write to file



class VPFit:
    
    def __init__(self,data_loc,catalog,name,custom=False):
        
        self.catalog=catalog
        self.object_name=name
        print(f"Catalog : {self.catalog}")
        print(f"Name : {name}")

        if catalog == 'combined':
            # Handle tuple of (red_paths, blue_paths, red_res, blue_res)
            red_paths, blue_paths, red_res, blue_res = data_loc

            from essential_functions import get_custom_data

            red_flux, red_err, red_wave, _ = get_custom_data(red_paths)
            blue_flux, blue_err, blue_wave, _ = get_custom_data(blue_paths)

            # Merge by concatenation and sort by wavelength
            combined = list(zip(np.concatenate([blue_wave, red_wave]),
                                np.concatenate([blue_flux, red_flux]),
                                np.concatenate([blue_err, red_err])))
            combined.sort(key=lambda x: x[0])  # sort by wavelength

            self.wavelength = np.array([w for w, _, _ in combined])
            self.flux = np.array([f for _, f, _ in combined])
            self.error = np.array([e for _, _, e in combined])

            self.boundry=blue_wave[-1]

            self.red_fwhm=red_res
            self.blue_fwhm=blue_res

            self.emitter_redshift = 2  # placeholder; could allow user input

        elif catalog=='custom':
            if name=='NMF':
                self.flux,self.error,self.wavelength,self.emitter_redshift=get_custom_data(data_loc,nmf=True)
            else:
                self.flux,self.error,self.wavelength,self.emitter_redshift=get_custom_data(data_loc)
                self.p=4
                self.N=3

        else:
            self.flux,self.error,self.wavelength,self.emitter_redshift=get_data(data_loc,catalog,name)
            self.boundry=self.wavelength[-1]


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
            self.tolerance_absorption_systems=30

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

        else:
            self.p=4
            self.N=3
            self.fwhm=10
            self.tolerance_absorption_systems=30


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

        print('wave')
        print(sub_wavelength)

        microlines=self.optimizedMethod(sub_wavelength,sub_flux,sub_error,beg_ind=start_ind,N=2)

        print(microlines)

        abs=AbsorptionLineSystem(self,microlines,full=True)

        abs.set_line_info(element,transition)

        return abs


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
    

    def optimizedMethod(self, wavelength=None, flux=None, error=None, beg_ind=None,N=5):
            
            resolution=self.fwhm
            
            if wavelength is None:

                default_mode=True

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
            padding_length = len(wavelength) - len(detected)
            detected = np.pad(detected, (0, padding_length), constant_values=False)


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

                        if default_mode:

                            if (wavelength[i] > (1215.67 * (1 + self.emitter_redshift))) and (self.wavelength[i] <= 6800):

                                micro_absorption_lines.append(MicroAbsorptionLine(self.wavelength[start_index-1:end_index+1],self.flux[start_index-1:end_index+1],self.error[start_index-1:end_index+1],start_index-1,end_index+1))

                        else:
                            start_index+=beg_ind
                            end_index+=beg_ind
                            micro_absorption_lines.append(MicroAbsorptionLine(self.wavelength[start_index-1:end_index+1],self.flux[start_index-1:end_index+1],self.error[start_index-1:end_index+1],start_index-1,end_index+1))

                        start_index=None
            
            #remove micro lines with error of zero
            #for i,microline in enumerate(self.micro_absorption_lines):
            #    print(microline.number_of_zeros_error)
            #    if microline.number_of_zeros_error<=3:
            #        self.micro_absorption_lines.pop(i)

            #print(f"Found {len(micro_absorption_lines)} microlines")

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

    def churchill_style_absorbers(self, velocity_window=800):
        trans = [
            {"name": "MgII_2796", "lambda": 2796.354, "f": 0.6123, "N_sigma": 5},
            {"name": "MgII_2803", "lambda": 2803.531, "f": 0.3054, "N_sigma": 3},
            {"name": "FeII_2600", "lambda": 2600.173, "f": 0.2239, "N_sigma": 3},
            {"name": "MgI_2852",  "lambda": 2852.964, "f": 1.83,   "N_sigma": 3},
        ]

        def gaussian_isf(fwhm, size=9):
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            x = np.arange(-size, size + 1)
            return np.exp(-0.5 * (x / sigma)**2)

        def compute_ews(wavelength, flux, error, isf):
            ew = np.zeros_like(flux)
            sigma_w = np.zeros_like(error)
            hw = len(isf) // 2
            for i in range(hw, len(flux) - hw):
                window_flux = flux[i-hw:i+hw+1]
                window_wave = wavelength[i-hw:i+hw+1]
                d_lambda = np.diff(window_wave)
                if len(d_lambda) == len(window_flux) - 1:
                    Dn = 1 - window_flux[:-1]
                    ew[i] = -np.sum((isf[:-1]**2) * Dn * d_lambda)
                    sigma_w[i] = np.sqrt(np.sum((error[i-hw:i+hw][:len(d_lambda)] * isf[:-1] * d_lambda)**2))
            return ew, sigma_w

        isf = gaussian_isf(self.fwhm)
        ew, sigma_w = compute_ews(self.wavelength, self.flux, self.error, isf)

        detected_indices = []
        for i, lam_obs in enumerate(self.wavelength):
            z = lam_obs / trans[0]["lambda"] - 1
            if i >= len(ew):
                continue
            if ew[i] < -trans[0]["N_sigma"] * sigma_w[i]:
                richness = 0
                for t in trans[2:]:
                    lam_t = t["lambda"] * (1 + z)
                    k = np.argmin(np.abs(self.wavelength - lam_t))
                    if k < len(ew) and ew[k] < -t["N_sigma"] * sigma_w[k]:
                        richness += 1
                if richness >= 1:
                    detected_indices.append(i)

        detected_indices.sort()

        # Group detections into systems
        systems = []
        current = []
        for i in detected_indices:
            z_i = self.wavelength[i] / trans[0]["lambda"] - 1
            if not current:
                current.append(i)
            else:
                last_z = self.wavelength[current[-1]] / trans[0]["lambda"] - 1
                delta_v = c_kms * (z_i - last_z) / (1 + last_z)
                if delta_v <= velocity_window:
                    current.append(i)
                else:
                    systems.append(current)
                    current = [i]
        if current:
            systems.append(current)

        absorption_systems = []
        for group in systems:
            i = group[0]
            z = self.wavelength[i] / trans[0]["lambda"] - 1
            lam_min = trans[0]["lambda"] * (1 + z) - 2
            lam_max = trans[0]["lambda"] * (1 + z) + 2
            i_range = np.where((self.wavelength >= lam_min) & (self.wavelength <= lam_max))[0]
            if len(i_range) > 0:
                #perhaps add the absorptionlinesystems by finding the microlines from the optimized method
                microlines=self.optimizedMethod(self.wavelength[i_range[0]:i_range[-1]],self.flux[i_range[0]:i_range[-1]],self.error[i_range[0]:i_range[-1]],i_range[0],N=1)
                print(microlines)
                absorption_systems.append(AbsorptionLineSystem(self, microlines))

        pairs=[]
        for line in absorption_systems:

            z_low,z_high=((line.wavelength[0]-2796.354)/2796.354),((line.wavelength[-1]-2796.354)/2796.354)

            start_ind = np.argmin(np.abs(self.wavelength - ((z_low+1)*2803.531)))
            stop_ind = np.argmin(np.abs(self.wavelength - ((z_high+1)*2803.531)))

            #microlines=self.optimizedMethod(self.wavelength[start_ind:stop_ind],self.flux[start_ind:stop_ind],self.error[start_ind:stop_ind],start_ind,N=1)
            #if len(microlines)>0:
            #    pairs.append((line,AbsorptionLineSystem(self, microlines)))
            pairs.append((line,AbsorptionLineSystem(self,(start_ind,stop_ind))))


        validated_systems=[]
        for pair in pairs:

            system_2796=pair[0]
            system_2803=pair[1]

            # Redshift check
            z_2796 = (system_2796.peak - trans[0]["lambda"]) / trans[0]["lambda"]
            z_2803 = (system_2803.peak - trans[1]["lambda"]) / trans[1]["lambda"]
            if abs(z_2796 - z_2803) > 0.01:
                continue

            # Microline count check
            if abs(len(system_2796.microLines) - len(system_2803.microLines)) > 2:
                continue

            # Microline alignment check
            successes = 0
            for m1 in system_2796.microLines:
                for m2 in system_2803.microLines:
                    z1 = (m1.peak - trans[0]["lambda"]) / trans[0]["lambda"]
                    z2 = (m2.peak - trans[1]["lambda"]) / trans[1]["lambda"]
                    if abs(z1 - z2) < 0.001:
                        successes += 1
                        break
            if len(system_2796.microLines) == 0 or successes / len(system_2796.microLines) < 0.8:
                continue

            validated_systems.append((system_2796,system_2803))


        filtered_rows = self.atomDB[self.atomDB['Transition'] == 'MgII']
        MgII_doublet=tuple(filtered_rows['Wavelength'])
        MgII_stregnths=tuple(filtered_rows['Strength'])

        print('churchill')
        for i in validated_systems:
            i[0].z=(i[0].peak-trans[0]["lambda"])/trans[0]["lambda"]
            i[0].z_err=.1

            i[0].suspected_line=MgII_doublet[0]
            i[0].name=f"MgII {MgII_doublet[0]}"
            i[0].update_line_attributes()

            i[1].z=i[0].z
            i[1].z_err=.1

            i[1].suspected_line=MgII_doublet[1]
            i[1].name=f"MgII {MgII_doublet[1]}"
            i[1].update_line_attributes()

            print('pair:')
            print(i[0].wavelength[0],i[0].wavelength[-1])
            print(i[1].wavelength[0],i[1].wavelength[-1])

        '''self.absorptions.append(pair[0])
        self.absorptions.append(pair[1])

        #fill absorbers list
        self.absorbers=[]

        for i,pair in enumerate(validated_systems):

            self.absorbers.append(Absorber(pair[0].z,pair[0].z_err,pair))

        return self.absorbers'''


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
    
    def smart_find(self,start,stop):

        start_ind = np.argmin(np.abs(self.wavelength - start))
        stop_ind = np.argmin(np.abs(self.wavelength - stop))

        line = AbsorptionLineSystem(self,(start_ind,stop_ind))

        #find 2803
        z_low,z_high=((line.wavelength[0]-2796.354)/2796.354),((line.wavelength[-1]-2796.354)/2796.354)

        start_ind = np.argmin(np.abs(self.wavelength - ((z_low+1)*2803.531)))
        stop_ind = np.argmin(np.abs(self.wavelength - ((z_high+1)*2803.531)))

        pair=((line,AbsorptionLineSystem(self,(start_ind,stop_ind))))



        filtered_rows = self.atomDB[self.atomDB['Transition'] == 'MgII']
        MgII_doublet=tuple(filtered_rows['Wavelength'])
        MgII_stregnths=tuple(filtered_rows['Strength'])

        pair[0].set_line_info('MgII',2796)
        pair[0].z=(pair[0].peak-2796.354/2796.354)
        pair[1].set_line_info('MgII',2803)
        pair[1].z=pair[0].z

        for thing in pair:

            thing.actual_ew_func()
            
            thing.calc_linear_n()

        #____________________________________________________________________________________________________
        #then make new absorber if it has a pair

        custom_absorber=Absorber(pair[0].z,.1,pair)


        #____________________________________________________________________________________________________
        #then do the mgII search to find other elements

        lines_to_find={
            'FeII 2600':2600.1720322,
            #'FeII 2586':2586.6492304,
            #'FeII 2382':2382.7639122,
            #'FeII 2374':2374.4599813,
            #'FeII 2344':2344.2126822,
            #'FeII 1608':1608.4509059,
            'MgI 2852': 2852.96342,
            'CaII 3934':3934.774716,
            #'CaII 3969':3969.5897875
        }

        for key,value in lines_to_find.items():

            
            start_ind = np.argmin(np.abs(self.wavelength - ((z_low+1)*value)))
            stop_ind = np.argmin(np.abs(self.wavelength - ((z_high+1)*value)))

            new_absorbtion=AbsorptionLineSystem(self,(start_ind,stop_ind))
            new_absorbtion.name=key
            new_absorbtion.MgII_dimensions(z_low,z_high)
            new_absorbtion.set_line_info(key.split(' ')[0],int(key.split(' ')[1]))

            custom_absorber.add_line(key,new_absorbtion)

        #save some data
        #vel window
        z_gal = 0.6582
        lambda_0=2796.354


        # Your wavelength window in Ångströms
        lambda_min = pair[0].wavelength[0]
        lambda_max = pair[0].wavelength[-1]

        # Wavelength of the MgII line at the galaxy's redshift
        lambda_gal = lambda_0 * (1 + z_gal)

        # Convert to velocity relative to galaxy
        v_min = c_kms * (lambda_min - lambda_gal) / lambda_gal
        v_max = c_kms * (lambda_max - lambda_gal) / lambda_gal


        # Create a dataframe
        df = pd.DataFrame({
            "z_gal": [z_gal],
            "lambda_min": [lambda_min],
            "lambda_max": [lambda_max],
            "velocity_min_kms": [v_min],
            "velocity_max_kms": [v_max]
        })

        for key,value in custom_absorber.lines.items():

            ew,ew_error=value.actual_ew_func()

            df[f"{key} EW"]=f"{ew} +- {ew_error}"



        # Save to CSV
        df.to_csv("absorber_data.csv", index=False)

        

        #____________________________________________________________________________________________________
        #now make a flux plot and save custom absorption

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
        for system in custom_absorber.lines.values():

            start_wavelength = system.wavelength[0]
            end_wavelength = system.wavelength[-1]

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
        
        fig = go.Figure(data=[trace_flux, trace_error], layout=layout)

        # Save the interactive plot as an HTML file
        pio.write_html(fig, file='static/Data/custom_absorber/FluxPlot.html', auto_open=False)

        #____________________________________________________________________________________________________
        #and return the custom absorber and save it too
        save_object(custom_absorber,'static/Data/custom_absorber/custom_absorber.pkl')

        print('smart find completed')

        return custom_absorber

    
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
    
    def upgraded_MgII_search(self):

        filtered_rows = self.atomDB[self.atomDB['Transition'] == 'MgII']
        MgII_doublet=tuple(filtered_rows['Wavelength'])
        MgII_stregnths=tuple(filtered_rows['Strength'])
        EW_ratio=MgII_stregnths[0]/MgII_stregnths[1]

        potential_doublets=[]

    
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

            if math.isclose(z_1,z_2,abs_tol=.01):

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
        
        
        #microline check
        #check just number of microlines
        microline_check=[]
        for i, pair in enumerate(z_check):
            if math.isclose(len(pair[0].microLines),len(pair[1].microLines),abs_tol=2):
                microline_check.append(pair)

        record_pair_csv(microline_check,"microline_check")

        #
        #see if the lines line up
        microline_line_check=[]
        for i, pair in enumerate(microline_check):

            pair0_microlines=[]
            for microline in pair[0].microLines:
                pair0_microlines.append((microline.peak-2796)/2796)

            pair1_microlines=[]
            for microline in pair[1].microLines:
                pair1_microlines.append((microline.peak-2803)/2803)

            success=[]
            for j,micro in enumerate(pair0_microlines):
                for k,micro2 in enumerate(pair1_microlines):
                    if math.isclose(micro,micro2,abs_tol=.001):
                        success.append(1)


            if (np.sum(np.array(success))/len(pair0_microlines))>=.8:
                microline_line_check.append(pair)

        record_pair_csv(microline_line_check,"microline_line_check")


        #go through the surviving pairs and calc EW and N

        for pair in microline_line_check:

            pair[0].find_N()
            pair[0].equivalent_width()

            pair[1].find_N()
            pair[1].equivalent_width()

        

        
        #ew check
        ew_check=[]
        target_ratio = EW_ratio
        for i, pair in enumerate(microline_line_check):
            target_ratio = EW_ratio
            if abs((pair[0].ew / pair[1].ew) - target_ratio) < 1.0:

            #if (1.5<=pair[0].ew/pair[1].ew<=2):
                
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

        
    def MgMatch(self,absorber=None):

        if absorber is None:
            absorber_list=self.absorbers
        else:
            absorber_list=[absorber]

        lines_to_check = self.atomDB["Wavelength"].to_numpy()
        elements_to_check = self.atomDB['Transition'].to_numpy()

        # Create a boolean mask where elements are NOT 'MgII'
        mask = elements_to_check != 'MgII'

        # Apply the mask to filter both arrays
        elements_to_check = elements_to_check[mask]
        lines_to_check = lines_to_check[mask]

        for i,absorber in enumerate(absorber_list):

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
                MgII_1.MgII_dimensions(MgII_1.start_ind,MgII_1.end_ind,MgII=True)
                MgII_2.MgII_dimensions(MgII_1.start_ind,MgII_1.end_ind,MgII=True)
            else:
                MgII_1.MgII_dimensions(MgII_2.start_ind,MgII_2.end_ind,MgII=True)
                MgII_2.MgII_dimensions(MgII_2.start_ind,MgII_2.end_ind,MgII=True)

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

            for i in absorber_list:

                if math.isclose(i.z,best[0],abs_tol=.01):

                    MgII_1=i.lines.get('MgII 2796.355099')

                    z_low=(MgII_1.wavelength[0]-2796.355099)/2796.355099
                    z_high=(MgII_1.wavelength[-1]-2796.355099)/2796.355099

                    line.MgII_dimensions(z_low,z_high)

                    i.add_line(f"{best[1]} {best[2]}",AbsorptionLineSystem(self,()))

                    line.find_mcmc_microlines()

                    line.z=i.z

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

        #self.churchill_style_absorbers()

        self.MgII_search()

        self.MgMatch()

        #self.vel_plots()

        self.PlotFlux()

        self.export_absorbers()