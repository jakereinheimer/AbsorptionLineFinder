import numpy as np
import requests
import yt
import trident
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import random

headers = {"api-key":"b07cd06d130435059345a4b063ed29cf"}

def get(path, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r



class Sim_spectra:

    def __init__(self,id):


        self.line_list = [
            # Hydrogen
            'H I 1216', 'H I 1025', 'H I 972',

            # Helium
            'He I 584', 'He II 304',

            # Carbon
            'C I 1560', 'C II 1334', 'C II 1335',
            'C III 977', 'C IV 1548', 'C IV 1550',

            # Nitrogen
            'N I 1200', 'N II 1083', 'N III 991',
            'N V 1238', 'N V 1242',

            # Oxygen
            'O I 1302', 'O II 832', 'O III 702',
            'O IV 787', 'O VI 1031', 'O VI 1037',

            # Magnesium
            'Mg I 2852', 'Mg II 2796', 'Mg II 2803',

            # Silicon
            'Si II 1260', 'Si II 1526', 'Si III 1206',
            'Si IV 1393', 'Si IV 1402',

            # Sulfur
            'S II 1250', 'S II 1253', 'S II 1259',
            'S III 1012', 'S III 1021', 'S IV 1062',
            'S VI 933', 'S VI 944',

            # Iron
            'Fe II 2344', 'Fe II 2383', 'Fe II 2586',
            'Fe II 2600',

            # Aluminum
            'Al II 1670', 'Al III 1854', 'Al III 1862',

            # Calcium
            'Ca II 3934', 'Ca II 3969',

            # Additional Metal Lines
            'Mn II 2576', 'Mn II 2594', 'Mn II 2606',
            'Ni II 1317', 'Ni II 1370'
        ]
            
        self.id=id

        url = "http://www.tng-project.org/api/TNG100-1/snapshots/z=2/subhalos/" + str(self.id) + "/cutout.hdf5"

        #url = "http://www.tng-project.org/api/Illustris-1/snapshots/z=2/subhalos/" + str(id) + "/cutout.hdf5"
        saved_filename = get(url)

        self.distance=2000 #kpc
        self.number_of_rays=2

        self.ds = yt.load(saved_filename)

        center = self.ds.r[:].argmax(("PartType0", "Density"))
        self.center_kpc=np.array([center[0].to_value(),center[1].to_value(),center[2].to_value()])
        print(f'Center:{self.center_kpc}')

        sphere = self.ds.sphere(self.center_kpc, (500, 'kpc'))  # Adjust the size as necessary

        #galaxy radius calculation
        stars = sphere['PartType4', 'particle_position']
        distances = np.sqrt((stars[:,0] - center[0])**2 + (stars[:,1] - center[1])**2 + (stars[:,2] - center[2])**2)
        self.galaxy_radius = np.max(distances).in_units('kpc').to_value()
        print(f"Radius:{self.galaxy_radius}")

        # Calculate the angular momentum vector of the region
        L = sphere.quantities.angular_momentum_vector()
        L = L / np.linalg.norm(L)
        self.L_kpc=np.array([L[0].to_value(),L[1].to_value(),L[2].to_value()])

        print(f"Angular Momentum Vector:{L}")

        # axis plotting
        plots=['x','y','z']
        for i in plots:
            #plot
            p = yt.ProjectionPlot(self.ds, i, ('gas', 'density'), center=self.center_kpc, width=(1000, "kpc"),buff_size=(1000, 1000))
            p.set_cmap(('gas', 'density'), 'inferno')
            p.set_background_color(('gas', 'density'), 'black')
            p.annotate_scale(corner="upper_right")
            p.annotate_timestamp(corner="upper_left", redshift=True, draw_inset_box=True)
            p.annotate_title(f"{i} View")
            
            p.save(f"static/Trident/Trident_plots/plot_{i}.png")

        #down the angular momentum vector
        p = yt.OffAxisProjectionPlot(self.ds, self.L_kpc, ('gas', 'density'), center=self.center_kpc, width=(1000, 'kpc'), buff_size=(1000, 1000))
        p.set_cmap(('gas', 'density'), 'inferno')
        p.set_background_color(('gas', 'density'), 'black')
        p.annotate_scale(corner="upper_right")
        p.annotate_timestamp(corner="upper_left", redshift=True, draw_inset_box=True)
        p.annotate_title("Angular Momentum View")

        radii=np.linspace(50,450,9)
        for r in radii:
            p.annotate_sphere(self.center_kpc,radius=(r, "kpc"))

        p.save(f"static/Trident/Trident_plots/plot_ang_momentum.png")

    def do_Trident(self,d_ray1,theta_ray1,d_ray2,theta_ray2):

        def random_point_in_plane(center, normal, radius):
            """
            Generate a random point in the plane defined by `normal`, centered at `center`,
            within a given `radius`.
            """
            # Generate a random point in a 2D disk
            rand_r = radius * np.sqrt(np.random.uniform(0, 1))  # Ensures uniform distribution
            rand_theta = 2 * np.pi * np.random.uniform(0, 1)

            x = rand_r * np.cos(rand_theta)
            y = rand_r * np.sin(rand_theta)

            # Create two orthonormal basis vectors perpendicular to `normal`
            arbitrary_vector = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
            basis1 = np.cross(normal, arbitrary_vector)
            basis1 /= np.linalg.norm(basis1)

            basis2 = np.cross(normal, basis1)

            # Convert 2D point to 3D in the galaxy plane
            point = self.center_kpc + x * basis1 + y * basis2
            return point

        def point_in_L_plane(center, normal, radius, angle):
            """
            Compute a point in the plane defined by `normal` (angular momentum vector),
            centered at `center`, at a given `radius` and `angle` (relative to the x-direction
            in the L-projected plane).
            """
            #radius/=2
            # Define an arbitrary reference vector to establish the x-axis in the plane
            reference_x = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])

            # Compute orthonormal basis vectors
            basis_x = np.cross(normal, reference_x)  # This is the x-direction in the plane
            basis_x /= np.linalg.norm(basis_x)  # Normalize

            basis_y = np.cross(normal, basis_x)  # This is the y-direction in the plane

            # Convert polar (radius, angle) to Cartesian coordinates in the plane
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            # Transform the 2D coordinates into 3D using basis vectors
            point = center + x * basis_x + y * basis_y
            return point
        
        def degrees_to_radians(degrees):
            return degrees * (np.pi / 180)
        

        if d_ray1 == "random":
            process="random"
        else:
            process="specific"

        data=[d_ray1,degrees_to_radians(theta_ray1),d_ray2,degrees_to_radians(theta_ray2)]

        print(f"data:{data}")

        self.rays=[]
        for i in range(self.number_of_rays):

            if process=="random":
                point = random_point_in_plane(self.center_kpc, self.L_kpc, self.galaxy_radius)
            elif process=="specific":
                point = point_in_L_plane(self.center_kpc,self.L_kpc,data[(i*2)]/2,data[(i*2)+1])

            end_point=self.center_kpc+(self.distance*self.L_kpc)
            start_point=point-(self.distance*self.L_kpc)

            yt_ray=self.ds.ray(start_point,end_point)
            
            ray = trident.make_simple_ray(self.ds,
                                        start_position=yt_ray.start_point,
                                        end_position=yt_ray.end_point,
                                        data_filename=f"ray_{i}.h5",
                                        lines='all',
                                        ftype='gas',
                                        )

            
            self.rays.append(ray)

        #down the angular momentum vector
        p = yt.OffAxisProjectionPlot(self.ds, self.L_kpc, ('gas', 'density'), center=self.center_kpc, width=(1000, 'kpc'), buff_size=(1000, 1000))
        p.set_cmap(('gas', 'density'), 'inferno')
        p.set_background_color(('gas', 'density'), 'black')
        p.annotate_scale(corner="upper_right")
        p.annotate_timestamp(corner="upper_left", redshift=True, draw_inset_box=True)
        p.annotate_title("OffAxis")

        for ray in self.rays:
            p.annotate_ray(ray,arrow=True)

        p.save("/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Trident/Trident_plots/ray_plot.png")

        #now make spectra
        for i,ray in enumerate(self.rays):
            sg = trident.SpectrumGenerator(lambda_min=1200, lambda_max=6000, dlambda=0.1)  # Wavelength range in Angstroms
            
            sg.make_spectrum(ray, lines='all',njobs=-1)
            
            sg.add_qso_spectrum(emitting_redshift=2, observing_redshift=.5)
            sg.add_milky_way_foreground()
            #sg.apply_lsf()
            sg.add_gaussian_noise(30)

            sg.save_spectrum(f'/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Trident/Trident_spectrum/spec_{i}.txt')
            sg.plot_spectrum(f'/Users/jakereinheimer/Desktop/Fakhri/VPFit/static/Trident/Trident_spectrum/spec_{i}.png')


    def get_data(self,path):

        data = pd.read_csv(path, delimiter=" ", header=1, names=['wavelength', 'tau', 'flux', 'error'])

        wavelength=data['wavelength'].to_numpy()
        flux=data['flux'].to_numpy()
        error=data['error'].to_numpy()

        return flux,error,wavelength
    
    def plot(self,data_path,result_path):

        flux,error,wavelength=self.get_data(data_path)

        # Create the plot with Plotly
        trace_flux = go.Line(
        x=wavelength, 
        y=flux, 
        mode='lines', 
        name='Flux',
        line=dict(width=1),
        #line_shape='hv'  # This makes it a horizontal-vertical step plot
        )

        trace_error = go.Line(
        x=wavelength, 
        y=error, 
        mode='lines', 
        name='Error', 
        line=dict(color='red', width=1),
        #line_shape='hv'  # This makes it a horizontal-vertical step plot
        )

        shapes = [
        ]
                
        
        layout = go.Layout(
            title="Spectral Line Analysis",
            xaxis=dict(title='Wavelength'),
            yaxis=dict(title='Flux'),
            shapes=shapes
        )
        
        fig = go.Figure(data=[trace_flux, trace_error], layout=layout)
        
        # Save the interactive plot as an HTML file
        pio.write_html(fig, file=result_path, auto_open=False)
