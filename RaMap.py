import numpy as np
import os
import pandas as pd
import magic  

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.weight": "bold",
    "text.usetex": False,
    "mathtext.default": "regular"
})
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.patches as mpatches

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF

from scipy.interpolate import interp1d, griddata

from mpl_toolkits.axes_grid1 import make_axes_locatable

from pybaselines.whittaker import iasls


######################################---USER INPUT---###########################################################################################################################
#################################################################################################################################################################################

# Path to Raman mapping file (CSV or TXT)

map_file = "data/Raman_mapping.<ext>"

# Path to reference spectra files (CSV or TXT)
ref1_path = "data/reference_spectrum1.<ext>"
ref2_path = "data/reference_spectrum2.<ext>"

# Path for saving results  
path_results = "data/"

#################################################################################################################################################################################
#################################################################################################################################################################################

# ======================
# Load mapping file
# ======================

def load_file(dpath):
    '''
    Loads the given mapping dataset.
    Determines whether the file is in CSV or TXT format and processes it accordingly.
    Extracts the necessary information as return values:

        - raman_shifts ... wavenumbers of the Raman spectrum; found in the first row
        - x_positions and y_positions ... the x and y coordinates of the measured spectra; found in the first and second columns, respectively
        - raman_spectra ... the Raman intensities of the measured spectra; found row-wise starting from the second row; 
                            each intensity is associated with a specific x and y value (in the same row)
    '''

    if not os.path.isfile(dpath):
        raise FileNotFoundError(f"File '{dpath}' was not found.")   
    
    mime = magic.from_file(dpath, mime=True)

    _, endung = os.path.splitext(dpath)
    endung = endung.lower()

    # .csv files
    if "csv" in mime or endung == ".csv":
        try:
            df = pd.read_csv(dpath, header=None, delimiter=';')
            raman_shifts_raw = df.iloc[0, 3:].to_numpy(dtype=float)
            data_r = df.iloc[1:, :].copy()
            coords = data_r.iloc[:, :3].to_numpy(dtype=float)
            x_positions_raw=coords[:,0]
            y_positions_raw=coords[:,1]
            raman_spectra_raw = data_r.iloc[:, 3:].to_numpy(dtype=float)
            print(f"{dpath} (.csv) file successfully loaded.")
            return raman_shifts_raw, x_positions_raw, y_positions_raw, raman_spectra_raw
        except Exception as e:
            print(f"Error when reading as .csv: {e}")

    # .txt files
    if "text" in mime or endung == ".txt":
        try:
            with open(dpath, 'r', encoding='latin1') as f:
                header = f.readline().strip().split()
                raman_shifts_raw = np.array(header, dtype=float)
                
            print(f"{dpath} (.txt) file successfully loaded.")
            data = np.loadtxt(map_file,skiprows=1)
            y_positions_raw = data[:, 0].astype(float)
            x_positions_raw = data[:, 1].astype(float)
            raman_spectra_raw = data[:, 2:]
            return raman_shifts_raw, x_positions_raw, y_positions_raw, raman_spectra_raw
        
        except Exception as e:
            print(f"Error when reading as .txt: {e}")

    raise ValueError(f"Unknown or unsupported file type: {mime}")

raman_shifts, x_positions, y_positions, raman_spectra = load_file(map_file)

# ======================
# Load reference files
# ======================

def load_ref(reference, delimiter=None):
    '''
    Loads the given reference datasets.
    Determines whether the file is in CSV or TXT format and processes it accordingly.
    Extracts the necessary information as return values:

        - raman_shifts ... wavenumbers of the Raman spectrum; found in the first row
        - intensities ... the Raman intensities of the spectrum; found in the second row
    '''

    if not os.path.isfile(reference):
        raise FileNotFoundError(f"File '{reference}' was not found.")

    mime = magic.from_file(reference, mime=True)
    _, endung = os.path.splitext(reference)
    endung = endung.lower()

    try:
        # .csv files
        if "csv" in mime or endung == ".csv":
            df = pd.read_csv(reference, header=None, delimiter=';')
            raman_shifts = df.iloc[:, 0].to_numpy(dtype=float)
            intensities = df.iloc[:, 1].to_numpy(dtype=float)
            print(f"{reference} (.csv) successfully loaded.")
            return raman_shifts, intensities

        # .txt files
        if "text" in mime or endung == ".txt":
            data = np.loadtxt(reference, delimiter=delimiter, encoding='latin1')
            raman_shifts = data[:, 0].astype(float)
            intensities = data[:, 1].astype(float)
            print(f"{reference} (.txt) successfully loaded.")
            return raman_shifts, intensities

    except Exception as e:
        raise ValueError(f"Error while loading Raman spectrum: {e}")

    raise ValueError(f"Unknown or unsupported file type: {mime}")

######################################---USER INPUT---###########################################################################################################################
#################################################################################################################################################################################

# Reference spectra are stored in a dictionary

dict_ref = {
    "ref1":{
    "ref_shifts":load_ref(ref1_path)[0],   
    "ref_spectrum":load_ref(ref1_path)[1],
    "plot color":"",
    "chemical formula":""
    },
    "ref2":{ ... },
  	}

#################################################################################################################################################################################
#################################################################################################################################################################################

# ======================
# Some more functions
# ======================

# Function for saving of figures
run_func = input("Do you want to view and save intermediate results? (y/n): ").strip().lower()
def Save_Figure(filename, decision):
    '''
    The user can decide whether the results should be saved as a PNG (yes/no).
    If the file already exists at the specified path, a prompt will be displayed asking whether the file should be overwritten or not.
    '''

    if decision == 'y':
        if os.path.exists(os.path.join(path_results, f'{filename}.png')):
            user_input = input(f"\n The file '{filename}.png' already exists. Do you want to overwrite it? (y/n): ").strip().lower()
            if user_input == 'y':
                plt.savefig(os.path.join(path_results, f'{filename}.png'), dpi=600, bbox_inches='tight')
                print(f"File '{filename}' has been overwritten.")
            else:
                print("File save has been canceled. No changes made.")
        else:
            plt.savefig(os.path.join(path_results, f'{filename}.png'), dpi=600, bbox_inches='tight')
            print(f"Saved {filename}.png")
    else:
        plt.close()


# Function for customized ticks for spatial map
def custom_ticks(x, pos):
    if x.is_integer(): 
        return f'{int(x)}'
    else:
        return ''  
    
# Function for custom ticks on x-axis in the Raman spectrum
def custom_xticks(x, pos):
    if x%200==0: 
        return f'{int(x)}'
    else:
        return ''  
    
# Min-Max normalization, [0,1]
def min_max(spectrum):

    return  (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min()) 

# Baseline
def baseline(y, lam=1e5, p=0.01):

    baseline, _ = iasls(y, lam=lam, p=p)

    return baseline


# Interpolate x-axis to  get the same number of measurement points
def interpol_x(reference, lb, ub):
    '''
    This function interpolates the x-axis (wavenumbers) of the measured spectra of the mapping 
    and the reference spectra so that they have the same number of points inside the selected wavenumber range (Region of Interest, RoI). 
    Otherwise, further processing is not possible.
    It returns the values:

        - interp_ref ... the interpolated Raman intensity values of the reference spectra
        - raman_shift_RoI ... the wavenumbers of the Region of Interest (RoI)
        - raman_signal_RoI ... the interpolated Raman intensity values of the mapping spectra
    '''

    # Region of interest (RoI) for mapping spectra
    RoI_map = (raman_shifts > lb)&(raman_shifts < ub) 
    raman_shift_RoI = raman_shifts[RoI_map]
    raman_signal_RoI =  raman_spectra[:,RoI_map]

    raman_shifts_ref  = dict_ref[reference]['ref_shifts']
    raman_signal_ref = dict_ref[f'{reference}']['ref_spectrum']
    
    RoI = (raman_shifts_ref > lb)&(raman_shifts_ref < ub) 
    raman_shift_ref_RoI = raman_shifts_ref[RoI]
    raman_signal_ref_RoI =  raman_signal_ref[RoI]
    
    f_intp = interp1d(raman_shift_ref_RoI, raman_signal_ref_RoI, kind='linear',  bounds_error=False, fill_value='extrapolate')

    # New intensities for reference spectrum 
    interp_ref = f_intp(raman_shift_RoI) 
    
    return interp_ref, raman_shift_RoI, raman_signal_RoI

# ======================
# Preprocessing spectra
# ======================

# Set Region of Interest (RoI); limits of the Raman spectrum

## Ask user to set the limits of the Raman spectra
lower_bound, upper_bound = map(float,input("\n Which wavenumber range should be considered?\n"
                             "Please enter the values in the format: minimum,maximum ").split(","))

## Check if the spectrum limits are within the dataset
## and ensure that the lower bound is smaller than the upper bound
if  lower_bound >= raman_shifts.min() and upper_bound <= raman_shifts.max() and lower_bound < upper_bound:
    pass
elif lower_bound >= upper_bound:
    raise ValueError("The lower limit is equal to or greater than the upper limit.")

elif lower_bound < raman_shifts.min() or upper_bound > raman_shifts.max():
    raise ValueError("The specified limits are outside the range of the Raman spectrum.")


###############################################################

# Preprocessing loop for Reference data

corrected_references = {}
for ref_name in dict_ref.keys():
    # Interpolation x-axis
    ## common_axis: the newly adjusted x-axis
    ## new_ref_intensities: the aligned values of the reference spectra
    ## raman_intensities_mapping: the adjusted values of the Raman mappings
    new_ref_intensities, common_axis, raman_intensities_mapping = interpol_x(ref_name, lower_bound, upper_bound)
    raman_intensities_mapping = np.array(raman_intensities_mapping, dtype=np.float64)

    # Baseline removal - reference spectrum
    baseline_est_ref = baseline(new_ref_intensities)
    corrected_signal_ref = new_ref_intensities - baseline_est_ref  

    # Min-Max normalisation
    corrected_signal_ref = min_max(corrected_signal_ref)

    # Alignment of the reference spectra; important for statistical methods
    new_ref_intensities = (new_ref_intensities - np.mean(new_ref_intensities)) / np.std(new_ref_intensities)
    new_ref_intensities = np.nan_to_num(new_ref_intensities, nan=0.0)

    # Plot - corrected reference spectrum  
    plt.plot(common_axis, corrected_signal_ref, color=dict_ref[ref_name]['plot color'])
    plt.title(f"Preprocessed Reference: {ref_name} ({dict_ref[ref_name]['chemical formula']})")
    plt.xlabel("Wavenumber / cm$^{-1}$")
    plt.ylabel("Normalized Raman intensity / arb. units")
    plt.ylim([0,1])
    plt.xlim([lower_bound,upper_bound])
    plt.gca().xaxis.set_major_locator(MultipleLocator(50))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_xticks))
    plt.tight_layout()
    Save_Figure(ref_name, f'{run_func}')
    plt.show()

    # Adding the proprocessed reference spectrum to the dictionary
    dict_ref[ref_name]['normalized_ref_spectrum'] = {
        'common_axis': common_axis, 
        'new_ref_intensities': corrected_signal_ref
    }

###############################################################
    
# Preprocessing mapping data
    
## Baseline removal and PL-detection
    
###  Create empty NumPy arrays for the estimated baseline and the corrected spectra
baseline_est = np.empty_like(raman_intensities_mapping)
corrected_signal = np.empty_like(raman_intensities_mapping)

pl_threshold = float(input("\n Please enter a value for the PL threshold. (Set 0 if no PL should be considered):"))    # It gives the intensity ratio between the Raman peak and the PL signal; no PL influence: = 0

if pl_threshold == '' or pl_threshold<0 or pl_threshold>1:
    raise ValueError("Please enter a valide PL threshold.")

### Create an empty NumPy array with the length equal to the number of mapping positions
###  Store boolean values in pl_mask indicating whether PL is dominant or not
pl_mask = np.zeros(raman_intensities_mapping.shape[0], dtype=bool)

### Loop over all mapping spectra
for i in range(raman_intensities_mapping.shape[0]):
    spectrum = np.array(raman_intensities_mapping[i])
    
    # Baseline
    b = baseline(spectrum)
    baseline_est[i, :] = b
    
    # Baseline-subtracted spectrum
    corrected_signal[i, :] = spectrum - b
    
    # Calculate the maximum value of the baseline-corrected spectrum (i.e., the maximum Raman peak signal)
    raman_peak_max = np.max(corrected_signal[i, :])
    raman_peak_idx = np.argmax(corrected_signal[i, :]) # Index of the maximum Raman peak signal

    # Calculate the value of the baseline at raman_peak_idx (i.e., the corresponding PL signal)
    baseline_max = b[raman_peak_idx]
    
    # Consider spectrum as PL-dominant only if the ratio between the baseline and the maximum Raman peak signal is high (pl_threshold < ratio <=1)
    # If this is the case, a boolean True is assigned here
    if baseline_max/(raman_peak_max + baseline_max) > pl_threshold:
        pl_mask[i] = True

    # Min-Max normalisation
    corrected_signal[i] = min_max(corrected_signal[i])

###############################################################

# Plot sample spectrum with coordinates (x_position_sample , y_position_sample)

## If the output of intermediate results is suppressed, this step is skipped
    
## Ask the user if a sample spectrum should be plotted
if run_func == 'y':
    u_input = input("\n Should a sample spectrum from the mapping be plotted? \n"
                    "If yes, please enter the coordinates in the following format: x,y. \n"
                     "If not, please type None.").strip()
else:
    u_input = "none"

## If not, the user can type 'None' to skip plotting
if u_input.lower() == "none" or u_input == "":       
    x_position_sample = None
    y_position_sample = None

 ## If yes, prompt for the coordinates in the format x,y
else:      
    try:
        x_position_sample, y_position_sample = map(float, u_input.split(","))    
    
    # Catches invalid input, e.g., non-integer values or wrong format
    except ValueError:
        print("Invalid input! Please enter x,y or None.")           


###############################################################
        
# Spatial unit for mapping
        
## Ask the user which spatial unit should be used for the mapping
unit_input = input("\n Please enter the units of the mapping.\n"
                    "(Default is micrometer). Press Enter to use the default: ").strip()

## Provide a default value if no input is given
if unit_input == "":
   unit = "$\mu$m" # micrometer in Latex notation

## Set the unit to the user-provided string (e.g. mm, cm, nm)
else:
    unit = unit_input

## Check the coordinates provided by the user
## No positional coordinates for sample spectrum are given (= None) -> skipping the plot
if x_position_sample is None or y_position_sample is None:
    print("Plotting of sample spectrum skipped. No positional values are given.")

## If x,y coordinates were provided  
else:
    idx = np.where((x_positions == x_position_sample) & (y_positions == y_position_sample))[0]
    # Determination of mapping spectrum that satifies the position requests
    if idx.size>0:
        plt.plot(common_axis, corrected_signal[idx[0]], color = 'Black')
        plt.title(f"Preprocessed sample spectrum at (x,y) = ({x_position_sample},{y_position_sample}) {unit}")
        plt.xlabel("Wavenumber / cm$^{-1}$")
        plt.ylabel("Normalized Raman intensity / arb. units")
        plt.ylim([0,1])
        plt.xlim([lower_bound,upper_bound])
        plt.gca().xaxis.set_major_locator(MultipleLocator(50))
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
        plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_xticks))
        plt.tight_layout()
        Save_Figure(f'Sample spectrum', f'{run_func}')
        plt.show()
    else: 
        # Coordinates are not found in mapping data set -> skipping the plot
        print("Plotting of sample spectrum skipped. Coordinates not found in data.")

# Calculating Signal-to-noise ratio

## Threshold for SNR
snr_threshold = float(input("\n Please enter a value for the SNR threshold:"))  

if snr_threshold == '' or snr_threshold <= 0:
    raise ValueError("Please enter a valide SNR threshold.")


## Set NaNs to zero
corrected_signal = np.nan_to_num(corrected_signal, nan=0.0)

## Function to find a region in a spectrum with no Raman signal
def find_noise_region(spectrum, window_size=50):
    '''
    This function automatically searches for an appropriate region within a spectrum with 50 points 
    for the calculation of the noise level (standard deviation of the noise).
    This region should not contain any Raman signals, which is why the window is chosen 
    where the standard deviation of the values is the smallest.
    The output values are the indices representing the start and end points of the window.
    '''

    min_std = float('inf')
    best_start = 0
    for i in range(0, len(spectrum) - window_size + 1):
        window = spectrum[i:i + window_size]
        std = np.std(window)  
        if std < min_std:

            min_std = std
            best_start = i
    return slice(best_start, best_start + window_size)

## Calculate SNR per spectrum
std_per_spectrum = []
noise_std_per_spectrum = []

for spectrum in corrected_signal:
    # Total std of spectrum
    spectrum_std = np.std(spectrum)
    std_per_spectrum.append(spectrum_std)
    
    # Noise std in "quietest" window
    noise_slice = find_noise_region(spectrum)
    noise_std = np.std(spectrum[noise_slice])
    noise_std_per_spectrum.append(noise_std)

std_per_spectrum = np.array(std_per_spectrum)
noise_std_per_spectrum = np.array(noise_std_per_spectrum)

## Compute SNR
snr_per_spectrum = std_per_spectrum / noise_std_per_spectrum

## Mask spectra below threshold
mask_good = snr_per_spectrum >= snr_threshold
snr_masked = np.where(mask_good, snr_per_spectrum, 0)

## Find indices where snr_masked == 0
indices = np.where(snr_masked == 0)[0]


if len(indices) > 0:
    print("\n There are spectra with SNR < threshold or with intensive PL.\n"
          f"The corresponding spectra ({np.sum(snr_masked == 0)}) will therefore be excluded from the following analysis.")

    # Collecting the corresponding spectra
    selected_spectra = corrected_signal[indices,:]   # shape: (n_selected, n_points)

    # Collect corresponding coordinates
    selected_x = x_positions[indices]  # shape: (n_selected,)
    selected_y = y_positions[indices]  # shape: (n_selected,)

    # Transpose -> (n_points, n_selected)
    selected_spectra = selected_spectra.T

    # Prepend coordinates as first two rows
    coords = np.vstack((
        selected_x[np.newaxis, :],  # first row = X
        selected_y[np.newaxis, :]   # second row = Y
    ))  # shape: (2, n_selected)

    # Combine coordinates and spectra
    result = np.vstack((coords, selected_spectra))  # shape: (2 + n_points, n_selected)
    
    # Create an array for the first column (same number of rows as result_matrix)
    first_col = np.empty((result.shape[0], 1), dtype=float)
    first_col[:] = np.nan               # Initialize with empty strings
    first_col[2:, 0] = common_axis  # Fill wavelengths starting from the 3rd row

    # Combine first column with the data
    result_save = np.hstack((first_col, result))
    
    # Check if the file already exists
    if os.path.exists(os.path.join(path_results, "Excluded_spectra.txt")):
            user_input = input("The file 'Excluded_spectra.txt' already exists. Do you want to overwrite it? (y/n): ").strip().lower()
            if user_input == 'y':
                np.savetxt(os.path.join(path_results,"Excluded_spectra.txt"), result_save)
                print("File 'Excluded_spectra.txt' has been overwritten.")
            else:
                print("File save has been canceled. No changes made.")
    else:
        np.savetxt(os.path.join(path_results,"Excluded_spectra.txt"), result_save)
        print("Spectra saved as 'Excluded_spectra.txt'.")

###############################################################

## Set PL-dominant spectra to zero
corrected_signal[pl_mask, :]=0 

## Set spectra with low SNR to zero
corrected_signal[~mask_good, :] = 0

###############################################################

# Plot SNR map
plt.figure(figsize=(8, 6))

if snr_per_spectrum.min() >= snr_threshold:
    sc = plt.scatter(x_positions, y_positions, c=snr_per_spectrum, cmap='Blues', s=30, vmin=0)
    plt.colorbar(sc, label='SNR per spectrum').ax.axhline(snr_threshold, color='red')
else:
    mask_b = np.all(corrected_signal == 0, axis=1)  # True, when row = 0
    mask_below = np.where(mask_b)[0]#mask_bad

    mask_above = np.where(~mask_b)[0]#mask_good
    
    # Plot good points
    sc = plt.scatter(x_positions[mask_above], y_positions[mask_above],
                     c=snr_per_spectrum[mask_above], cmap='Blues', s=30, vmin=0)
    # Plot bad points in red
    plt.scatter(x_positions[mask_below], y_positions[mask_below], c='red', label='excluded positions', s=30)
    plt.legend(bbox_to_anchor=(0.5,-0.2), fontsize=10)
    plt.colorbar(sc, label='SNR per spectrum').ax.axhline(snr_threshold, color='red')

plt.xlabel(f'x-coordinate / {unit}')
plt.ylabel(f'y-coordinate / {unit}')
plt.title('SNR mapping')
plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))       
plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_ticks))
plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_ticks))
plt.gca().set_aspect('equal')
Save_Figure('SNR_map', f'{run_func}')
plt.show()

# ================================================
# Non-negative matrix factorization (unsupervised)
# ================================================

# Find optimal number of components

## Shift spectra to non-negative 
corrected_signal_shifted = corrected_signal - np.min(corrected_signal, axis=1, keepdims=True)

## Calculating explained variance

### Total norm of corrected data matrix
total_energy = np.linalg.norm(corrected_signal_shifted, 'fro')**2 #

components_range = range(1, 10)
explained_variance_list = []

for n in components_range:
    nmf = NMF(n_components=n, init='nndsvda', max_iter=500, solver='mu') 
    w = nmf.fit_transform(corrected_signal_shifted)
    h = nmf.components_

    ## Reconstructed matrix
    corrected_signal_shifted_reconstructed = w @ h
    ## Frobenius error
    reconstruction_error = np.linalg.norm(corrected_signal_shifted - corrected_signal_shifted_reconstructed, 'fro')**2

    ## Explained variance
    explained_variance = 1 - (reconstruction_error / total_energy)

    explained_variance_list.append(explained_variance)

### Determination of elbow point
def calculate_elbow(x, y):
    # Line between first and last point
    line_vec = np.array([x[-1]-x[0], y[-1]-y[0]])
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    # Distance of each point to the line
    distances = []
    for xi, yi in zip(x, y):
        point_vec = np.array([xi - x[0], yi - y[0]])
        proj = np.dot(point_vec, line_vec_norm) * line_vec_norm
        dist_vec = point_vec - proj
        distances.append(np.linalg.norm(dist_vec))
   
    elbow_index = np.argmax(distances) # Elbowpoint has the largest distance to the line
    return x[elbow_index]

elbow = calculate_elbow(components_range, explained_variance_list)


###############################################################

### Plot
plt.plot(components_range, explained_variance_list, marker='o')
plt.plot(components_range[elbow-1], explained_variance_list[elbow-1], marker='o', color='red', markersize=10) # Highlight the elbow point
plt.xlabel('Number NMF components $K$')
plt.ylabel('Explained variance')
Save_Figure('NMF_number_of_components', f'{run_func}')
plt.show()

###############################################################
 
# NMF decomposition

n_components = elbow # Set optimal number of components to elbow point

## NMF model
model = NMF(n_components=n_components, init='nndsvda', max_iter=500,  solver='mu') #nndsvda
W = model.fit_transform(corrected_signal_shifted)
H = model.components_

####--> End of NMF decomposition

## Calculation Mean Squared Error of reconstructed NMF model
X_reconstructed = W @ H
mse_values = np.mean((corrected_signal_shifted - X_reconstructed)**2, axis=1)

## Comparison NMF endmember with references, cosine similarity
reference_spectra = [dict_ref[ref]['normalized_ref_spectrum']['new_ref_intensities'] for ref in dict_ref.keys()]
dict_keys = list(dict_ref.keys())
NMF_phases = []
for i, comp_std in enumerate(H):
    comp_reshaped = comp_std.reshape(1, -1)
    sims = [cosine_similarity(comp_reshaped, ref_std.reshape(1, -1))[0,0] for ref_std in reference_spectra]
    
    best_ref_idx = np.argmax(sims)
    best_ref_key = dict_keys[best_ref_idx]
    NMF_phases.append(best_ref_key)
    
    # Adding to dictionary
    dict_ref[best_ref_key]['NMF_decomposition'] = {
        'NMF_weighting': W[:, i],
        'cosine_similarity': sims[best_ref_idx]
        } 
     
    ###############################################################

    # Plot
    fig, axs = plt.subplots(1,2,figsize=(14,5))

    ## Component spectrum vs best reference spectrum with highest similarity
    axs[0].plot(common_axis, min_max(H[i]), label=f"Endmember {i+1}", color='black') # Plot component spectra
    axs[0].plot(common_axis, dict_ref[best_ref_key]['normalized_ref_spectrum']['new_ref_intensities'], label=f"Best reference: {dict_ref[best_ref_key]['chemical formula']}", color=dict_ref[best_ref_key]['plot color']) # Plot reference spectra
    axs[0].legend()
    axs[0].set_xlabel("Wavenumber / cm$^{-1}$")
    axs[0].set_ylabel("Normalized Raman intensity / arb. unit")
    axs[0].xaxis.set_major_locator(MultipleLocator(50))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.1))
    axs[0].xaxis.set_major_formatter(FuncFormatter(custom_xticks))
    axs[0].set_xlim(common_axis.min(), common_axis.max())
    axs[0].set_ylim(0, 1)

    ## Spatial distribution weighting factors
    custom_map = LinearSegmentedColormap.from_list("custom_cmap", ['#FFFFFF', dict_ref[best_ref_key]['plot color']])
    sc = axs[1].scatter(x_positions, y_positions, c=W[:, i], cmap=custom_map, s=30, vmin=0, vmax=1)
    axs[1].set_xlabel(f'x-coordinate / {unit}')
    axs[1].set_ylabel(f'y-coordinate / {unit}')
    axs[1].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[1].yaxis.set_major_locator(MultipleLocator(0.5))  
    axs[1].xaxis.set_major_formatter(FuncFormatter(custom_ticks))
    axs[1].yaxis.set_major_formatter(FuncFormatter(custom_ticks))
    axs[1].set_aspect('equal')
    plt.colorbar(sc, ax=axs[1], label=f'Weighting factor of endmember {i+1}')
    plt.tight_layout()
    Save_Figure(f"NMF_component_{i+1}_map", f'{run_func}')
    plt.show()

###############################################################
    
# Add NMF assignment of a reference phase to the dictionary
for ref in dict_keys:
        if ref in NMF_phases:
            dict_ref[ref]['NMF_assignment'] = True
        else:
            dict_ref[ref]['NMF_assignment'] = False


###############################################################
            
# Plot Mean Squared Error map 
plt.figure(figsize=(8,6))
scatter = plt.scatter(x_positions, y_positions, c=mse_values, s=30, cmap='BuPu', vmin=0)
plt.colorbar(scatter, label='Mean Squared Error')
plt.title('Mean Squared Error map')
plt.xlabel(f'x-coordinate / {unit}')
plt.ylabel(f'y-coordinate / {unit}')
plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))  
plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_ticks))
plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_ticks))
plt.gca().set_aspect('equal')
Save_Figure("NMF_Mean_Squared_Error", f'{run_func}')
plt.show()

# ==================
# Cosine similarity
# ==================

corrected_signal = np.nan_to_num(corrected_signal, nan=0.0) 
zero_mask = np.all(np.isclose(corrected_signal, 0), axis=1)

# Iterating through the list of reference spectra
for ref_name, ref_data in dict_ref.items():
    new_ref_intensities = dict_ref[ref_name]['normalized_ref_spectrum']['new_ref_intensities']
    new_ref_intensities = (new_ref_intensities - np.mean(new_ref_intensities)) / np.std(new_ref_intensities)
    new_ref_intensities = np.nan_to_num(new_ref_intensities, nan=0.0)

    ## Calculation of cosine similarity per Raman mapping spectrum
    similarity_per_point = (cosine_similarity(corrected_signal, new_ref_intensities.reshape(1, -1)).flatten())

    ### Set cosine similarity to zero for previously excluded spectra (for plotting)
    similarity_per_point[zero_mask] = 0
   
    # Adding cosine similatiry to dictionary for each reference phase
    dict_ref[ref_name]['Cosine_similarity'] = similarity_per_point

    ###############################################################

    # Plot cosine similarity to each reference phase
    custom_map = LinearSegmentedColormap.from_list("custom_cmap", ['#FFFFFF', dict_ref[ref_name]['plot color']])
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x_positions, y_positions, c=similarity_per_point, cmap=custom_map, s=30, vmin=0, vmax=1)
    plt.colorbar(sc, label=f'Cosine similarity')
    plt.xlabel(f'x-coordinate / {unit}')
    plt.ylabel(f'y-coordinate / {unit}')
    plt.title(f"{dict_ref[ref_name]['chemical formula']}")
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_ticks))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_ticks))
    plt.gca().set_aspect('equal')
    Save_Figure(f"Cosine_similarity_map_{ref_name}", f'{run_func}')
    plt.show()

###############################################################
    
# Plot the reference phase with highest similarity per point
similarities = []
ref_names = list(dict_ref.keys())
for ref_name in dict_ref:
    similarities.append(dict_ref[ref_name]['Cosine_similarity']) # Collect the cosine similarities for each reference in a list
similarity_matrix = np.array(similarities)
max_indices = np.argmax(similarity_matrix, axis=0) # Returns the index with the maximum similarity for each position in the matrix
remaining_refs = []

plt.figure(figsize=(7, 5))
for i, ref_name in enumerate(dict_ref.keys()): # Iterates through the max indices and masks the positions where a phase has the maximum similarity
    mask = (max_indices == i) & (~zero_mask)
    if np.any(mask):
        remaining_refs.append(ref_name)
        dict_ref[ref_name]['Cosine_similarity_assignment'] = True # If a phase has the highest similarity, then according to the cosine similarity, it is also present in the region; 
                                                                    # in the dictionary, this is stored as 'Cosine_similarity_assignment' = True
        sc = plt.scatter(
            x_positions[mask],
            y_positions[mask],
            color=dict_ref[ref_name]['plot color'],
            s=30
        )
    else:
        dict_ref[ref_name]['Cosine_similarity_assignment'] = False

plt.xlabel(f'x-coordinate / {unit}')
plt.ylabel(f'y-coordinate / {unit}')
plt.title('Reference phase with highest cosine similarity')
plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))
plt.gca().set_aspect('equal')
plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_ticks))
plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_ticks))
patches = [mpatches.Patch(color=dict_ref[ref_name]['plot color'], label=dict_ref[ref_name]['chemical formula']) for ref_name in remaining_refs]
plt.legend(handles=patches,  loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
plt.tight_layout()  
Save_Figure('Cosine_similarity_map_best_reference', f'{run_func}')
plt.show()

# ==========================================
# Compare NMF and cosine similarity results
# ==========================================

# Reference names
ref_names = list(dict_ref.keys())

# Collecting Cosine Similarities 
similarity_matrix = np.array(
    [dict_ref[ref]['Cosine_similarity'] for ref in ref_names]
)
# Index with highest cosine similarity per point
max_indices = np.argmax(similarity_matrix, axis=0)
nmf_flags = np.array(
    [dict_ref[ref]['NMF_assignment'] for ref in ref_names]
)

# Final mask (1 = NMF agreement, 0 = conflict)
mask_uncertanity = nmf_flags[max_indices].astype(int)

# Find indices where mask_uncertanity == 0
indices_un = np.where(mask_uncertanity == 0)[0]

if len(indices_un) > 0:
    print("\n There is inconsistency between the NMF decomposition and the cosine similarity results.\n"
          f"The corresponding spectra ({np.sum(mask_uncertanity == 0)}) will therefore be excluded from the following analysis.")

    # Collecting the corresponding spectra
    selected_spectra = corrected_signal[indices_un]   # shape: (n_selected, n_points)

     # Collect corresponding coordinates
    selected_x = x_positions[indices_un]  # shape: (n_selected,)
    selected_y = y_positions[indices_un]  # shape: (n_selected,)

    # Transpose -> (n_points, n_selected)
    selected_spectra = selected_spectra.T

    # Prepend coordinates as first two rows
    coords = np.vstack((
        selected_x[np.newaxis, :],  # first row = X
        selected_y[np.newaxis, :]   # second row = Y
    ))  # shape: (2, n_selected)

    # Combine coordinates and spectra
    result = np.vstack((coords, selected_spectra))  # shape: (2 + n_points, n_selected)
    
    # Create an array for the first column (same number of rows as result_matrix)
    first_col = np.empty((result.shape[0], 1), dtype=float)
    first_col[:] = np.nan               # Initialize with empty strings
    first_col[2:, 0] = common_axis  # Fill wavelengths starting from the 3rd row

    # Combine first column with the data
    result_save = np.hstack((first_col, result))

    ''' # Array
    result = np.column_stack((common_axis, selected_spectra))'''
    
    # Check if the file already exists
    if os.path.exists(os.path.join(path_results, "Inconsistency_spectra.txt")):
            user_input = input("The file 'Inconsistency_spectra.txt' already exists. Do you want to overwrite it? (y/n): ").strip().lower()
            if user_input == 'y':
                np.savetxt(os.path.join(path_results,"Inconsistency_spectra.txt"), result_save)
                print("File 'Inconsistency_spectra.txt' has been overwritten.")
            else:
                print("File save has been canceled. No changes made.")
    else:
        np.savetxt(os.path.join(path_results,"Inconsistency_spectra.txt"), result_save)
        print("Spectra saved as 'Inconsistency_spectra.txt'.")
         

# ================================================================
# Combination NMF weighting and cosine similarity: Combined score
# ================================================================

for ref, ref_data in dict_ref.items():
    if ref_data.get('Cosine_similarity_assignment') and ref_data.get('NMF_assignment'): # Iterate only over references that satisfy both conditions: Assignment by Cosine Similarity and NMF decomposition

        # Calculating combined_score
        combined_score = min_max(ref_data['NMF_decomposition']['NMF_weighting'] * ref_data['Cosine_similarity']) * mask_uncertanity # mask_uncertainty sets the combined score to zero at positions in the mapping where 
                                                                                                                                    # cosine similarity (highest value) and NMF did not yield a unique phase assignment. 
                                                                                                                                    # These points should generally be excluded from further analysis.
        custom_map = LinearSegmentedColormap.from_list("custom_cmap", ['#FFFFFF', ref_data['plot color']])    

        ## Plot combined score
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(x_positions, y_positions, c=combined_score, cmap=custom_map, s=30, vmin=0, vmax=1)
        plt.colorbar(sc, label=f'Combined score')
        plt.xlabel('x-coordinate / µm')
        plt.ylabel('y-coordinate / µm')
        plt.title(f"{ref_data['chemical formula']}")
        plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))
        plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_ticks))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_ticks))
        plt.gca().set_aspect('equal')
        Save_Figure(f'Combined_score_map_{ref}', f'{run_func}')
        plt.show()

# ========================
# Plotting phase mapping
# ========================

        points = np.column_stack((x_positions, y_positions))
        grid_x, grid_y = np.mgrid[min(x_positions):max(x_positions):1000j,
                                min(y_positions):max(y_positions):1000j]
        grid_z = griddata(points, combined_score, (grid_x, grid_y), method='cubic')
        fig, ax = plt.subplots(figsize=(6, 8))
        im = ax.imshow(grid_z.T, extent=(min(x_positions), max(x_positions), min(y_positions), max(y_positions)), origin='lower', cmap=custom_map, aspect='equal', vmin=0)
        scale = make_axes_locatable(ax)
        cax = scale.append_axes("right", size="5%", pad=0.2)
        fig.colorbar(im, cax=cax, label="Phase score")
        ax.set_title(f"{dict_ref[ref]['chemical formula']}")
        ax.set_xlabel(f"x-coordinate / {unit}")
        ax.set_ylabel(f"y-coordinate / {unit}")
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(FuncFormatter(custom_ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(custom_ticks))
        plt.tight_layout()
        Save_Figure(f'Phase mapping {ref}', 'y')
        plt.show()

# ====================
# End