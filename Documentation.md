# Documentation for RaMap
## Raman data
### Input Data Format Raman mapping

The Raman mapping data should be provided in `.csv` or `.txt` format.  
The expected data structure is as follows:

1. **First column:** x-coordinate  
2. **Second column:** y-coordinate  
3. **First row:** Wavenumbers  
4. **Subsequent rows:** Raman intensities corresponding to each (x, y) position

Make sure that the file follows this structure so that the script can correctly read and process the mapping.

### Input Data Format Raman spectra for Reference Materials

For the analysis of the Raman mapping data, reference spectra from reference materials are required and should be provided in `.csv` or `.txt` format.  
These reference spectra are used to align and compare the measured mapping spectra and are essential for statistical analyses.
The expected data structure is as follows:

1. **First column:** Wavenumbers 
2. **Second column:** Raman intensities  

## Usage
### Getting started
1. Place your Raman `.csv` or `.txt` files in the `data/` directory before running the workflow.
2. Add necessary components to the `RaMap.py` script 

	a) Path to the Raman mapping file

		   map_file = data/Raman_mapping.<ext>
  
	
	b) Paths to the reference spectra and assigning names

		   ref1_path = data/reference_spectrum1.<ext>
		   ref2_path = data/reference_spectrum2.<ext>


     You can add more references as needed.

	c) The references are stored in a dictionary. Each entry corresponds to a reference and contains the following fields:
      ```python
			dict_ref = {
        "ref1":{
        "ref_shifts":load_ref(ref1_path)[0],   
        "ref_spectrum":load_ref(ref1_path)[1],
        "plot color":"",
        "chemical formula":""
        },
				"ref2":{ ... },
					...
			} 
      ```
      
      How to use:
   
      1. Each reference in the dictionary should be named according to the material it represents, rather than using generic keys like `ref1`.  
      2. Update "plot color" with a color string (e.g., "red") for plotting.
      3. Optionally, fill "chemical formula" with the chemical formula of the phase.
      4. Add additional references by creating new keys ("ref2", "ref3", …) following the same format.
   
      The corresponding Raman spectrum from the directory `ref1_path` is stored in:
  	  - **ref_shifts**: Wavenumbers  
      - **ref_spectrum**: Raman intensities
        
      These values are automatically loaded when the script runs.

    d) Path to save the result images

		   path_results = data/
       

	3. Run the script

		   python3 RaMap.py

### User inputs and outputs

  ```
     Do you want to save intermediate results as .png images? (y/n):
   ```

   ```y```: The workflow generates and saves multiple intermediate results. These outputs help to:
   - Improve the traceability of the workflow
   - Visualize potential sources of errors during phase assignment and analysis
   
   Examples of intermediate results include:
   - Baseline-corrected spectra for each reference material
   - Baseline-corrected spectra of a specified mapping position (see below)
   - Scatter plots of Signal-to-Noise ratio per mapping spectrum and highlighting excluded mapping spectra due to a low Signal-to-noise ratio (SNR) or intensive photoluminescence background (see below)
   - Statistical analysis outputs (e.g. NMF components, Cosine similarity) (see below)
  
   All intermediate results are saved to the folder specified in ```path_results```.

   ```n```: These outputs will not be shown or saved. The workflow directly displays the final phase mappings saved as _Phase Mapping "".png_.
     Output:

							Saved Phase mapping "".png

 &nbsp;
 
  ```
    Which wavenumber range should be considered? Please enter the values in the format: minimum,maximum
  ```
    
  The script asks for the spectral range of the Raman spectrum.
  - Enter the range in the format: `minimum_wavenumber,maximum_wavenumber`
  - Values should be provided as `floats`
  - The units are $\mathrm{cm}^{-1}$  

&nbsp;

  ```
    Please enter a value for the PL threshold. (Set 0 if no PL should be considered):
  ```
  The script asks for a threshold to exclude spectra in the Raman mapping with intensive PL signals which influence the subsequent evaluation.
  It gives the intensity ratio between the maximum Raman peak (baseline-corrected spectrum) and the PL background.
  There is no universally valid value for the PL (photoluminescence) threshold, so this parameter should be tested and adjusted for your specific dataset.
  The value is usually above 0.8.
  If no PL should be considered, set the value to 0.  

  &nbsp;

  ```
    Should a sample spectrum from the mapping be plotted? If yes, please enter the coordinates in the following format: x,y. If not, please type None.
  ```
  ```None```: Skipping sample spectrum plot. 
  
  ```x_value,y_value ```: The positions are temporarily stored, and a sample spectrum from these positions is plotted after the request for the unit of the mapping. Saved as _Sample spectrum.png_. 
         Output:

    							Saved Sample spectrum.png

&nbsp;

  ```
    Please enter the units of the mapping. (Default is micrometer: $\mu$m). Press Enter to use the default:
  ```
  The script asks for the units of the spatial resolution of the Raman mapping. The `default` is micrometer (µm), which can be accepted by simply pressing _Enter_. Other possible units include `cm`, `mm`, or `nm`, depending on the units in which the mapping position data are stored.  

&nbsp;

  ```
    Please enter a value for the SNR threshold:
  ```
  The script asks for a threshold value for the Signal-to-Noise Ratio (SNR).
  - All spectra with SNR below this threshold will be excluded from the analysis.
  - There is no universally valid value for the SNR threshold, so this parameter should be tested and adjusted for your specific dataset.
  - A threshold value of `3` has produced good results in previous analyses.
  - These low-SNR spectra are considered too noisy and could degrade the quality of the analysis. They are exported as _Excluded_spaectra.txt_. Output:
     
               There are spectra with SNR < threshold or with intensive PL.
               The corresponding spectra ("") will therefore be excluded from the following analysis.
               Spectra saved as 'Excluded_spectra.txt'.

  An image ```SNR_map.png``` is generated, showing the SNR per measured Raman spectrum of the mapping. Output:

    							Saved SNR_map.png
			
  - Positions where the Raman spectrum is either too noisy or has strong PL and is therefore excluded from the analysis are marked in red.
  - This visualization helps to quickly identify which areas of the mapping were not considered in the final analysis.

#### Outputs
```
  Saved NMF_number_of_components.png
```
            
  A figure is generated that is important for the Non-negative Matrix Factorization (NMF) decomposition.
  - The plot shows the explained variance as a function of the number of NMF components.
  - The optimal number of components is highlighted in red.
  - This helps to determine the appropriate component number for the NMF analysis.
  - The figure is saved as `NMF_number_of_components.png`.

&nbsp;

```
    Saved NMF_component_""_map.png
```
    
  The generated NMF component spectra are now displayed along with their spatially distributed weighting factors.
  - The number of components is based on the previously determined optimal number.
  - This visualization shows how each NMF component contributes to the Raman mapping across the sample area.
  - The NMF component spectra are assigned to real phases based on the reference spectra (for example ```ref1```) stored in ```dict_ref```.
  - The figure is saved as `NMF_component_""_map.png`.

&nbsp;

```
    Saved NMF_Mean_Squared_Error.png
```

  The mean squared error (MSE) between the NMF model and the measured Raman mapping spectra is calculated and displayed for each measurement position.
  - This provides a spatial view of the fit quality of the NMF model across the sample.
  - Higher MSE values indicate positions where the model does not accurately represent the measured spectra.
  - The figure is saved as `NMF_Mean_Squared_Error.png`.
    
&nbsp;

```
    Saved Cosine_similarity_map_"".png
```

  The cosine similarity between each measured spectrum and each reference spectrum "" in `dict_ref` is calculated and displayed.
  - This allows assessment of how closely each measurement matches the reference spectra.
  - High cosine similarity values indicate a strong resemblance to a particular reference material.
  - The similarity value per measurement point is saved as `Cosine_similarity_map_"".png`.

&nbsp;

```
  Saved Cosine_similarity_map_best_reference.png
```

  The reference phases are now displayed for each measurement point based on which reference spectrum has the highest cosine similarity.
  - This visualization shows the most likely phase present at each mapping position according to the results of cosine similarity.
  - The figure is saved as `Cosine_similarity_map_best_reference.png`.

&nbsp;

```
  Saved Combined_score_map_"".png
```

  The combined score for each clearly assignable phase "" is now plotted as a spatial distribution and saved as `Combined_score_map_"".png`.  

&nbsp;

```
  Saved Phase mapping "".png"
```

  The constructed phase mappings are now displayed with a phase score ranging from 0 to 1.
  - The phase score can be interpreted as the probability of presence of each phase "" at a given mapping position.
  - This visualization provides an intuitive overview of phase distribution and confidence across the sample.


#### Error Messages

```
  Which wavenumber range should be considered? Please enter the values in the format: minimum,maximum
```
  ```ValueError: The specified limits are outside the range of the Raman spectrum.```
  
  If the entered values for the lower and upper limits are outside the wavenumber range of the given Raman spectra, an error will occur.  
  Please adjust the incorrect value(s) and ensure that both limits lie within the valid spectral range before running the analysis again.

  ```ValueError: The lower limit is equal to or greater than the upper limit.```
  
  The lower limit is equal to or greater than the upper limit.  
	Please check your input and ensure that the first value (lower limit) is smaller than the second value (upper limit).

&nbsp;




```
  Please enter a value for the PL threshold. (Set 0 if no PL should be considered):
```
  ```ValueError: Please enter a valide PL threshold.```
  
  The entered value is not a valid PL threshold.
  Please provide a valid positive value in [0,1] (0 if PL should not be considered).

&nbsp;

```
  Please enter a value for the SNR threshold:
```
  ```ValueError: Please enter a valide SNR threshold.```
  The entered SNR threshold is not valid.
  It must be a positive number greater than zero.  
  Please provide a valid SNR threshold.

&nbsp;

```
Should a sample spectrum from the mapping be plotted? If yes, please enter the coordinates in the following format: x,y. If not, please type None.
```
  ```Plotting of sample spectrum skipped. Coordinates not found in data.```
  The specified coordinates for the sample spectrum do not exist in the mapping data.  
  Please check the input coordinates and ensure they correspond to positions within the Raman mapping.

&nbsp;

```There are spectra with SNR<threshold or with intensive PL. The corresponding spectra ("") will therefore be excluded from the following analysis.```

There are spectra that either have an overly intense PL (photoluminescence) background or a low signal-to-noise ratio (SNR).
- These spectra can interfere with the analysis and will therefore be excluded.
- Spectra saved as _Excluded_spectra.txt_ with the following format:
  - **First row:** x-position
  - **Second row:** y-position
  - **First column:** Wavenumber
  - **Following columns:** Raman intensities of the excluded spectra
  
&nbsp;

```There is inconsistency between the NMF decomposition and the cosine similarity results. The corresponding spectra ("") will therefore be excluded from the following analysis. Spectra saved as 'Inconsistency_spectra.txt'.```

There was an inconsistent assignment of Raman spectra between the cosine similarity and the NMF decomposition. 
- Spectra for which the highest cosine similarity was assigned to a reference phase, but this phase was not observed in the NMF decomposition, are considered inconsistent.  
- These spectra are excluded from further analysis, as a clear phase assignment could not be made using both methods.
- The corresponding spectra are saved as _Inconsistency_spectra.txt_ with the following format:
  - **First row:** x-position
  - **Second row:** y-position
  - **First column:** Wavenumber
  - **Following columns:** Raman intensities of the inconsistent spectra

&nbsp;
	
```The file "" already exists. Do you want to overwrite it? (y/n):```

The file already exists in the specified results folder. Please confirm if you want to overwrite the existing file.

`y` The existing file "" will be overwritten with the new one. Output: 

    							File "" has been overwritten.

  `n` The overwrite operation has been skipped. The existing file remains unchanged. Output: 
  
    							File save has been canceled. No changes made.
                  
&nbsp;

```FileNotFoundError: File "" was not found.```
The specified Raman mapping file or reference file could not be found at the given path.  
Please check the file path and ensure the file exists.

&nbsp;

```Exception: Error when reading as <ext>: ""```
The file structure does not match the expected format, or the content is corrupted.  
Please check the file "" and ensure it meets the required specifications.

&nbsp;

```ValueError: Unknown or unsupported file type: <ext> ```
An attempt was made to load a file that does not match the expected file type (`.txt` or `.csv`).  
Please check whether this is the correct file and verify the filename and extension for any spelling errors.

## Tips
- typical values for PL threshold: 0.8 < $\mathrm{R}_{pl}$ < 1
- typical values for SNR threshold: 2 < $\mathrm{SNR}_{th}$ < 5

## License
This project is licensed under the Apache License 2.0 
