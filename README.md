# RaMap
Workflow for constructing phase maps from Raman spectroscopy data.
 Author: Felix Drechsler (Felix.Drechsler@physik.tu-freiberg.de)
 Language: Python

For detailed usage instructions, see the [documentation](documentation.md).
Examples are available in the [tutorial](docs/tutorial.md).
```
## Requirements
Python >= 3.11 
 
_Package requirements are listed in 'requirements.txt'_

The required packages can be installed by
```sh
pip install -r src/requirements.txt
```

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
  	  - **ref_shifts**: wavenumbers  
      - **ref_spectrum**: Raman intensities
        
      These values are automatically loaded when the script runs.

    d) Path to save the result images

		   path_results = data/
       

	3. Run the script

		   python3 RaMap.py

## License
This project is licensed under the Apache License 2.0 



