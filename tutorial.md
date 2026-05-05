# Tutorial for RaMap

This tutorial is intended to guide the application of RaMap. It provides a step-by-step instruction and demonstrates its use through two example Raman mapping datasets.

At the beginning, it should be checked whether all rerquired python packages are installed. Please refer to the [readme](README.md#Requirements) file for details.

## 1 Run the script

## 2 Import the Raman data
### Load Raman Mapping File

After executing the script [RaMap.py](RaMap.py), a file explorer window will open.
Use this window to select the Raman mapping dataset.

<p align="center">
  <img src="docs/RaMap_tutorial/explorer_load_mapping.jpg" width="600"/><br>
  <em>Figure 1: File selection window for Raman mapping data.</em>
</p>

Further information regarding supported file formats and required data structures can be found in the [documentation](documentation.md#input-data-format-raman-mapping).

### Load Raman Reference Spectra

Once the mapping dataset has been selected, a second window will appear for importing reference spectra.
In this step, select and open all reference files.

<p align="center">
  <img src="docs/RaMap_tutorial/explorer_load_references.jpg" width="600"/><br>
  <em>Figure 2: File selection window for reference Raman spectra.</em>
</p>

Details on supported file formats are also provided in the [documentation](documentation.md#input-data-format-raman-spectra-for-reference-materials).


After the reference data have been loaded, they can be assigned names in the terminal:
```
File: path-to/Tutorial_Raman_data/References/Cr2O3.txt
Compound name: chromiumoxide
```


After confirming with ENTER, the chemical formula can be entered next. For this purpose, LaTeX notation should be used.
```
File: path-to/Tutorial_Raman_data/References/Cr2O3.txt
Compound name: chromiumoxide
Chemical formula: \alpha -Cr_2O_3
```


After confirming again with ENTER, a color can be assigned to this compound. Both hexadecimal color codes and named colors can be used.

```
File: path-to/Tutorial_Raman_data/References/Cr2O3.txt
Compound name: chromiumoxide
Chemical formula: \alpha -Cr_2O_3
Plot color (press Enter for system color): #D500D5
```

Pressing ENTER without specifying a color assigns a default system color. By default, the Set1 color palette is used.

