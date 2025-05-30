
## What is in this folder?

The main objectives of these codes are to monitor Marine Heat Waves. The main three steps done here are : downloading required data, computing the metrics needed and viewing the results.

## What is in the subfolders?

 - data_fetchers/ :
        Codes that downloads and manage input data (Satellite SST, Satellite climatology, bathymetry...).

 - mhws_computers/ :
        Codes that computes different metrics for MHWs using input data, resulting in output data.

 - data_plotters/ :
        Codes that display the input or output data.

 - utils/ :
        Codes that helps for other codes, helping with plotting for example.

## How should this code be runned?

Make sure that the *Datos* folder is at the right place (see External files section) for the codes to run smoothly.

Normally, you only need to run any script (except the utils) using python3 and the required python packages installed (see *requirements.txt*).
These scripts haven't been tested under Windows. Even if some precautions were taken, some errors may occur regarding file paths.

If issues occur with Python interpreter or packages, please run *utils/check_packages_version.py* to check versions.

## External files

For most of the code here, external files are expected, namely the data files. The expected file tree should be as follow:

Root folder
 ├── Codigos/ (where this README is)
 ├── Datos/
 └── Figuras/

## References

These codes have been developped by Arthur Gonnet.

It uses the *marineHeatWaves* module for python developped by Eric C. J. Oliver
