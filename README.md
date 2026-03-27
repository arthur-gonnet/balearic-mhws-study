# <h1 style="text-align: center;"> Balearic MHWs study </h1>

This repository contains all the code used to produce figures for the master thesis entitled “Marine heatwaves in the Balearic Islands region”.

## Overview

This code is meant to compute and visualise marine heatwaves (MHWs) in the Balearic Islands region. It is intended to provide a reproducible, modular and extensible workflow for:

- Downloading and handling large oceanographic datasets
- Computing MHW metrics using Hobday et al. (2016) definition of MHWs
- Dask’s chunking, vectorisation, and parallelisation for efficient and adaptive computing
- Generating figures for reports and presentations

## What is in here?

 - `Code/` :
    Python notebooks and scripts that compute and visualise MHWs metrics.

 - `Data/` :
    Holds downloaded CMEMS datasets and computed MHWs datasets.

 - `Documents/` :
    Folder containing the original master thesis report and presentation using the code herein.

## How should this code be runned?

See the `README.md` inside the Code folder for more details. 

## External data

For most of the code here, external data is expected. Due to the large size of those files, they are not included in the GitHub repository.

See the `README.md` inside the Data folder for more details.

## License

This code has been developed by Arthur Gonnet, and is licensed under the GNU General Public License v3.0 (GPLv3).

This code includes a modified version of the *marineHeatWaves* module for python developed by Eric C. J. Oliver (see https://github.com/ecjoliver/marineHeatWaves), under GPLv3 license.

This code uses the *pyMannKendall* module for python developed by Md. Manjurul Hussain and Ishtiak Mahmud (see https://github.com/mmhs013/pyMannKendall), under MIT license.

This work makes use of E.U. Copernicus Marine Service Information; https://doi.org/10.48670/moi-00173; https://doi.org/10.25423/CMCC/MEDSEA_MULTIYEAR_PHY_006_004_E3R1, under a permissive license.

## Contact

> Arthur Gonnet <br>
> br.arthur.gonnet@gmail.com <br>
> https://github.com/arthur-gonnet/
