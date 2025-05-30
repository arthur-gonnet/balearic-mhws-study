
## What is in this folder?

All the data file present in this folder are meant to help investigate Marine Heat Waves in the Mediterranean Sea, and especially in the Balearic Islands.

## What is in the subfolders?

 - **argo/** :
        Argo data, primarly used to verify the models (MED-MFC and WMOP).

 - **bathymetry/** :
        Bathymetry files around Balearic Islands from MED-MFC model or GEBCO model.

 - **CMEMS-MFC/** :
        MED-MFC temperature data from 1987 to 2022 (Copernicus Marine Data ID: *MEDSEA_MULTIYEAR_PHY_006_004*).

 - **CMEMS-SST/** :
        Reprocessed Mediterranean SST from 1982 to 2023 (Copernicus Marine Data ID: *SST_MED_SST_L4_REP_OBSERVATIONS_010_021*).

 - **puerto_del_estado/** :
        Temperature data from buoys accessed from Puerto del Estado (https://portus.puertos.es/).

 - **marineinsitu-SST/** :
        Temperature data from marineinsitu website (https://marineinsitu.eu/dashboard/).

## How should this code be runned?

Normally, you only need to run any script (except the utils) using python3 and the required python packages installed (see *requirements.txt*).
These scripts haven't been tested under Windows. Even if some precautions were taken, some errors may occur regarding file paths.

If issues occur with Python interpreter or packages, please run *utils/check_packages_version.py* to check if you're Python environment is up to date.