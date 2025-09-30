# Data folder

The data here is meant to be used to compute MHW metrics in the Mediterranean Sea. Most of the data used has been provided by the Copernicus Marine Service.

## What is in here?

 - `bathymetry/` **(Input)** :
     Bathymetry files, from MEDREA or GEBCO.

 - `REP/` **(Input)** :
     Reprocessed satellite-derived SST over the Mediterranean Sea from 1982 to 2023 provided by Copernicus Marine Service (SST_MED_SST_L4_REP_OBSERVATIONS_010_021).

 - `MEDREA/` **(Input)** :
     Physical reanalysis over the Mediterranean Sea from 1987 to 2022 provided by Copernicus Marine Service (MEDSEA_MULTIYEAR_PHY_006_004).

 - `mhws/` **(Output)** :
     NetCDF files containing pre-computed MHW metrics, to be used to produce figures.

*Please note that this code can also be adapted to work with other data.*

## How to download external data?

In the GitHub repository, these folders are empty. To get the data used in the study, please run `Code/02_data_download.ipynb`. The datasets can also be downloaded manually.

Detailed information for downloading each dataset can be found in the `README.md` within its respective folder.

## Data folders structure

In order to use the code without any modifications, the Data folder must be provided using the original folder structure. If the data structure is modified, please modify the `Code/pyscripts/load_save_dataset.py` folder paths.

The expected file tree inside the Data folder should be as follow:

```
Data/
├── bathymetry/
│   ├── Bathymetry_GEBCO_2023_IBERIAN.nc
│   └── Bathymetry_MEDREA_006_004_mask_bathy_BalearicIslands.nc
│
├── REP/
│   └── MEDITERRANEAN/SST-L4-REP-HR/DATA-NEW/DAILY/
│       ├── 1982/
│       │   ├── SST_MED_SST_L4_REP_OBSERVATIONS_010_021_y1982m01.nc
│       │   └── ...
│       ├── 1983/
│       └── ...
│
├── MEDREA/
│   └── MEDITERRANEAN/REANALYSIS/DATA/DAILY/BalearicIslands/
│       ├── 1987/
│       │   ├── TEMP_MEDSEA_MULTIYEAR_PHY_006_004_y1987m01_BalearicIslands.nc
│       │   └── ...
│       ├── 1988/
│       └── ...
│
└── mhws/
    ├── yearly/
    │   ├── rep_mhws_balears_1987_2021.nc
    │   └── medrea_mhws_balears_1987_2021.nc
    │
    └── all_events/
        ├── rep_mean_mhws_balears_1987_2021.nc
        └── medrea_mean_mhws_balears_1987_2021.nc
```

## License

This work makes use of E.U. Copernicus Marine Service Information; https://doi.org/10.48670/moi-00173; https://doi.org/10.25423/CMCC/MEDSEA_MULTIYEAR_PHY_006_004_E3R1, under a permissive license.
