# MHW datasets

This folder holds the computed MHW datasets derived from the REP and the MEDREA datasets. To obtain these datasets, the two options are to compute it from REP and MEDREA datasets or to contact the author.

## How to compute?

Follow the procedure described in `Code/README.md`, running the Python notebooks sequentially.

## Structure

Make sure that the data is structured under the following data structure to be used directly with the code provided here. If the data structure is modified, please modify the `Code/pyscripts/load_save_dataset.py` folder paths.

```
mhws/
 ├─ yearly/
 │  ├─ rep_mhws_balears_1987_2021.nc
 │  └─ medrea_mhws_balears_1987_2021.nc
 │
 └─ all_events/
    ├─ rep_mean_mhws_balears_1987_2021.nc
    └─ medrea_mean_mhws_balears_1987_2021.nc
```
