########################################################################################################################
########################################################################################################################

###### USER NOTES:
# This script gather all the codes to load and save datasets using xarray. 

## Functions description
#
# - load_bathy(...):
#       Loads a bathymetry dataset using xarray (either GEBCO or MEDREA).
#
# - load_rep(...):
#       Loads the REP dataset using xarray.
#
# - load_medrea(...):
#       Loads the MEDREA dataset using xarray.
#
# - save_mhws_dataset(...):
#       Saves a MHW dataset using xarray.
#
# - load_mhws_dataset(...):
#       Loads a MHW dataset using xarray.
#
# - save_dataset_to_nc(...):
#       Basic function to save a dataset using xarray.
#
# - load_nc_to_dataset(...):
#       Basic function to load a dataset using xarray.

########################################################################################################################
##################################### IMPORTS ##########################################################################
########################################################################################################################

# Basic imports
import os
import glob as glob

# Advanced imports
import xarray as xr
from dask.diagnostics import ProgressBar, ResourceProfiler, Profiler, visualize

# Local imports
from pyscripts.process_data import detrend_dataset

# Load relative file paths
# These locations must be accurate to load the data
codigos_location = os.path.abspath(os.getcwd())
datos_location = os.path.join(codigos_location, '..', 'Datos')

########################################################################################################################
##################################### USER INPUT #######################################################################
########################################################################################################################


# USER INPUT - Bathy dataset path
bathy_GEBCO_path       = os.path.join(datos_location, 'bathymetry', 'Bathymetry_GEBCO_2023_IBERIAN.nc')
bathy_MEDREA_path      = os.path.join(datos_location, 'bathymetry', 'Bathymetry_MEDREA_006_004_mask_bathy_BalearicIslands.nc')
bathy_MEDREA_med_path  = os.path.join(datos_location, 'bathymetry', 'Bathymetry_MEDREA_006_004_mask_bathy.nc')

# USER INPUT - Dragonera insitu dataset path
dragonera_insitu_path           = os.path.join(datos_location, 'puerto_del_estado', 'dragonera_hist_temp.nc')
dragonera_insitu_marta_s_path   = os.path.join(datos_location, 'puerto_del_estado', 'dragonera_hist_temp_marta_s.nc')

# USER INPUT - REP paths
REP_folder_path   = os.path.join(datos_location, 'REP', 'MEDITERRANEAN', 'SST-L4-REP-HR', 'DATA-NEW', 'DAILY')
REP_ds_pattern    = os.path.join(REP_folder_path, '{year}', 'SST_MED_SST_L4_REP_OBSERVATIONS_010_021_y{year}m{month}.nc')

# USER INPUT - MEDREA pathes
MEDREA_folder_path   = os.path.join(datos_location, 'MEDREA', 'MEDITERRANEAN', 'REANALYSIS', 'DATA', 'DAILY', 'BalearicIslands')
MEDREA_ds_pattern    = os.path.join(MEDREA_folder_path, '{year}', 'TEMP_MEDSEA_MULTIYEAR_PHY_006_004_y{year}m{month}_BalearicIslands.nc')

# USER INPUT - MHWs datasets pathes
mhws_dataset_pattern = os.path.join(datos_location, 'mhws', '{type}', '{dataset}_mhws_{region}{detrended}_{clim_start}_{clim_end}.nc')


########################################################################################################################
##################################### CODES ############################################################################
########################################################################################################################


def load_bathy(source: str = 'MEDREA', drop_vars = 'all', region_selector: str | None = 'balears') -> xr.Dataset:
    """
    Loads a bathymetry dataset using xarray (either GEBCO or MEDREA).

    Parameters
    ----------
    source: str, default='MEDREA'
        Source of the bathymetry dataset to load. Can be `'GEBCO'`, `'MEDREA'` or `'MEDREA_MED'`.
    
    Returns
    ----------
    ds_bathy: xr.Dataset
        The loaded bathymetry dataset.
    """

    if source.lower() == 'gebco':
        # Loading the dataset
        ds_bathy_GEBCO = load_nc_to_dataset(
            bathy_GEBCO_path,
            name_dict = {
                'elevation': 'depth'
            },
            remove_bay_of_biscay = True,
            region_selector = region_selector
        )

        # Adjusting the depth parameter, name and value
        # The expected depth parameter should be positive at depth
        ds_bathy_GEBCO = ds_bathy_GEBCO.where(ds_bathy_GEBCO['depth'] < 10)
        ds_bathy_GEBCO['depth'] = -ds_bathy_GEBCO['depth']

        # Printing the good news
        print("Loaded GEBCO bathymetry dataset.")

        # Returning the final dataset
        return ds_bathy_GEBCO
    
    elif source.lower() == 'medrea' or source.lower() == 'medrea_med':
        # Loading the dataset
        ds_bathy_MEDREA = load_nc_to_dataset(
            bathy_MEDREA_path if source.lower() == 'medrea' else bathy_MEDREA_med_path,
            name_dict = {
                'longitude': 'lon',
                'latitude': 'lat',
            },
            remove_bay_of_biscay = False,
            region_selector = region_selector,
            drop_variables = ['deptho_lev', 'mask'] if drop_vars == 'all' else drop_vars
        )

        # Adjusting the parameters, name and value
        ds_bathy_MEDREA = ds_bathy_MEDREA.drop_dims('depth')
        ds_bathy_MEDREA = ds_bathy_MEDREA.rename({
            'deptho': 'depth'
        })

        # Printing the good news
        print(f"Loaded MEDREA bathymetry dataset{' (all med)' if source.lower() == 'medrea_med' else ''}.")

        # Returning the final dataset
        return ds_bathy_MEDREA
    
    else:
        print(f"Bathymetry dataset not recognized ({source}), please choose 'GEBCO', 'MEDREA' or 'MEDREA_MED'.")



def load_rep(
        # Time preselection
        years: list[int] = range(1982, 2024),
        months: list[int] = range(1, 13),

        # Selectors
        time_selector: str | slice | None = None,
        lon_selector: float | slice | None = None,
        lat_selector: float | slice | None = None,
        region_selector: str | None = 'balears',
        
        # Dataset options
        chunks: int | dict | str | None = 'auto',
        only_sst: bool = True,
) -> xr.Dataset:
    """
    Loads the REP dataset using xarray. By default, the entire dataset is loaded,
    but optional spatio-temporal subsettings are available for faster loading times.

    Parameters
    ----------
    years: list[int], default=range(1982, 2024)
        Years to load from the dataset, as integers (e.g., `range(1983, 1985)` or `[1983]`).
        Use `['*']` to load all available years (1982-2023) as it uses a glob pattern.
        This parameter defines the files to load, so loading time is highly depends on this range size.
    
    months: list[int | str], default=range(1, 13)
        Months to load from the dataset, as integers (e.g., `range(1, 5)` or `[1]`).
        Use `['*']` to load all months (1-12) as it uses a glob pattern.
        This parameter defines the files to load, so loading time is highly depends on this range size.
    
    time_selector: str | slice[str], optional
        Time selection applied using xarray's `.sel()`. Can be a string (e.g., `'1993-01-21'`) or a slice
        (e.g., `slice('1993-01-21', '1993-01-25')`). If `None`, no time filtering is applied.
    
    lon_selector: float | slice[float], optional
        Longitude selector applied using xarray's `.sel()`. Accepts a float or a slice.
    
    lat_selector: float | slice[float], optional
        Latitude selector applied using xarray's `.sel()`. Accepts a float or a slice.
    
    region_selector: str, default='balears', optional
        Applies a spatial selector to the dataset. Overrides `lon_selector` and `lat_selector`.
        For now, the only choice is `'balears'`.
    
    chunks: int | dict | 'auto', default='auto', optional
        xarray parameter.
        Dictionary with keys given by dimension names and values given by chunk sizes.
        In general, these should divide the dimensions of each dataset. If int, chunk
        each dimension by ``chunks``. By default, chunks will be chosen to load entire
        input files into memory at once. This has a major impact on performance.
    
    only_sst: bool, default=True
        If `True`, all other variables than SST will be discarded.
    
    Returns
    ----------
    ds_rep: xr.Dataset
        The REP dataset.
    """

    # Using temporal preselection to choose which files to load
    files: list[str] = []

    for year in years:
        if year != '*' and (int(year) < 1982 or int(year) > 2023):
            print(f"Incorrect year selection for REP dataset, got {years}, expected a year range between 1982 and 2023")
        
        for month in months:
            if month != '*' and (int(month) < 1 or int(month) > 12):
                print(f"Incorrect month selection for REP dataset, got {months}, expected a month range between 1 and 12")
            
            month_str: str = str(month) if (month == '*' or month > 9) else ('0'+str(month))
            pattern: str = REP_ds_pattern.format(year=year, month=month_str)
            files.extend(glob.glob(pattern))
    
    # Load region selection
    if region_selector == 'balears':
        lon_selector = slice(-0.9, 5.1)
        lat_selector = slice(37.6, 41.1)

    # Preprocess applied before concatenating the datasets
    def preprocess(ds: xr.Dataset) -> xr.Dataset:
        """
        The preprocess applies dataset operations before concatenating the multiple datasets.
        Here, performing spatial selection.
        """
        
        # Remove bay of Biscay
        ds = ds.where(((ds.lon > 0) | (ds.lat < 42)), drop=True)

        # Spatial selection
        if not lon_selector is None:
            # Depending if lon_selector is slice or float, use different method
            ds = ds.sel(lon=lon_selector, method=(None if type(lon_selector) == slice else 'nearest'))
        
        if not lat_selector is None:
            ds = ds.sel(lat=lat_selector, method=(None if type(lat_selector) == slice else 'nearest'))
        
        return ds

    # Options for loading the dataset
    drop_vars = ['analysis_error', 'mask', 'sea_ice_fraction'] if only_sst else None


    # LOADING THE DATASET !
    ds_rep = xr.open_mfdataset(
        paths = files,
        preprocess = preprocess,
        drop_variables = drop_vars,
        decode_times = True,
        chunks = chunks,
    )


    # Applying time selection (some issues occur when selecting in preprocess)
    if not time_selector is None:
        ds_rep = ds_rep.sel(time=time_selector)
    
    # Uniformying variables name to T for temperature
    ds_rep = ds_rep.rename({'analysed_sst': 'T'})

    # Changing unit from K to °C
    ds_rep['T'] = ds_rep.T - 273.15
    ds_rep.T.attrs['unit'] = '°C'

    # Printing the good news
    print(f"Loaded REP dataset.")
    
    # Returning the final dataset
    return ds_rep



def load_medrea(
        years: list[int|str] = range(1987, 2023),
        months: list[int|str] = range(1, 13),

        time_selector: str|slice|None = None,
        lon_selector: float|slice|None = None,
        lat_selector: float|slice|None = None,
        depth_selector: float|slice|None = slice(0, 3000), # Should not remove any data in the study region
        region_selector: str | None = 'balears',

        move_to_0am: bool = True,
        only_botT: bool = False,
        detrended: bool = False,

        drop_vars = [],
        
        chunks: int | dict | str | None = 'auto',
) -> xr.Dataset:
    """
    Loads the MEDREA dataset using xarray. By default, the entire dataset is loaded, but optional spatio-temporal subsettings are available
    for faster loading times.

    Parameters
    ----------
    years: list[int|str], default=['*']
        Years to load from the dataset, as integers or strings (e.g., range(1993, 1995) or ['1993']).
        Use ['*'] to load all available years (1987-2022) as it uses a glob pattern.
    
    months: list[int|str], default=['*']
        Months to load from the dataset, as integers or strings (e.g., range(1, 5) or ['1']).
        Use ['*'] to load all months (1-12) as it uses a glob pattern.
    
    time_selector: str | slice[str], optional
        Time selection applied using xarray's `.sel()`. Can be a string (e.g., '1993-01-21') or a slice
        (e.g., slice('1993-01-21', '1993-01-25')). If None, no time filtering is applied.
    
    lon_selector: float | slice[float], optional
        Longitude selector applied using xarray's `.sel()`. Accepts a float or a slice. If None, no longitude filtering is applied.
    
    lat_selector: float | slice[float], optional
        Latitude selector applied using xarray's `.sel()`. Accepts a float or a slice. If None, no latitude filtering is applied.
    
    depth_selector: float | slice[float], default=slice(0, 3000), optional
        Depth selector applied using xarray's `.sel()`. Accepts a float or a slice. If None, no depth filtering is applied.
    
    region_selector: str, default='balears', optional
        Applies a spatial selector to the dataset. Overrides `lon_selector` and `lat_selector`.
        For now, the only choice is `'balears'`.
    
    chunks: int | dict | 'auto', default='auto', optional
        xarray parameter.
        Dictionary with keys given by dimension names and values given by chunk sizes.
        In general, these should divide the dimensions of each dataset. If int, chunk
        each dimension by ``chunks``. By default, chunks will be chosen to load entire
        input files into memory at once. This has a major impact on performance:.
    
    Returns
    ----------
    ds_medrea: xr.Dataset
        The loaded MEDREA dataset with optional spatio-temporal subsetting.
    """

    # Selecting the files to load
    files: list[str] = []

    for year in years:
        if year != '*' and (int(year) < 1982 or int(year) > 2023):
            print(f"Incorrect year selection for MEDREA dataset, got {years}, expected a year range between 1982 and 2023")
        
        for month in months:
            if month != '*' and (int(month) < 1 or int(month) > 12):
                print(f"Incorrect month selection for MEDREA dataset, got {months}, expected a month range between 1 and 12")
            
            month_str: str = str(month) if month == '*' or month > 9 else '0'+str(month)
            pattern: str = MEDREA_ds_pattern.format(year=year, month=month_str)
            files.extend(glob.glob(pattern))
    
    # Load region selection
    if region_selector == "balears":
        lon_selector = slice(-0.9, 5.1)
        lat_selector = slice(37.6, 41.1)
    
    # Preprocess applied before concatenating the datasets
    def preprocess(ds: xr.Dataset) -> xr.Dataset:
        """ This preprocess selects the data while loading, which makes an efficient loading """
        
        # Remove bay of Biscay
        if not region_selector == 'balears':
            ds = ds.where(((ds.lon > 0) | (ds.lat < 42)), drop=True)
        
        # Spatial selection
        if not lon_selector is None:
            # Depending if lon_selector is slice or float, use different method
            ds = ds.sel(lon=lon_selector, method=(None if type(lon_selector) == slice else 'nearest'))
        
        if not lat_selector is None:
            ds = ds.sel(lat=lat_selector, method=(None if type(lat_selector) == slice else 'nearest'))
        
        if not depth_selector is None and not only_botT:
            ds = ds.sel(depth=depth_selector, method=(None if type(depth_selector) == slice else 'nearest'))
        
        return ds

    # Acquiring only bottom temperature, those don't matter
    if only_botT:
        drop_vars = ['thetao', 'depth']


    # LOADING THE DATASET !
    ds_medrea = xr.open_mfdataset(
        files,
        preprocess=preprocess,
        drop_variables=drop_vars,
        decode_times=True,
        chunks=chunks,
    )

    # Applying time selection (some issues occur when selecting in preprocess)
    if not time_selector is None:
        ds_medrea = ds_medrea.sel(time=time_selector)
    
    # Resampling if necessary
    if move_to_0am:
        ds_medrea['time'] = ds_medrea.time.dt.floor('1D')

    if only_botT:
        # Compute the depth
        ...
    
    # Uniformying variables name to T for temperature
    if not 'thetao' in drop_vars:
        ds_medrea = ds_medrea.rename({'thetao': 'T'})

    # Setting unit to °C
    if not 'thetao' in drop_vars:
        ds_medrea.T.attrs['unit'] = '°C'

    # Detrend dataset if demanded
    if detrended:
        ds_medrea = detrend_dataset(ds_medrea)

    # Printing the good news
    print("Loaded MEDREA dataset.")

    # Returning the final dataset
    return ds_medrea



def save_mhws_dataset(
        # Dataset
        ds_mhws: xr.Dataset,

        # File path parameters
        ds_type: str,
        dataset_used: str,
        detrended: bool = False,
        region: str = 'balears',
        clim_period: tuple[int, int] = (1987, 2021),

        # Options
        progress_bar: bool = True,
        profilers: bool = False
) -> xr.Dataset:
    """
    Saves a MHWs dataset using xarray.

    Parameters
    ----------
    ds_mhws: xr.Dataset
        MHWs dataset to save.

    ds_type: str
        Can be 'yearly' or 'all_events'.

    dataset_used: str 
        Describes the dataset from which the MHWs computations were performed.
        Can be `'rep'`, `'medrea_bot'` or `'medrea_50m'` for example.

    region: str, default='balears'
        Region in which the MHWs computations were performed. Can be `'balears'` or `'med'` for example.

    clim_period: tuple[int, int], default=(1987,2021)
        Climatology period used for the MHWs computations.
    
    progress_bar: bool, default=True
        If `True`, shows a progress bar when saving the dataset.
    
    profilers: bool, default=True
        If `True`, shows the profilers after the dataset has been saved.
    
    Returns
    ----------
    ds_mhws: xr.Dataset
        The computed MHWs dataset.
    """
    
    # Getting dataset's associated filepath
    mhws_dataset_path = mhws_dataset_pattern.format(
        type = ds_type,
        dataset = dataset_used,
        detrended = '_detrended' if detrended else '',
        region = region,
        clim_start = clim_period[0],
        clim_end = clim_period[1]
    )

    # Printing the good news
    print(f"Saving MHWs dataset to {mhws_dataset_path}")

    # Saving dataset to nc
    ds_mhws = save_dataset_to_nc(
        ds = ds_mhws,
        file_path = mhws_dataset_path,
        progress_bar = progress_bar,
        profilers = profilers
    )

    # Printing the good news
    print(f" -> Saved!")

    # Returning the dataset after Dask calculations have been performed
    return ds_mhws

def load_mhws_dataset(
        # File path parameters
        ds_type: str,
        dataset_used: str,
        detrended: bool = False,
        region: str = 'balears',
        clim_period: tuple[int, int] = (1987, 2021),
) -> xr.Dataset:
    """
    Loads a MHWs dataset using xarray.

    Parameters
    ----------
    ds_type: str
        Can be 'yearly' or 'all_events'.
    
    dataset_used: str 
        Describes the dataset from which the MHWs computations were performed.
        Can be `'rep'`, `'medrea_bot'` or `'medrea_50m'` for example.

    region: str, default='balears'
        Region in which the MHWs computations were performed. Can be `'balears'` or `'med'` for example.

    clim_period: tuple[int, int], default=(1987,2021)
        Climatology period used for the MHWs computations.
    
    Returns
    ----------
    ds_mhws: xr.Dataset
        The loaded MHWs dataset.
    """

    # Getting dataset's associated filepath
    mhws_dataset_path = mhws_dataset_pattern.format(
        type = ds_type,
        dataset = dataset_used,
        detrended = '_detrended' if detrended else '',
        region = region,
        clim_start = clim_period[0],
        clim_end = clim_period[1]
    )

    # Loading the dataset
    ds_mhws = load_nc_to_dataset(
        file_path = mhws_dataset_path,
        remove_bay_of_biscay = False
    )

    # Printing the good news
    print("Loaded MHWs dataset.")
    
    # Returning the final dataset
    return ds_mhws



def save_dataset_to_nc(
        # Dataset
        ds: xr.Dataset,
        file_path: str,

        # Options
        progress_bar: bool = True,
        profilers: bool = False,
) -> xr.Dataset:
    """
    Basic function to save a dataset using xarray.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset to save.

    file_path: str 
        Filepath where the dataset should be saved.
    
    progress_bar: bool, default=True
        If `True`, shows a progress bar when saving the dataset.
    
    profilers: bool, default=True
        If `True`, shows the profilers after the dataset has been saved.
    
    Returns
    ----------
    ds: xr.Dataset
        The computed dataset.
    """

    # Compute the dataset using a progress bar
    if progress_bar:
        with ProgressBar():
            # ds = ds.compute()
            ds.to_netcdf(file_path)
    
    # Compute the dataset using a progress bar and other profilers
    elif progress_bar and profilers:
        with (ProgressBar(), ResourceProfiler() as rprof, Profiler() as prof):
            # ds = ds.compute()
            ds.to_netcdf(file_path)
        
        # Save resource profiles as html file
        from datetime import datetime
        path = os.path.join(codigos_location, '.dask_profiles', datetime.now().isoformat() + '.html')
        visualize([prof, rprof], path, show=True, save=True)

    # Compute the dataset using no visualisation tools
    else:
        # ds = ds.compute()
        ds.to_netcdf(file_path)

    # Returning the final dataset
    # return ds



def load_nc_to_dataset(
        file_path: str,
        name_dict: dict = None,
        remove_bay_of_biscay: bool = False,
        region_selector: str | None = None,
        **kwargs
) -> xr.Dataset:
    """
    Loads a MHWs dataset using xarray.

    Parameters
    ----------
    file_path: str 
        Filepath from where the dataset should be loaded.

    remove_bay_of_biscay: bool, default=True
        If `True`, all the values of the dataset in the bay of biscay is discarded.
        Note that the coordinates must be named `lon` and `lat` for it to work.
    
    Returns
    ----------
    ds: xr.Dataset
        The loaded dataset.
    
    Other Parameters
    ----------------
    **kwargs
        All other arguments are passed to `xr.open_dataset`
    """

    # Loading the dataset
    ds = xr.open_dataset(
        file_path,
        decode_times=True,
        **kwargs
    )

    # Rename the variables and coordinates as asked
    if name_dict:
        ds = ds.rename(name_dict)

    # Apply region selector
    if region_selector == 'balears':
        lon_selector = slice(-0.9, 5.1)
        lat_selector = slice(37.6, 41.1)
        ds = ds.sel(lat=lat_selector, lon=lon_selector)

    # Remove bay of Biscay
    if remove_bay_of_biscay:
        ds = ds.where((ds.lon > 0) | (ds.lat < 42))
    
    # Returning the final dataset
    return ds