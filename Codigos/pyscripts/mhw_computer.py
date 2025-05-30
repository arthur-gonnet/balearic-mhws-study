########################################################################################################################
########################################################################################################################

###### USER NOTES:
# This script gathers all the codes to load and save datasets using xarray. 

## Functions description
#
# - compute_mhw_maps_apply_ufunc(...):
#       ...
#
# - compute_mhw_wrapped(...):
#       ...

########################################################################################################################
##################################### IMPORTS ##########################################################################
########################################################################################################################

# Basic imports
import os
import glob as glob

# Advanced imports
import numpy as np
import xarray as xr

import math
from datetime import date

# Local imports
import pyscripts.marineHeatWaves as mhw
import pyscripts.options as opts
import pyscripts.rt_anatools as rt

########################################################################################################################
##################################### CODES ############################################################################
########################################################################################################################


def compute_mhw_yearly(
        ds: xr.Dataset | xr.DataArray,
        using_dataset: str,
        var_name: str = "T",

        clim_period: tuple[int, int] = (1987,2021),
        detrend = False,
) -> xr.Dataset:
    # Compatibility snippet so that the function can work with both DataArray and Dataset
    if isinstance(ds, xr.Dataset):
        ds = ds[var_name]

    # Investigate the stacking possibilities of the dataarray
    stackable_dims = [
        dim for dim in ds.dims
        if not dim == "time" and ds[dim].size > 1
    ]
    stack = len(stackable_dims) > 1

    # Stacking possible dimensions for perfomance sake
    if stack:
        print("Stacking dimensions :", stackable_dims)
        ds = ds.stack(pos=stackable_dims)
    
    # Years of the dataset
    years = np.unique(ds["time.year"].values)

    # Computing the mhws
    #   result contains the resulting mhws statistics
    #   shape is (mhws_stats=18, pos, years)
    results = xr.apply_ufunc(
        # Inputs
        compute_mhw_yearly_wrapped,
        ds.time,
        ds.chunk({dim: -1 if dim == "time" else "auto" for dim in ds.dims}),
        kwargs={
            "clim_period": clim_period,
            "detrend": detrend
        },

        # Dimensions of input and output
        input_core_dims     = [['time'], ['time']],
        output_core_dims    = [(['year'])] * len(opts.mhws_stats),
        
        # Type and size of output
        output_dtypes       = [float] * len(opts.mhws_stats),
        dask_gufunc_kwargs  = dict(
            output_sizes = {
                'year': len(years)
            }
        ),

        # Dask Options
        vectorize   = True,
        dask        = 'parallelized',
    )

    # Assigning coordinates
    output_ds = xr.Dataset({
        var: data.assign_coords(year=years)
        for var, data in zip(opts.mhws_stats, results)
    })

    # Unstacking pos
    if stack:
        output_ds = output_ds.unstack("pos")

    # Changing dimensions order for plotting maps
    if "lat" in stackable_dims and "lon" in stackable_dims:
        output_ds = output_ds.transpose('lat', 'lon', 'year', ...)
    
    # Adding variables metadata
    for stat in opts.mhws_stats:
        output_ds[stat].attrs["shortname"]  = opts.mhws_stats_shortname[stat]
        output_ds[stat].attrs["longname"]   = opts.mhws_stats_longname[stat]
        output_ds[stat].attrs["unit"]       = opts.mhws_stats_units[stat]

    # Adding dataset attributes
    output_ds.attrs['climatologyPeriod'] = f'{clim_period[0]}-{clim_period[1]}'
    output_ds.attrs['description'] = opts.mhw_yearly_dataset_description

    if using_dataset.lower() == "rep":
        output_ds.attrs['acknowledgment'] = opts.rep_acknowledgment

    elif using_dataset.lower() == "medrea":
        output_ds.attrs['acknowledgment'] = opts.medrea_acknowledgment

    # Returning the final Dataset!
    return output_ds

def compute_mhw_yearly_wrapped(t: np.array, sst: np.array, clim_period: tuple[int, int], detrend: bool):
    # Ignore all-nan timeseries for performance sake
    if np.isnan(sst).all():
        nans = np.array([
            np.nan
            for _ in range(t[0].astype('datetime64[Y]').astype(int) + 1970, t[-1].astype('datetime64[Y]').astype(int) + 1970+1)
        ])
        return tuple(nans for _ in opts.mhws_stats)

    # Array manipulation to fit mhw module requirements
    t = t.astype('datetime64[D]').astype(int) + 719163 # to ordinal time
    temp = sst.copy()

    if detrend:
        temp = rt.detrend_timeserie(temp)

    # Computing MHWs using mhw module
    mhws, clim = mhw.detect(t, temp, climatologyPeriod=clim_period, cutMhwEventsByYear=True)
    mhwBlock = mhw.blockAverage(t, mhws, clim, temp=temp)

    # Return yearly averages of MHW stats
    return tuple(mhwBlock[stat] for stat in opts.mhws_stats)


    ## All events approach (not yearly stats)

# Every stats availables
mhws_all_events_stats = [
    'time_start', 'time_end', 'time_peak',
    # 'date_start', 'date_end', 'date_peak',
    'index_start', 'index_end', 'index_peak',
    'duration', 'duration_moderate', 'duration_strong', 'duration_severe', 'duration_extreme',
    'intensity_max', 'intensity_mean', 'intensity_var', 'intensity_cumulative', 'intensity_max_relThresh',
    'intensity_mean_relThresh', 'intensity_var_relThresh', 'intensity_cumulative_relThresh', 'intensity_max_abs',
    'intensity_mean_abs', 'intensity_var_abs', 'intensity_cumulative_abs', 'category', 'rate_onset', 'rate_decline',
]

# Stats to compute timeseries of
mhws_all_events_useful_stats = [
    'duration', 'duration_moderate', 'duration_strong', 'duration_severe', 'duration_extreme',
    'intensity_max', 'intensity_mean', 'intensity_var', 'intensity_cumulative', 'intensity_max_abs',
    'intensity_mean_abs', 'intensity_var_abs', 'intensity_cumulative_abs', 'category', 'rate_onset', 'rate_decline',
]

clim_keys = ['thresh', 'seas', 'missing', 'sst']

def compute_mhw_all_events(
        ds: xr.Dataset | xr.DataArray,
        using_dataset: str,
        var_name: str = "T",

        clim_period: tuple[int, int] = (1987,2021),
        detrend = False,
) -> xr.Dataset:
    if isinstance(ds, xr.Dataset):
        ds = ds[var_name]

    # dims_to_stack = [dim for dim in ds.dims if dim != "time"]
    # should_stack = sum([ds[dim].size > 1 for dim in dims_to_stack]) > 1
    if "lon" in ds.dims and "lat" in ds.dims:
        stack = ds.lon.size > 1 and ds.lat.size > 1
    else:
        stack = False

    # Stacking lon and lat for perfomance sake
    if stack:
        ds = ds.stack(pos=("lon", "lat"))
    
    # from dask.diagnostics import ProgressBar
    # # As operations are not dask-oriented, need to compute
    # with ProgressBar():
    #     ds = ds.compute()

    # Computing the mhws
    #   result contains the resulting mhws statistics
    #   shape is (things=33, pos, event_number/time)
    results = xr.apply_ufunc(
        # Inputs
        compute_mhw_all_events_wrapped,
        ds.time,
        ds.chunk({dim: -1 if dim == "time" else "auto" for dim in ds.dims}),
        kwargs={
            'clim_period': clim_period,
            "detrend": detrend
        },

        # Dimensions of input and output
        input_core_dims     = [['time'], ['time']],
        output_core_dims    = [['event_number']] * len(mhws_all_events_stats) + [['time']] * len(clim_keys),
        
        # Type and size of output
        output_dtypes       = [float] * (len(mhws_all_events_stats) + len(clim_keys)),
        dask_gufunc_kwargs  = dict(
            output_sizes = {
                # As using parallelization, dimension sizes must be fixed. This is the maximum
                # theoretical value possible for event_number
                'event_number': math.floor(ds.time.size/6),
                'time': ds.time.size,
            }
        ),

        # Dask Options
        vectorize   = True,
        dask        = 'parallelized',
    )

    # print(results)

    # Assigning coordinates
    output_ds = xr.Dataset({
        var: data.assign_coords(event_number=range(math.floor(ds.time.size/6)))
        for var, data in zip(mhws_all_events_stats, results[:len(mhws_all_events_stats)])
    } | {
        'clim_' + var: data.assign_coords(time=ds.time)
        for var, data in zip(clim_keys, results[len(mhws_all_events_stats):])
    })

    # Unstacking pos
    if stack:
        output_ds = output_ds.unstack("pos")

    # Étape 2: Identifier les variables qui dépendent de 'event_number'
    # vars_event = [var for var in output_ds.data_vars if 'event_number' in output_ds[var].dims]

    # # Étape 3: Créer un masque booléen qui dit s’il existe une valeur non-NaN à chaque event_number
    # # On combine tous les masques avec un "ou" logique
    # valid_mask = sum(~output_ds[var].isnull() for var in vars_event) > 0

    # # Étape 4: Réduire le masque sur toutes les dimensions sauf 'event_number'
    # # => On garde True si au moins un point sur toutes les autres dimensions est non-NaN
    # valid_any = valid_mask.any(dim=("depth", "lat", "lon"))

    # # Étape 5: Trouver le dernier index valide
    # valid_event_numbers = output_ds['event_number'].values[valid_any.values]
    # if len(valid_event_numbers) > 0:
    #     max_valid_event_number = valid_event_numbers.max()
        
    #     # Optionnel : tronquer le dataset
    #     output_ds = output_ds.sel(event_number=slice(0, max_valid_event_number))

    from dask.diagnostics import ProgressBar

    with ProgressBar():
        print("Computing MHWs dataset")
        output_ds = output_ds.compute()

    dims_to_check = [dim for dim in output_ds.dims if dim not in ("time", "event_number")]

    if len(dims_to_check) > 0:
        nan_mask = output_ds.time_start.isnull().all(dim=dims_to_check)
    else:
        nan_mask = output_ds.time_start.isnull()
    
    if nan_mask.any():
        first_nan = nan_mask.argmax().item()
        if first_nan != 0:
            output_ds = output_ds.sel(event_number=slice(0, first_nan))

    else:
        first_nan = None


    # Changing dimensions order
    # output_ds = output_ds.transpose('lat', 'lon', 'year')
    # output_ds = output_ds.transpose('lat', 'lon', 'year')
    # output_ds = output_ds.transpose('year')

    # for stat in opts.mhws_stats:
    #     output_ds[stat].attrs["shortname"]  = opts.mhws_stats_shortname[stat]
    #     output_ds[stat].attrs["longname"]   = opts.mhws_stats_longname[stat]
    #     output_ds[stat].attrs["unit"]       = opts.mhws_stats_units[stat]

    # Adding attributes
    output_ds.attrs['climatologyPeriod'] = f'{clim_period[0]}-{clim_period[1]}'
    output_ds.attrs['description'] = opts.mhw_dataset_description

    match using_dataset.lower():
        case "cmems_sst":
            output_ds.attrs['acknowledgment'] = opts.cmems_sst_acknowledgment

        case "cmems_mfc":
            output_ds.attrs['acknowledgment'] = opts.cmems_mfc_acknowledgment

    return output_ds

def compute_mhw_all_events_wrapped(t: np.array, sst: np.array, clim_period: tuple[int, int], detrend: bool):
    # print(f"received time array of type {type(t)}, shape {t.shape}")
    # print(f"received sst array of type {type(sst)}, shape {sst.shape}")

    # t = ((t - np.datetime64("0001-01-01")) / np.timedelta64(1, "D")).astype(int)
    t = t.astype('datetime64[D]').astype(int) + 719163
    temp = sst.copy()

    if detrend:
        temp = rt.detrend_timeserie(temp)

    # print(t.shape, t)
    # print(sst.shape, sst)
    mhws, clim = mhw.detect(t, temp, climatologyPeriod=clim_period)

    # for stat in mhws_all_events_stats:
    #     print(f"{stat}: {'list' if isinstance(np.array(mhws[stat]), list) else np.array(mhws[stat]).dtype}, {np.array(mhws[stat]).shape}" )
    
    # for key in clim_keys:
    #     print(f"{key}: {clim[key].dtype}, {clim[key].shape}" )
        
    categories = {'Moderate': 1, 'Strong': 2, 'Severe': 3, 'Extreme': 4}
    mhws["category"] = [categories[cat] for cat in mhws["category"]]
    clim["sst"] = temp.copy()

    # print([mhws[stat] for stat in mhws_all_events_stats] + [clim[key] for key in clim_keys])
    return tuple(
        [
            np.pad(np.array(mhws[stat], dtype=np.float64), (0, math.floor(len(t)/6)-len(mhws[stat])), constant_values=np.nan)
            for stat in mhws_all_events_stats
        ] +
        [clim[key].astype(float) for key in clim_keys]
    )


def get_mhw_ts_from_ds(
        ds_mhws,
        lon,
        lat,
        depth,

        calculate_mhw_mask=True
):
    if not (lon is None or lat is None or depth is None):
        ds = ds_mhws.sel(lon=lon, lat=lat, depth=depth, method='nearest')
    elif not depth is None:
        ds = ds_mhws.sel(depth=depth, method='nearest')
    else:
        ds = ds_mhws

    time = ds.time.values

    if ds.time_start.isnull().all():
        return -1, -1

    first_nan = ds.time_start.isnull().argmax().item()

    if first_nan != 0:
        # Cut the dataset when all the values are nan
        ds = ds.isel(event_number=slice(0, first_nan))
    
    mhws = {}

    for stat in mhws_all_events_stats:
        mhws[stat] = ds[stat].values
    
    for stat in clim_keys:
        mhws["clim_"+stat] = ds["clim_"+stat].values

    mhws["n_events"] = len(mhws["time_start"])
    
    # print(mhws['time_start'].astype(int))
    mhws["date_start"] = [date.fromordinal(time) for time in mhws['time_start'].astype(int)]
    mhws["date_end"] = [date.fromordinal(time) for time in mhws['time_end'].astype(int)]
    mhws["date_peak"] = [date.fromordinal(time) for time in mhws['time_peak'].astype(int)]

    mhws["index_start"] = mhws["index_start"].astype(int)
    mhws["index_end"] = mhws["index_end"].astype(int)
    
    if calculate_mhw_mask:
        mhw_mask = np.ones(ds.time.size, dtype=bool)
        # mhws["mhw_number"] = np.zeros(ds.time.size, dtype=float)
        mhws["mhw_number"] = np.full(ds.time.size, np.nan)

        for stat in mhws_all_events_useful_stats:
            mhws[f"mhw_{stat}"] = np.full(ds.time.size, np.nan)

        for ev in range(mhws["n_events"]):
            mhw_mask[mhws["index_start"][ev]:mhws["index_end"][ev]+1] = False
            mhws["mhw_number"][mhws["index_start"][ev]:mhws["index_end"][ev]+1] = ev+1

            for stat in mhws_all_events_useful_stats:
                mhws[f"mhw_{stat}"][mhws["index_start"][ev]:mhws["index_end"][ev]+1] = mhws[stat][ev]

        mhws["mhw_intensity"] = mhws["clim_sst"] - mhws["clim_seas"]
        
        # mhws["mhw_number"][mhw_mask] = np.nan
        mhws["mhw_intensity"][mhw_mask] = np.nan

        mhws["mhw_mask"] = mhw_mask

    return time, mhws
