
import os

import matplotlib.pyplot as plt

import numpy as np
import xarray as xr

from dask.diagnostics import ProgressBar, ResourceProfiler, Profiler, visualize

# from pyscripts.load_save_dataset import load_bathy
from shapely.geometry import Point, Polygon

# from dask.distributed import Client
# client = Client()
codigos_location = os.path.abspath(os.getcwd())
datos_location = os.path.join(codigos_location, "..", "Datos")


def apply_regional_mask(ds: xr.Dataset | xr.DataArray, region: str | int, ds_bathy_mfc: xr.Dataset | xr.DataArray, return_mask = False):
    # if ds_bathy_mfc is None:
    #     ds_bathy_mfc = load_bathy(source="mfc")
    #     ds_bathy_mfc = ds_bathy_mfc.sel(lon=slice(-0.9, 5.1)).sel(lat=slice(37.6, 41.1))
    #     # ds_bathy_mfc = ds_bathy_mfc.sel(lon=slice(-1,5.5)).sel(lat=slice(37,41))
    
    if isinstance(ds_bathy_mfc, xr.Dataset):
        ds_bathy_mfc = ds_bathy_mfc.depth

    regions = [
        "continental_coast",
        "balearic_coast",
        "balearic_sea_deep",
        "west_algerian_deep",
    ]
    regions_shortnames = {
        "CC": "continental_coast",
        "BIC": "balearic_coast",
        "DBS": "balearic_sea_deep",
        "DWAR": "west_algerian_deep",
    }

    if isinstance(region, int): region = regions[region]
    if region in regions_shortnames: region = regions_shortnames[region]

    # Création des grilles de points
    lon, lat = np.meshgrid(ds.lon.values, ds.lat.values)

    # Flatten pour itération
    lon_flat = lon.flatten()
    lat_flat = lat.flatten()

    def to_xarray(mask):
        return xr.DataArray(
            mask,
            coords={"lat": ds.lat, "lon": ds.lon},
            dims=("lat", "lon")
        )

    if region in ["continental_coast", "balearic_coast"]:
        # Define polygon
        coords_island_poly = [
            (0.69, 39.19),
            (3.4, 40.75),
            (4.86, 40.26),
            (4.5, 39.2),
            (1, 38)
        ]  # (lon, lat)
        island_polygon  = Polygon(coords_island_poly)
        island_mask     = np.array([island_polygon.contains(Point(lon_, lat_)) for lon_, lat_ in zip(lon_flat, lat_flat)])
        island_mask_2d  = island_mask.reshape(len(ds.lat), len(ds.lon))

        if region == "continental_coast":
            continental_coast_mask = to_xarray((ds_bathy_mfc < 200) & ~island_mask_2d)

            if return_mask:
                return continental_coast_mask
            else:
                return ds.where(continental_coast_mask)
        
        elif region == "balearic_coast":
            balearic_coast_mask = to_xarray((ds_bathy_mfc < 200) & island_mask_2d)

            if return_mask:
                return balearic_coast_mask
            else:
                return ds.where(balearic_coast_mask)

    elif region in ["balearic_sea_deep", "west_algerian_deep"]:
        # Define polygon
        coords_basin_poly = [
            (20, 20), (-5, 20), (-5, 38.8), # Basic rectangle
            (0.12, 38.8), (1.41, 38.83), # Ibiza Channel
            (1.6, 39.08), (3.13, 39.94), # Mallorca Channel
            (4.24, 40), (20, 40), # Menorca Channel
        ]  # (lon, lat)
        basin_polygon   = Polygon(coords_basin_poly)
        basin_mask      = np.array([basin_polygon.contains(Point(lon_, lat_)) for lon_, lat_ in zip(lon_flat, lat_flat)])
        basin_mask_2d   = basin_mask.reshape(len(ds.lat), len(ds.lon))

        if region == "balearic_sea_deep":
            balearic_sea_deep_mask = to_xarray((ds_bathy_mfc > 200) & ~basin_mask_2d)

            if return_mask:
                return balearic_sea_deep_mask
            else:
                return ds.where(balearic_sea_deep_mask)
        
        elif region == "west_algerian_deep":
            west_algerian_deep_mask = to_xarray((ds_bathy_mfc > 200) & basin_mask_2d)

            if return_mask:
                return west_algerian_deep_mask
            else:
                return ds.where(west_algerian_deep_mask)

    print(f"Region not found {region}")


def detrend_dataset(ds):

    # TODO : Detrend

    return ds

def calculate_linear_trend(ds, var_name: str = "T"):

    # Converting time as float year unit
    ds["time"] = ds['time.year'].data + (ds['time.dayofyear'].data - 1) / 365.25

    # Operations to calculate the trend
    ds_trend = ds[var_name].polyfit(dim='time', deg=1)
    ds_trend = ds_trend.sel(degree=1, drop=True)
    ds_trend = ds_trend.rename({"polyfit_coefficients" : "sst_trend"})

    return ds_trend


def calculate_climatology(ds, time_name="time"):
    # Operations to calculate the climatology
    ds_clim = ds.groupby(f"{time_name}.dayofyear").mean()

    # Calculate 10th and 90th percentile
    

    return ds_clim
