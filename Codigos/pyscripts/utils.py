


from matplotlib.ticker import MaxNLocator

import xarray as xr
import numpy as np
from geopy.distance import geodesic

import copy


def extract_transect(ds: xr.Dataset | xr.DataArray, pos0: tuple[float, float], pos1: tuple[float, float]) -> xr.Dataset:
    """
    Extract a transect from a dataset.
    
    Parameters
    ----------
    ds: xr.Dataset | xr.DataArray
        Dataset from which the transect should be extracted.
    
    pos0: tuple[float, float]
        Starting position of the transect, given as (lat, lon).
        If the longitude of `pos1` is less than the one from `pos0`, then `pos0`
        becomes the ending point of the transect to keep longitudes ascending.
    
    pos1: tuple[float, float]
        Ending position of the transect, given as (lat, lon).

    Returns
    ----------
    ds_transect: xr.Dataset
        Dataset containing the transect.
    """

    # Points de départ et d'arrivée
    lat0, lon0 = pos0
    lat1, lon1 = pos1

    if lon1 < lon0:
        print(f"Reversing point order to keep longitudes ascending (given initial lon0:{lon0:.1f}, lon1:{lon1:.1f}).")
        lon0, lon1 = lon1, lon0
        lat0, lat1 = lat1, lat0

    # Récupérer les coordonnées existantes
    lat_vals = ds['lat'].values
    lon_vals = ds['lon'].values

    # Trouver les indices des points de départ et d'arrivée
    i0 = np.abs(lat_vals - lat0).argmin()
    j0 = np.abs(lon_vals - lon0).argmin()
    i1 = np.abs(lat_vals - lat1).argmin()
    j1 = np.abs(lon_vals - lon1).argmin()

    # Nombre de points = max(delta_lat, delta_lon) + 1
    npts = max(abs(i1 - i0), abs(j1 - j0)) + 1

    # Tracer les indices intermédiaires (discrets)
    lat_idx = np.round(np.linspace(i0, i1, npts)).astype(int)
    lon_idx = np.round(np.linspace(j0, j1, npts)).astype(int)

    # Coordonnées réelles
    lat_real = ds['lat'].isel(lat=("transect", lat_idx))
    lon_real = ds['lon'].isel(lon=("transect", lon_idx))

    # Distance cumulée
    dist_km = [0.0] + [geodesic((lat0, lon0), (float(lat_real[i]), float(lon_real[i]))).km for i in range(1, npts)]
    dist_da = xr.DataArray(dist_km, dims="transect")

    # Extraction globale avec isel
    ds_transect = ds.isel(
        lat = xr.DataArray(lat_idx, dims="transect"),
        lon = xr.DataArray(lon_idx, dims="transect")
    )

    # Réassigner/ajouter coordonnées transect + distance
    ds_transect = ds_transect.assign_coords({
        "transect": np.arange(npts),
        "lat": lat_real,
        "lon": lon_real,
        "distance": dist_da
    })

    return ds_transect


def lon_to_str(
    value: float,
    axis = "lon",
    precision = 2,
) -> str:
    """
    Uses matplotlib's MaxNLocator to compute nice rounded range and ticks.
    
    Parameters
    ----------
    vmin: float | None
        Initial vmin value. Note that if one of vmin or vmax is None, the range is returned unchanged.
    
    vmax: float | None
        Initial vmin value. Note that if one of vmin or vmax is None, the range is returned unchanged.
    
    nbins: int | 'auto', default: 10
        Matplotlib's MaxNLocator parameter.
        Maximum number of intervals; one less than max number of ticks.
        If the string 'auto', the number of bins will be automatically determined based on the length of the axis.

    Returns
    ----------
        nice_vmin, nice_vmax
    """

    precision = f":.{precision}f" if not precision is None else ""

    # if precision:
    #     value = round(value, precision)

    if axis == "lon":
        suffix = 'W' if value < 0 else '' if value == 0 else 'E'
    elif axis == "lat":
        suffix = 'S' if value < 0 else '' if value == 0 else 'N'

    pattern = "{value" + precision + "}°{suffix}"
    return pattern.format(value=abs(value), suffix=suffix)


def nice_range(
    vmin: float | None,
    vmax: float | None,
    nbins: int = 10
) -> tuple[float, float]:
    """
    Uses matplotlib's MaxNLocator to compute nice rounded range and ticks.
    
    Parameters
    ----------
    vmin: float | None
        Initial vmin value. Note that if one of vmin or vmax is None, the range is returned unchanged.
    
    vmax: float | None
        Initial vmin value. Note that if one of vmin or vmax is None, the range is returned unchanged.
    
    nbins: int | 'auto', default: 10
        Matplotlib's MaxNLocator parameter.
        Maximum number of intervals; one less than max number of ticks.
        If the string 'auto', the number of bins will be automatically determined based on the length of the axis.

    Returns
    ----------
        nice_vmin, nice_vmax
    """

    if vmin is None or vmax is None:
        return vmin, vmax

    if vmin == 0 and vmax == 0:
    # if (vmin < 1e-10 and vmin > 1e-10 and vmax < 1e-10 and vmax > 1e-10) or (vmin is None and vmax is None):
        return 0., 1.
    
    locator = MaxNLocator(nbins=nbins, prune=None)
    ticks = locator.tick_values(vmin, vmax)

    return ticks[0], ticks[-1] # ticks[1] - ticks[0] is the step


def soft_add_values(original: dict, values: dict, inplace: bool = True):
    """
    Copies the key and value pairs of the `values` dictionnary into the `original` dictionnary.
    If a key exists in both `values` and `original` dictionnary, the one of the `original` dictionnary is kept.

    Parameters
    ----------
    original: dict
        Original dictionnary to modify.
    
    values: dict
        Dictionnary containing the values to add to the `original` dictionnary.
    
    inplace: bool, default=True
        If True, this function returns a modified copy of `original` dictionnary using `copy.deepcopy()`.
        If False, the original dictionnary is modified in place.
    
    Returns
    ----------
    original: dict
        Either a copy or original `original` dictionnary.
    """

    if not inplace:
        original = copy.deepcopy(original)

    for key in values:
        if key not in original:
            original[key] = values[key]
    
    return original


def soft_override_value(dict: dict, key, value):
    """
    Add a key and value pair into a dictionnary only if `value` is not `None`.

    Parameters
    ----------
    dict: dict
        Original dictionnary to modify.
    
    key
        The key of the value to modify.
    
    value
        The value to be added. If `None`, then nothing is done.
    """

    if not value is None:
        dict[key] = value


def not_null(*args):
    """
    Returns the first not null value from the passed arguments.

    Parameters
    ----------
    *args
        Choices of value. The first not null value is returned.
    """

    for arg in args:
        if not arg is None:
            return arg
    
    return


def bold(text):
    return r"$\bf{" + str(text).replace(' ', ' \ ') + "}$"
