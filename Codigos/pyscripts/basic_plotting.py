########################################################################################################################
########################################################################################################################

###### USER NOTES:
# This script gather all the codes to load and save datasets using xarray. 

## Functions description
#
# - subplot(...):
#       Loads a bathymetry dataset using xarray (either GEBCO or MEDREA).
#
# - plot_map(...):
#       Loads the REP dataset using xarray.
#
# - plot_transect(...):
#       Loads the MEDREA dataset using xarray.
#
# - plot_vertical_mean(...):
#       Saves a MHW dataset using xarray.
#
# - plot_timeserie(...):
#       Loads a MHW dataset using xarray.
#
# - add_colorbar(...):
#       Basic function to save a dataset using xarray.
#
# - get_locator_from_ticks (...):
#       Basic function to load a dataset using xarray.

########################################################################################################################
##################################### IMPORTS ##########################################################################
########################################################################################################################

# Advanced imports
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator, MultipleLocator, StrMethodFormatter, FuncFormatter, FixedLocator, AutoMinorLocator
import cmocean.cm as cmo
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Local imports
from pyscripts.utils import nice_range, soft_override_value, soft_add_values


########################################################################################################################
##################################### CODES ############################################################################
########################################################################################################################


def subplot(
        # Parameter of subplot
        nrows: int, ncols: int,
        subplots_settings: list[dict],
        pad_subplots: tuple[float, float] = None,

        # Parameter of figure
        fig: plt.Figure = None,
        figsize: tuple[float, float] = None,
        subplotsize: tuple[float, float] = (7.3, 4.67),
        figdpi: float = 100,
        fig_fontsize: int = 14,
        fig_title: str = None,
        fig_is_a_map: bool = False,

        # Parameter of colorbar
        fig_cbar: bool = False,
        fig_cbar_row: bool = False,
        fig_vmin: float = None,
        fig_vmax: float = None,
        fig_cmap: str = None,
        fig_cbar_unit: str = None,
        fig_cbar_ticks: int | tuple[float, float] | list[float] | list[list[float], list[str]] = None,
        fig_cbar_orientation: str = 'vertical',
        fig_cbar_pad: float = 0.005,
        fig_cbar_fraction: float = 0.025,

        # Parameter for saving the figure
        save_plot: bool = False,
        save_path: str = "",

        show_plots: bool = False,
):
    """
    Build a figure with subplots with matplotlib. The settings for each subplot must
    be provided as a dictionnary containing the position of the subplot in the figure,
    the function to be called to plot on the desired axe and all other arguments that 
    will be passed to the plotting function. Other options of this function are used to
    define the figure, as well as the optional colorbar.

    Parameters
    ----------
    nrows: int 
        The number of rows in the subplot grid.

    ncols: int 
        The number of columns in the subplot grid.

    subplots_settings: list[dict]
        A list containing the settings of each subplot. Each subplot
        setting dictionnary must contain :

        'pos': the index of the position of the subplot as in the function `plt.subplot()`. 
        'func': the plotting function to plot on the desired axe.
        ...: all the arguments to be passed to the plotting function.

        Note : if using a colorbar, the displayed data show be named 'variable_data'.
    
    pad_subplots: tuple[float, float], optional
        This option adjust the padding existing between the subplots. Uses a relative unit.
        Beware that it disables the constrained_layout property (enabled by default).
    
    fig: plt.Figure, optional
        If created beforehand, the figure to which subplots will be added. Otherwise, a new
        figure is created.

    figsize: tuple[float, float], optional
        Width, height of the figure in inches. Overrides `subplotsize`. 

    subplotsize: tuple[float, float], default=(7.3, 4.67)
        The size of each subplots in inches. Thus, `figsize=(ncols*subplotsize[0], nrows*subplotsize[1])`.
        Overriden if `figsize` is defined.
    
    figdpi: float, default=100   
        The resolution of the figure in dots-per-inch.

    fig_fontsize: int, default=14
        Fontsize that will be applied to the whole figure.
    
    fig_title: str, optional
        Title of the whole figure.
    
    fig_is_a_map: bool, default=False
        This option should be enabled when displaying a map. It defines the projection
        mode for the axe, which is required for some cases when plotting maps.
    
    fig_cbar: bool, default=False
        Add a colorbar at the figure scope. When doing so, the vmin and vmax values are
        defined at figure scope with the minimal and maximal values of all the subplots.
        Also, subplot-scoped colorbar are disabled. Overriden if `fig_cbar_row` is enabled.
    
    fig_cbar_row: bool, default=False
        Add a colorbar for each row of subplots. When doing so, the vmin and vmax values are
        defined at row scope with the minimal and maximal values of all the subplots in the given row.
        Also, subplot-scoped colorbar are disabled. Overrides `fig_cbar`.
    
    fig_cbar_ticks: int | tuple[float, float] | list[float], optional
        Defines the ticks of the colorbar. Giving an int will use a `MaxNLocator`.
        Giving a tuple will use a `MultipleLocator`, the first float being the base and the second the offset.
        Giving a list of float will use a `FixedLocator` with the values given.
    
    fig_cbar_pad: float, default=0.005
        Defines the pad between the plot and the colorbar. Uses a relative unit.

    fig_cbar_fraction: float, default=0.025
        Defines the fraction size of the colorbar.

    show_plots: bool, default=False
        Shows the pending plots to show using `plt.show()`.
    """
    
    # Apply subplotsize so that figsize = ncols*subplotwidth, nrows*subplotheight
    if figsize is None:
        figsize = (ncols*subplotsize[0], nrows*subplotsize[1])
    
    # If adding a figure colorbar, compute the range of it using data of whole figure
    if fig_cbar and not fig_cbar_row:
        if fig_vmin is None:
            fig_vmin = np.nanmin([np.nanmin(subplot_setting['variable_data']) for subplot_setting in subplots_settings])
        
        if fig_vmax is None:
            fig_vmax = np.nanmax([np.nanmax(subplot_setting['variable_data']) for subplot_setting in subplots_settings])
            
    # Apply fontsize globally
    with plt.rc_context({'font.size': fig_fontsize}):
        # Create a new figure only if the figure didn't exist beforehand
        if not fig:
            fig = plt.figure(figsize=figsize, dpi=figdpi, constrained_layout=(pad_subplots is None))
            print(f"Making figure of {figsize[0]*figdpi}x{figsize[1]*figdpi}px")

        # Adjust subplot padding (beware that it disables the constrained_layout property)
        if pad_subplots:
            fig.subplots_adjust(wspace=pad_subplots[0], hspace=pad_subplots[1])

        # If the colorbars are defined by rows, compute the range of it using data of the row
        if fig_cbar_row:
            fig_vmins = []
            fig_vmaxs = []

            for row in range(nrows):
                vmin = np.nanmin([np.nanmin(subplot_setting['variable_data']) for subplot_setting in subplots_settings if (subplot_setting['pos']-1)//ncols == row])
                vmax = np.nanmax([np.nanmax(subplot_setting['variable_data']) for subplot_setting in subplots_settings if (subplot_setting['pos']-1)//ncols == row])
                
                fig_vmins.append(vmin)
                fig_vmaxs.append(vmax)

        # Plot every subplot on its own axe
        for subplot_setting in subplots_settings:
            subplot_setting = subplot_setting.copy()
            pos = subplot_setting.pop('pos')
            plotting_func = subplot_setting.pop('func')

            row = (pos-1) // ncols
            col = (pos-1) % ncols

            # Creating the subplot ax, using projection or not depending if it is a map or not
            if fig_is_a_map:
                projection_mode = ccrs.PlateCarree()
                ax = fig.add_subplot(nrows, ncols, pos, projection=projection_mode)
            else:
                ax = fig.add_subplot(nrows, ncols, pos)

            # If the colorbar is at the row scope, adjust subplot scoped colorbar
            if fig_cbar_row:
                soft_override_value(subplot_setting, "vmin", fig_vmins[row])
                soft_override_value(subplot_setting, "vmax", fig_vmaxs[row])
                
                # If it is the last column of the row, displays the colorbar of the row
                if col == ncols-1:
                    soft_add_values(subplot_setting, {"add_cbar": True})
                    soft_override_value(subplot_setting, "cbar_shrink", fig_cbar_fraction)
                    soft_override_value(subplot_setting, "cbar_pad", fig_cbar_pad)
                    soft_override_value(subplot_setting, "cbar_unit", fig_cbar_unit)
                
                else:
                    soft_add_values(subplot_setting, {"add_cbar": False})
            
            # If the colorbar is at the figure scope, disable subplot scoped colorbar
            elif fig_cbar:
                soft_add_values(subplot_setting, {"add_cbar": False})


            # Basic
            soft_override_value(subplot_setting, "vmin", fig_vmin)
            soft_override_value(subplot_setting, "vmax", fig_vmax)
            soft_override_value(subplot_setting, "cmap", fig_cmap)
            soft_override_value(subplot_setting, "fontsize", fig_fontsize)

            # Finally, plotting onto the axe
            plotting_func(fig=fig, ax=ax, **(subplot_setting))
        
        # Add figure title
        if fig_title:
            fig.suptitle(fig_title)

        # If the colorbar is at the figure scope, displays it 
        if fig_cbar and not fig_cbar_row:
            norm = mcolors.Normalize(vmin=fig_vmin, vmax=fig_vmax)
            mappable = cm.ScalarMappable(norm=norm, cmap=fig_cmap)

            opts = dict(
                fraction=fig_cbar_fraction, pad=fig_cbar_pad, orientation=fig_cbar_orientation
            )

            fig_cbar_tick_labels = None
            
            if fig_cbar_ticks:
                if isinstance(fig_cbar_ticks, list) and isinstance(fig_cbar_ticks[0], list):
                    fig_cbar_ticks, fig_cbar_tick_labels = fig_cbar_ticks
                    

                cbar_locator = get_locator_from_ticks(fig_cbar_ticks)
                opts["ticks"] = cbar_locator

            cbar = fig.colorbar(mappable, label=fig_cbar_unit, ax=fig.axes, **opts)

            if fig_cbar_tick_labels:
                cbar.ax.set_xticklabels(fig_cbar_tick_labels)
        
        # Align the y labels
        fig.align_ylabels()

        # If saving the figure is asked
        if save_plot:
            fig.savefig(save_path, format="png")
        
        # If showing the figure is asked
        if show_plots:
            plt.show()
            plt.clf()
            plt.close("all")
    
    # If the figure is not to be shown nor to be saved, returning it
    if not show_plots and not save_plot:
        return fig

def plot_map(
    # Parameter of data
    lon: xr.DataArray, lat: xr.DataArray,
    data: xr.DataArray,

    # Parameter of figure
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    figsize: tuple[int, int] = (10, 6),
    figdpi = 100,

    # Parameter of text
    fontsize: float = 14,
    title: str = "",
    fontsize_title: float = 1.2,

    # Parameters of graph
    extent: list[float] | str | None = "balears",
    aspect: float | None = 1.29, # Best fit
    cmap: str | mcolors.Colormap = 'viridis',
    zero_to_nan: bool = False,
    contours_levels: int | list[float] = None,
    contours_labels_fontsize = 8,
    xticks = None,
    yticks = None,
    bottom_labels = True,
    left_labels = True,

    # Parameter of colorbar
    add_cbar: bool = True,
    cbar_unit: str = "",
    cbar_orientation: str = "vertical",
    cbar_shrink: float = 1,
    cbar_pad: float = 0.005,
    cbar_ticks = None,
    cbar_inversed: bool = False,
    vmin=None, vmax=None,
    vlim = None,
    ylabel = None,
    ylabel_pad = -0.22,

    # Parameter for saving the figure
    save_plot: bool = False,
    save_path: str = "",

    show_plots: bool = False,

    **kwargs
):
    """
    Plots 2D data onto a map with land features using matplotlib's `pcolormesh` function.

    Parameters
    ----------
    lon: xr.DataArray
        DataArray containing the longitude coordinates of the data.

    lat: xr.DataArray,
        DataArray containing the latitude coordinates of the data.
    
    data: xr.DataArray
        DataArray containing the data.
    
    fig: plt.Figure, optional
        If created beforehand, the figure to plot on. Otherwise, a new figure is created.
    
    ax: plt.Axes, optional
        If created beforehand, the axe to plot on. Otherwise, a new axe is created.
    
    figsize: tuple[float, float], default=(10, 6)
        If creating a new figure, (width, height) of the figure in inches.
    
    figdpi: float, default=100   
        The resolution of the figure in dots-per-inch.
    
    fontsize: int, default=14
        Fontsize that will be applied to the plot.
    
    title: str, optional
        Title of the axe.

    fontsize_title: int, default=1.2
        Fontsize that will be applied to the title, relative to the current fontsize.
        Final fontsize of title is `fontsize*fontsize_title`.
    
    extent: list[float] | str | None, default="balears"
        Extent of the map to plot. Can be a list of the shape `[min_lon, max_lon, min_lat, max_lat]`,
        one region of `'med'` or `'balears'`, or None to fit the extent of the data.

    aspect: float | None, default=1.29
        Aspect ratio of the map. A value of 1.29 gives Mercator-like aspect in the Balearic region,
        even if using a `PlateCarree` projection. This enables to avoid projecting to Mercator, as it
        can be computationally-requiring, thus saving some computational resources.
    
    cmap: str | matplotlib.colors.Colormap, default='viridis'
        The colormap to be used when displaying the data. Passed directly to the `pcolormesh` function.

    zero_to_nan: bool, default=False
        Changes every zero values of the data into nans before plotting.
    
    contours_levels: int | list[float], optional
        Option to add contour levels on the map. If given as int, a list of levels is generated using
        `MaxNLocator`, otherwise if given as list, using the list of levels directly.
     

    Returns
    -------
    fig: :class:`matplotlib.figure.Figure`
    ax: :class:`matplotlib.axes.Axes`

    Example
    --------
        import xarray as xr
        from pyscripts.basic_plotting import plot_map

        ds = xr.load_dataset(...)

        plot_map(lon=ds.lon, lon=ds.lat, lon=ds.sst)
    """

    # Apply fontsize globally
    with plt.rc_context({'font.size': fontsize}):
        # Define the projection mode for the map
        projection_mode = ccrs.PlateCarree()

        # If the figure is not configured, initialise a new one
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=figdpi, layout='tight')
        
        # If the axe is not configured, initialise a new one
        if ax is None:
            ax = fig.add_subplot(111, projection=projection_mode)

        # Apply region selector
        if extent == "med":
            extent = [-6, 36.33, 30, 46]
        elif extent == "balears":
            extent = [-1, 5, 37.7, 41]

        # Defines the plot extent
        if extent:
            ax.set_extent(extent, crs=projection_mode)

        # Defines the plot aspect
        if aspect:
            ax.set_aspect(aspect) # This is a trick for performance sake, giving faster results, it imitates Mercator projection in 10sec vs 60sec
        
        # Change the grid settings and the coordinates labels
        gl = ax.gridlines(linestyle=':', draw_labels=True, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = bottom_labels
        gl.left_labels = left_labels
        
        # Define the x ticks
        if xticks:
            xlocator = get_locator_from_ticks(xticks)
            gl.xlocator = xlocator
        
        # Define the y ticks
        if yticks:
            ylocator = get_locator_from_ticks(yticks)
            gl.ylocator = ylocator

        # Option for plotting
        pcm_opts = {
            "transform": projection_mode,
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
            
            # Extra
            # "shading": "nearest"
            **kwargs
        }
        
        # Changes 0 into nans if asked
        if zero_to_nan:
            data = data.where(data != 0, np.nan)

        # If vlims are given into tuple, split it
        if vlim:
            vmin, vmax = vlim

        # if vmin is None:
        #     vmin = None if np.isnan(data).all() else np.nanmin(data)
        
        # if vmax is None:
        #     vmax = None if np.isnan(data).all() else np.nanmax(data)

        # Plotting the data!!
        pcm = ax.pcolormesh(lon, lat, data, zorder=-1, shading="auto", **pcm_opts)
        ax.set_title(title, fontsize=fontsize*fontsize_title)

        # If ylabel is asked, adding it as so
        if ylabel:
            ax.text(ylabel_pad, 0.55, ylabel, va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor', fontsize=fontsize*fontsize_title,
                transform=ax.transAxes)

        # Shows the isolines on the plots
        if contours_levels:
            if isinstance(contours_levels, int):
                if vmin is None:
                    vmin = np.nanmin(data)
                if vmax is None:
                    vmax = np.nanmax(data)

                # If the contour level is a int, generate nice contour levels using MaxNLocator
                locator = MaxNLocator(contours_levels)
                contours_levels = locator.tick_values(vmin, vmax)
            
            contours = ax.contour(lon, lat, data, levels=contours_levels, colors='k', linewidths=1, alpha=0.6, transform=projection_mode, zorder=-1)
            ax.clabel(contours, levels=contours_levels, colors='k', fontsize=contours_labels_fontsize, zorder=-1)

        # Add colorbar
        if add_cbar:
            opts = dict(
                label=cbar_unit, orientation=cbar_orientation, shrink=cbar_shrink, pad=cbar_pad,
            )
            
            if cbar_ticks:
                cbar_locator = get_locator_from_ticks(cbar_ticks)
                opts["ticks"] = cbar_locator

            cbar = fig.colorbar(pcm, ax=ax, **opts)

            if cbar_inversed:
                cbar.ax.invert_yaxis()


        # Additions of coastlines and land
        ax.coastlines(resolution="10m")       # Coastlines
        ax.add_feature(cfeature.LAND)        # Land
        # ax.add_feature(cfeature.BORDERS, alpha=0.2)

        # If saving the figure is asked
        if save_plot:
            fig.savefig(save_path, format="png")
        
        # If showing the figure is asked
        if show_plots:
            plt.show()
            plt.clf()
            plt.close("all")
    
    # If the figure is not to be shown nor to be saved, returning it
    if not show_plots and not save_plot:
        return fig, ax

def plot_transect(
    # Parameter of data
    abscissa, depth,
    variable_data,

    along_lon = False,
    along_lat = False,
    abscissa_is_time = False,

    # Parameter of figure
    fig=None,figsize=(10,6),figdpi=100,
    ax: plt.Axes = None,

    # Parameter of text
    fontsize: int = 14,
    title: str="",
    show_pos=True,

    # Parameter of colorbar
    add_cbar: bool=True,
    cbar_unit: str="",
    cbar_orientation: str="vertical",
    cbar_shrink: float=1,
    cbar_pad: float = 0.005,
    cbar_ticks=None,
    cbar_inversed: bool=False,
    vmin=None, vmax=None,

    bottom_labels = True,
    left_labels = True,

    # Parameters of graph
    cmap="managua",
    zero_to_nan = False,
    norm="linear",
    contours_levels=None,
    contours_labels_fontsize=8,
    xticks=None,
    yticks=None,

    # Parameter for saving the figure
    save_plot: bool=False,
    save_path: str="",
    dpi: int=160,

    show_plots: bool = False,
):
    """
    Creates a transect

    """

    # Sometimes, data is given transposed, then transposing it makes it work
    if variable_data.shape != (len(depth), len(abscissa)):
        variable_data = variable_data.T

    # Apply fontsize globally
    with plt.rc_context({'font.size': fontsize}):
        # If the figure is not configured, initialise a new one
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=figdpi, layout='tight')
        
        # If the axe is not configured, initialise a new one
        if ax is None:
            ax = fig.add_subplot(111)

        # Change the grid settings and the coordinates labels
        ax.grid(True, ls=':', alpha=0.7)
        
        # Define the x ticks
        if xticks:
            xlocator = get_locator_from_ticks(xticks)
            ax.xaxis.set_major_locator(xlocator)
        
        # Define the y ticks
        if yticks:
            ylocator = get_locator_from_ticks(yticks)
            ax.yaxis.set_major_locator(ylocator)

        # Option for plotting
        pcm_opts = {
            "cmap": cmap,
            "norm": norm,
            
            # Extra
            # "shading": "nearest"
        }

        # Changes 0 into nans if asked
        if zero_to_nan:
            variable_data = variable_data.where(variable_data != 0, np.nan)

        # if vmin is None:
        #     vmin = None if np.isnan(variable_data).all() else np.nanmin(variable_data)
        
        # if vmax is None:
        #     vmax = None if np.isnan(variable_data).all() else np.nanmax(variable_data)
        #     vmin, vmax = nice_range(vmin, vmax)

        # Plotting the data!!
        pcm = ax.pcolormesh(abscissa, depth, variable_data, shading="auto", vmin=vmin, vmax=vmax, **pcm_opts)
        ax.set_title(title)

        # Invert the y axis, as depth should be given in positive values
        ax.invert_yaxis()
        
        # Format the depth values to include meters
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f} m"))
        
        # If plotting along an horizontal basis
        if not abscissa_is_time:
            # If abscissa is given in longitudinal degrees
            if along_lon:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x)}°{'W' if x < 0 else '' if x == 0 else 'E'}"))

            # If abscissa is given in latitudinal degrees
            elif along_lat:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x)}°{'S' if x < 0 else '' if x == 0 else 'N'}"))

            # Otherwise, abscissa should be given in km
            else:
                ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}km"))
            
            if along_lat and show_pos:
                lat0 = variable_data.lat.isel(transect=0, depth=0).item()
                lat1 = variable_data.lat.isel(transect=variable_data.transect.size-1, depth=0).item()
                ax.text(
                    0.01, 0.01,
                    f"{abs(lat0):.1f}°{'S' if lat0 < 0 else '' if lat0 == 0 else 'N'}",
                    ha='left', va='bottom', transform=ax.transAxes
                )
                ax.text(
                    0.99, 0.01,
                    f"{abs(lat1):.1f}°{'S' if lat1 < 0 else '' if lat1 == 0 else 'N'}",
                    ha='right', va='bottom', transform=ax.transAxes
                )
            elif show_pos:
                lon0 = variable_data.lon.isel(transect=0, depth=0).item()
                lon1 = variable_data.lon.isel(transect=variable_data.transect.size-1, depth=0).item()
                ax.text(
                    0.01, 0.01,
                    f"{abs(lon0):.1f}°{'W' if lon0 < 0 else '' if lon0 == 0 else 'E'}",
                    ha='left', va='bottom', transform=ax.transAxes
                )
                ax.text(
                    0.99, 0.01,
                    f"{abs(lon1):.1f}°{'W' if lon1 < 0 else '' if lon1 == 0 else 'E'}",
                    ha='right', va='bottom', transform=ax.transAxes
                )

        # Hide labels on axis if not necessary
        if not bottom_labels:
            ax.set_xticklabels([])
        if not left_labels:
            ax.set_yticklabels([])

        # Shows the isolines on the plots
        if contours_levels:
            if isinstance(contours_levels, int):
                if vmin is None:
                    vmin = np.nanmin(variable_data)
                if vmax is None:
                    vmax = np.nanmax(variable_data)

                # If the contour level is a int, generate nice contour levels using MaxNLocator
                locator = MaxNLocator(contours_levels)
                contours_levels = locator.tick_values(vmin, vmax)
            
            contours = ax.contour(abscissa, depth, variable_data, levels=contours_levels, colors='k', linewidths=1, alpha=0.6)
            ax.clabel(contours, levels=contours_levels, colors='k', fontsize=contours_labels_fontsize)


        # Add colorbar
        if add_cbar:
            opts = dict(
                label=cbar_unit, orientation=cbar_orientation, shrink=cbar_shrink, pad=cbar_pad,
            )

            if cbar_ticks:
                cbar_locator = MaxNLocator(nbins=cbar_ticks)
                opts["ticks"] = cbar_locator

            cbar = fig.colorbar(pcm, ax=ax, **opts)

            if cbar_inversed:
                cbar.ax.invert_yaxis()

        # If saving the figure is asked
        if save_plot:
            fig.savefig(save_path, format="png", dpi=dpi)
        
        # If showing the figure is asked
        if show_plots:
            plt.show()
            plt.clf()
            plt.close("all")
    
    # If the figure is not to be shown nor to be saved, returning it
    if not show_plots and not save_plot:
        return fig, ax

def plot_vertical_mean(
        # Parameter of data
        depths: dict[str, xr.DataArray] | xr.DataArray,
        vars: dict[str, xr.DataArray] | xr.DataArray,
        # labels: dict[str, str] | str | None = 'auto',
        colors: dict[str, str] | str | None = None,
        ls: dict[str, str] | str | None = None,
        nans_to_zero: bool = False,

        # Parameter of figure
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple[int, int] = (18, 5),
        figdpi = 100,

        # Parameter of text
        fontsize: int = 14,
        title: str | None = None,
        unit: str | None = None,

        # Parameters of graph
        grid: bool = True,
        xlim: tuple[int|None, int|None] = (None, None),
        ylim: tuple[int|None, int|None] = (None, None),
        xticks = None,
        xticks_minor = None,
        yticks = None,
        yticks_minor = None,
        xticks_formatter = None,
        bottom_labels = True,
        left_labels = True,
        top_labels = False,
        legend: bool = True,

        # Parameter for saving the figure
        show_plots: bool = False,
        save_plot: bool = False,
        save_path: str = "",

        **kwargs
):
    """
    Plots a time serie with the defined settings.

    Parameters
    ----------
    vars: dict[str, xr.DataArray]
        Dictionary attributing to a name a data array representing the variable to plot.
    
    times: dict[str, xr.DataArray]
        Dictionary attributing to a name a data array representing the time array associated with the variable to plot.
    
    vars_stds: dict[str, xr.DataArray] | None, default=None
        Dictionary attributing to a name a data array representing the std array associated with the variable to plot.
    
    labels: dict[str, str] | None, default=None
        Dictionary attributing to a name the label to assign to the variable to plot.
    
    colors: dict[str, str] | None, default=None
        Dictionary attributing to a name the color to assign to the variable to plot.
    
        
    fig: plt.Figure | None, default=None
        Figure to use if previously created. None value will create a new figure only if ax is None.
        If ax is not None, no new figure will be created.
    
    ax: plt.Axes | None, default=None
        Axes to use if previously created. None value will create a new axes.
    
    figsize: tuple[int, int], default=(18, 5)
        Figure size to use if creating a new figure.
    
    
    fontsize: int = 14
        Font size to be used in the plot.

    title: str|None = None,
    unit: str|None = None,

    # Axe options
    grid: bool = True,
    xlim: tuple = (None, None),
    ylim: tuple = (None, None),

    # Saving options
    save_plot: bool = False,
    save_path: str = "",

    years: list[int|str], default=["*"], optional
        Years to load from the dataset, as integers or strings (e.g., range(1983, 1985) or ["1983"]).
        Use ["*"] to load all available years (1982-2023) as it uses a glob pattern.
    
    months: list[int|str], default=["*"], optional
        Months to load from the dataset, as integers or strings (e.g., range(1, 5) or ["1"]).
        Use ["*"] to load all months (1-12) as it uses a glob pattern.
    
    time_selector: str | slice[str] | None, default=None, optional
        Time selection applied using xarray's `.sel()`. Can be a string (e.g., "1993-01-21") or a slice
        (e.g., slice("1993-01-21", "1993-01-25")). If None, no time filtering is applied.
    
    lon_selector: float | slice[float] | None, default=None, optional
        Longitude selector applied using xarray's `.sel()`. Accepts a float or a slice.
    
    lat_selector: float | slice[float] | None, default=None, optional
        Latitude selector applied using xarray's `.sel()`. Accepts a float or a slice.
    
    only_sst: bool, default=False, optional
        If True, all other variables than SST will be discarded.
    
    Returns
    ----------
    ds_cmems_sst: xr.Dataset
        The loaded CMEMS-SST dataset with optional spatio-temporal subsetting.
    """

    # Apply fontsize globally
    with plt.rc_context({'font.size': fontsize}):
        # If the figure is not configured, initialise a new one
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=figdpi, layout='tight')
        
        # If the axe is not configured, initialise a new one
        if ax is None:
            ax = fig.add_subplot(111)
        
        # Should handle single dataset plotting
        if isinstance(vars, dict):
            for var in vars:
                opts_args = {}

                # Apply the required color
                if colors:
                    if isinstance(colors, dict) and var in colors:
                        opts_args["color"] = colors[var]
                    elif not isinstance(colors, dict):
                        opts_args["color"] = colors
                
                # Apply the required line style
                if ls:
                    if isinstance(ls, dict) and var in ls:
                        opts_args["ls"] = ls[var]
                    elif not isinstance(ls, dict):
                        opts_args["ls"] = ls
                
                # Apply labels
                # if labels == 'auto':
                #     opts_args["label"] = var
                # elif labels and var in labels:
                #     opts_args["label"] = labels[var]
                
                # Apply nan filter
                if nans_to_zero:
                    vars[var] = np.nan_to_num(vars[var])
                
                # Choose right time array
                if isinstance(depths, dict):
                    depths_ = depths[var]
                else:
                    depths_ = depths
                
                # Finally, plot the line
                ax.plot(vars[var], depths_, **opts_args)
        
        else:
            if unit == None and isinstance(vars, xr.DataArray) and vars.attrs.get("unit"):
                unit = vars.attrs.get("unit")

            # if labels == 'auto': labels=None

            if nans_to_zero:
                vars = np.nan_to_num(vars)

            ax.plot(depths, vars, color=colors, ls=ls, lw=1)  

        
        if ylim == (None, None):
            if isinstance(depths, dict):
                ylim = (
                    np.nanmin([np.nanmin(depths[name]) for name in depths]),
                    np.nanmax([np.nanmax(depths[name]) for name in depths])
                )
            
            else:
                ylim = (np.nanmin(depths), np.nanmax(depths))
        
        # Set manually the abscissa ticks
        if xticks:
            xlocator = get_locator_from_ticks(xticks)
            ax.xaxis.set_major_locator(xlocator)

        if xticks_minor:
            xlocator = get_locator_from_ticks(xticks_minor, which="minor")
            ax.xaxis.set_minor_locator(xlocator)

        if xticks_formatter:
            ax.xaxis.set_major_formatter(xticks_formatter)
        
        # Set manually the ordinate ticks
        if yticks:
            ylocator = get_locator_from_ticks(yticks)
            ax.yaxis.set_major_locator(ylocator)

        if yticks_minor:
            ylocator = get_locator_from_ticks(yticks_minor, which="minor")
            ax.yaxis.set_minor_locator(ylocator)
        

        # if not bottom_labels:
        #     ax.set_xticklabels([])
        
        # if not left_labels:
        #     ax.set_yticklabels([])
        # else:
        if left_labels:
            ax.set_ylabel("Depth [m]")

        ax.tick_params(which='both', top=top_labels, labeltop=top_labels, bottom=bottom_labels, labelbottom=bottom_labels, left=left_labels, labelleft=left_labels)

        # Change the ax color
        # for spine in ax.spines.values():
        #     spine.set_edgecolor('green')

        ax.set_title(title)
        if unit:
            ax.set_xlabel(f"[{unit}]")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.gca().invert_yaxis()

        if grid: ax.grid(alpha=0.5)
        if grid: ax.grid(which="minor", alpha=0.3, ls="--")
        # if labels and legend: ax.legend()

        if show_plots:
            plt.show()
            plt.clf()
            plt.close("all")
    
    return fig, ax

def plot_timeserie(
        # Parameter of data
        times: dict[str, xr.DataArray] | xr.DataArray,
        vars: dict[str, xr.DataArray] | xr.DataArray,
        vars_stds: dict[str, xr.DataArray] | xr.DataArray | None = None,
        labels: dict[str, str] | str | None = 'auto',
        colors: dict[str, str] | str | None = None,
        ls: dict[str, str] | str | None = None,
        nans_to_zero: bool = False,

        # Parameter of figure
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple[int, int] = (18, 5),
        figdpi = 100,

        # Parameter of text
        fontsize: int = 14,
        title: str | None = None,
        ylabel = None,

        # Parameters of graph
        grid: bool = True,
        xlim: tuple[int|None, int|None] = (None, None),
        ylim: tuple[int|None, int|None] = (None, None),
        xticks = None,
        xticks_minor = None,
        yticks = None,
        yticks_minor = None,
        yticks_formatter = None,
        bottom_labels = True,
        left_labels = True,
        legend: bool | dict = True,

        # Parameter for saving the figure
        show_plots: bool = False,
        save_plot: bool = False,
        save_path: str = "",

        **kwargs
):
    """
    Plots a time serie with the defined settings.

    Parameters
    ----------
    vars: dict[str, xr.DataArray]
        Dictionary attributing to a name a data array representing the variable to plot.
    
    times: dict[str, xr.DataArray]
        Dictionary attributing to a name a data array representing the time array associated with the variable to plot.
    
    vars_stds: dict[str, xr.DataArray] | None, default=None
        Dictionary attributing to a name a data array representing the std array associated with the variable to plot.
    
    labels: dict[str, str] | None, default=None
        Dictionary attributing to a name the label to assign to the variable to plot.
    
    colors: dict[str, str] | None, default=None
        Dictionary attributing to a name the color to assign to the variable to plot.
    
        
    fig: plt.Figure | None, default=None
        Figure to use if previously created. None value will create a new figure only if ax is None.
        If ax is not None, no new figure will be created.
    
    ax: plt.Axes | None, default=None
        Axes to use if previously created. None value will create a new axes.
    
    figsize: tuple[int, int], default=(18, 5)
        Figure size to use if creating a new figure.
    
    
    fontsize: int = 14
        Font size to be used in the plot.

    title: str|None = None,
    unit: str|None = None,

    # Axe options
    grid: bool = True,
    xlim: tuple = (None, None),
    ylim: tuple = (None, None),

    # Saving options
    save_plot: bool = False,
    save_path: str = "",

    years: list[int|str], default=["*"], optional
        Years to load from the dataset, as integers or strings (e.g., range(1983, 1985) or ["1983"]).
        Use ["*"] to load all available years (1982-2023) as it uses a glob pattern.
    
    months: list[int|str], default=["*"], optional
        Months to load from the dataset, as integers or strings (e.g., range(1, 5) or ["1"]).
        Use ["*"] to load all months (1-12) as it uses a glob pattern.
    
    time_selector: str | slice[str] | None, default=None, optional
        Time selection applied using xarray's `.sel()`. Can be a string (e.g., "1993-01-21") or a slice
        (e.g., slice("1993-01-21", "1993-01-25")). If None, no time filtering is applied.
    
    lon_selector: float | slice[float] | None, default=None, optional
        Longitude selector applied using xarray's `.sel()`. Accepts a float or a slice.
    
    lat_selector: float | slice[float] | None, default=None, optional
        Latitude selector applied using xarray's `.sel()`. Accepts a float or a slice.
    
    only_sst: bool, default=False, optional
        If True, all other variables than SST will be discarded.
    
    Returns
    ----------
    ds_cmems_sst: xr.Dataset
        The loaded CMEMS-SST dataset with optional spatio-temporal subsetting.
    """

    # min_date = np.min([ds_vars["time"][0] for ds_vars in datasets_vars])
    # max_date = np.min([ds_vars["time"][-1] for ds_vars in datasets_vars])

    with plt.rc_context({'font.size': fontsize}):
        # Plotting
        if fig is None or ax is None:
            fig = plt.figure(figsize=figsize, dpi=figdpi, layout='tight')
            ax = fig.add_subplot(111)
        
        # Should handle single dataset plotting
        if isinstance(vars, dict):
            first_var = vars[next(iter(vars))]

            # if unit == None and isinstance(first_var, xr.DataArray) and first_var.attrs.get("unit"):
            #     unit = first_var.attrs.get("unit")
            
            for var in vars:
                opts_args = {}

                # Apply the required color
                if colors:
                    if isinstance(colors, dict) and var in colors:
                        opts_args["color"] = colors[var]
                    elif not isinstance(colors, dict):
                        opts_args["color"] = colors
                
                # Apply the required line style
                if ls:
                    if isinstance(ls, dict) and var in ls:
                        opts_args["ls"] = ls[var]
                    elif not isinstance(ls, dict):
                        opts_args["ls"] = ls
                
                # Apply labels
                if labels == 'auto':
                    opts_args["label"] = var
                elif labels and var in labels:
                    opts_args["label"] = labels[var]
                
                # Apply nan filter
                if nans_to_zero:
                    vars[var] = np.nan_to_num(vars[var])
                
                # Choose right time array
                if isinstance(times, dict):
                    times_ = times[var]
                else:
                    times_ = times
                
                # Finally, plot the line
                ax.plot(times_, vars[var], lw=1, **opts_args)
        
        else:
            # if unit == None and isinstance(vars, xr.DataArray) and vars.attrs.get("unit"):
            #     unit = vars.attrs.get("unit")

            if labels == 'auto': labels=None

            if nans_to_zero:
                vars = np.nan_to_num(vars)

            ax.plot(times, vars, color=colors, label=labels, ls=ls, lw=1)  

        
        if xlim == (None, None):
            if isinstance(times, dict):
                xlim = (
                    np.nanmin([np.nanmin(times[name]) for name in times]),
                    np.nanmax([np.nanmax(times[name]) for name in times])
                )
            
            else:
                xlim = (np.nanmin(times), np.nanmax(times))
        
        # Set manually the abscissa ticks
        if xticks:
            xlocator = get_locator_from_ticks(xticks)
            ax.xaxis.set_major_locator(xlocator)

        if xticks_minor:
            xlocator = get_locator_from_ticks(xticks_minor, which="minor")
            ax.xaxis.set_minor_locator(xlocator)
        
        # Set manually the ordinate ticks
        if yticks:
            ylocator = get_locator_from_ticks(yticks)
            ax.yaxis.set_major_locator(ylocator)

        if yticks_minor:
            ylocator = get_locator_from_ticks(yticks_minor, which="minor")
            ax.yaxis.set_minor_locator(ylocator)

        if yticks_formatter:
            ax.yaxis.set_major_formatter(yticks_formatter)
        

        if not bottom_labels:
            ax.set_xticklabels([])
        
        if not left_labels:
            ax.set_yticklabels([])

        # Change the ax color
        # for spine in ax.spines.values():
        #     spine.set_edgecolor('green')

        ax.set_title(title)
        if ylabel:
            ax.set_ylabel(ylabel)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if grid: ax.grid(alpha=0.5)
        if grid: ax.grid(which="minor", alpha=0.3, ls="--")
        if labels and legend:
            if isinstance(legend, dict):
                ax.legend(**legend)

            else:
                ax.legend()

        if show_plots:
            plt.show()
            plt.clf()
            plt.close("all")
    
    return fig, ax


def plot_mhw_timeserie():
    pass


def plot_bars(
        # Parameter of data
        depths: xr.DataArray,
        vars: dict[str, xr.DataArray],
        stds: dict[str, xr.DataArray] = None,
        # labels: dict[str, str] | str | None = 'auto',
        colors: dict[str, str] | str = None,
        ls: dict[str, str] | str = None,
        hatch: dict[str, str] | str = None,
        nans_to_zero: bool = False,

        # Parameter of figure
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple[int, int] = (10, 5),
        figdpi = 100,

        # Parameter of text
        fontsize: int = 14,
        title: str | None = None,
        unit: str | None = None,

        # Parameters of graph
        grid: bool = True,
        xlim: tuple[int|None, int|None] = (None, None),
        xticks = None,
        xticks_minor = None,
        yticks = None,
        yticks_minor = None,
        xticks_formatter = None,
        bottom_labels = True,
        left_labels = True,
        top_labels = False,
        legend: bool = True,
        ylabel = None,
        bars_pad = 2,
        texts: list[dict] = [],

        # Parameter for saving the figure
        show_plots: bool = False,
        save_plot: bool = False,
        save_path: str = "",

        **kwargs
):
    
    if not colors:
        colors = {
            var: f'C{i}'
            for i, var in enumerate(vars)
        }

    
    # Apply fontsize globally
    with plt.rc_context({'font.size': fontsize}):
        # If the figure is not configured, initialise a new one
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=figdpi, layout='tight')
        
        # If the axe is not configured, initialise a new one
        if ax is None:
            ax = fig.add_subplot(111)
        
        last_tick = 0
        ticks_pos = []

        should_define_xlim = xlim == (None, None)

        for depth in depths:
            vars_displayed = 0

            for var in vars:
                value = vars[var].sel(depth=depth)
                
                # If all nan, return
                if value.isnull().all():
                    continue

                opts_args = {}

                # Apply the required color
                if colors:
                    if isinstance(colors, dict) and var in colors:
                        opts_args["color"] = colors[var]
                    elif not isinstance(colors, dict):
                        opts_args["color"] = colors
                
                # Apply the required line style
                if ls:
                    if isinstance(ls, dict) and var in ls:
                        opts_args["ls"] = ls[var]
                    elif not isinstance(ls, dict):
                        opts_args["ls"] = ls
                
                # Apply the required line style
                if hatch:
                    if isinstance(hatch, dict) and var in hatch:
                        if isinstance(hatch[var], dict) and depth in hatch[var]:
                            opts_args["hatch"] = hatch[var][depth]
                        else:
                            opts_args["hatch"] = hatch[var]
                        
                    elif not isinstance(hatch, dict):
                        opts_args["hatch"] = hatch
                
                # Apply labels
                # if labels == 'auto':
                #     opts_args["label"] = var
                # elif labels and var in labels:
                #     opts_args["label"] = labels[var]
                
                # Apply nan filter
                if nans_to_zero:
                    vars[var] = np.nan_to_num(vars[var])
                
                # Choose right time array
                # if isinstance(depths, dict):
                #     depths_ = depths[var]
                # else:
                #     depths_ = depths

                if stds:
                    std = stds[var].sel(depth=depth)
                    opts_args['xerr'] = std
                    # opts_args['ecolor'] = 'k'
                    opts_args['capsize'] = 3
                    opts_args['error_kw'] = {
                        'alpha': 0.4,
                    }
                
                # Finally, plot the line
                # ax.plot(vars[var], depths_, **opts_args)
                a=ax.barh(
                    last_tick + vars_displayed + 0.5,
                    value,
                    **opts_args
                )

                for bc in a:
                    import matplotlib.colors as c
                    bc._hatch_color = c.to_rgba("w")
                    bc.stale = True


                # if stds:
                #     ax.barh(
                #         last_tick + vars_displayed + 0.5,
                #         std,
                #         **opts_args
                #     )

                if should_define_xlim:
                    if xlim == (None, None):
                        xlim = (np.nanmin(value), np.nanmax(value))
                    else:
                        xlim = (min(np.nanmin(value), xlim[0]), max(np.nanmax(value), xlim[1]))
                    
                    if stds:
                        std = stds[var].sel(depth=depth)
                        xlim = (min(np.nanmin(value-std), xlim[0]), max(np.nanmax(value+std), xlim[1]))

                vars_displayed += 1

            ticks_pos.append(last_tick + vars_displayed/2)
            last_tick += vars_displayed + bars_pad
        
        if should_define_xlim:
            vmin, vmax = xlim
            pad = (vmax - vmin)*0.1

            vmin = 0 if vmin*(vmin-pad)<=0 else vmin-pad
            vmax = 0 if vmax*(vmax+pad)<=0 else vmax+pad

            xlim = (vmin, vmax)
        
        # if ylim == (None, None):
        #     if isinstance(depths, dict):
        #         ylim = (
        #             np.nanmin([np.nanmin(depths[name]) for name in depths]),
        #             np.nanmax([np.nanmax(depths[name]) for name in depths])
        #         )
            
        #     else:
        #         ylim = (np.nanmin(depths), np.nanmax(depths))
        
        # Set manually the abscissa ticks
        if xticks:
            xlocator = get_locator_from_ticks(xticks)
            ax.xaxis.set_major_locator(xlocator)

        if xticks_minor:
            xlocator = get_locator_from_ticks(xticks_minor, which="minor")
            ax.xaxis.set_minor_locator(xlocator)

        if xticks_formatter:
            ax.xaxis.set_major_formatter(xticks_formatter)
        
        # Set manually the ordinate ticks
        # if yticks:
        #     ylocator = get_locator_from_ticks(yticks)
        #     ax.yaxis.set_major_locator(ylocator)

        # if yticks_minor:
        #     ylocator = get_locator_from_ticks(yticks_minor, which="minor")
        #     ax.yaxis.set_minor_locator(ylocator)
        
        ticks_pos = np.array(ticks_pos)

        idx = range(len(ticks_pos))

        if isinstance(yticks, int):
            idx = range(0, len(ticks_pos), yticks)

        ax.set_yticks(ticks_pos, minor=True)
        ax.set_yticks(ticks_pos[idx], labels=[f"{depth:.0f} m" for depth in depths[idx]])


        # if not bottom_labels:
        #     ax.set_xticklabels([])
        
        # if not left_labels:
        #     ax.set_yticklabels([])
        # else:
        # if left_labels:
        #     ax.set_ylabel("Depth [m]")

        ax.tick_params(which='both', top=top_labels, labeltop=top_labels, bottom=bottom_labels, labelbottom=bottom_labels, left=left_labels, labelleft=left_labels)
        ax.tick_params(which='both', length=0)

        # Change the ax color
        # for spine in ax.spines.values():
        #     spine.set_edgecolor('green')

        ax.set_title(title)
        if unit:
            ax.set_xlabel(f"[{unit}]")
        
        ax.set_ylabel(ylabel)

        ax.set_xlim(xlim)
        # ax.set_ylim(-bars_pad, last_tick)
        ax.set_ylim(-bars_pad/2, last_tick-bars_pad/2)
        ax.invert_yaxis()


        if grid: ax.grid(alpha=0.5)
        if grid: ax.grid(which="minor", alpha=0.3, ls="--")
        # if labels and legend: ax.legend()

        for text in texts:
            ax.text(transform=ax.transAxes, **text)

        if show_plots:
            plt.show()
            plt.clf()
            plt.close("all")
    
    return fig, ax

def add_colorbar(
    fig: plt.Figure,
    ax,
    mappable,

    # Parameter of colorbar
    cbar_unit: str = None,
    cbar_orientation: str = "vertical",
    cbar_shrink: float = None,
    cbar_fraction: float = None,
    cbar_pad: float = None,
    cbar_ticks = None,
    cbar_formatter = None,
    cbar_inversed: bool = False,
    **kwargs
):
    opts = dict(
        orientation=cbar_orientation, shrink=cbar_shrink, pad=cbar_pad, fraction=cbar_fraction
    )

    if cbar_unit:
        opts["label"] = f"[{cbar_unit}]"
    
    if cbar_formatter:
        opts["format"] = cbar_formatter

    if cbar_ticks:
        cbar_locator = get_locator_from_ticks(cbar_ticks)
        opts["ticks"] = cbar_locator

    cbar = fig.colorbar(mappable, ax=ax, **opts, **kwargs)

    if cbar_inversed:
        cbar.ax.invert_yaxis()
    

def get_locator_from_ticks(ticks, which="major"):
    if isinstance(ticks, int):
        if which == "major":
            return MaxNLocator(nbins=ticks)
        elif which == "minor":
            return AutoMinorLocator(ticks)
    
    elif isinstance(ticks, tuple):
        if len(ticks) == 2:
            return MultipleLocator(base=ticks[0], offset=ticks[1])
        else:
            return MultipleLocator(base=ticks[0])
    
    elif isinstance(ticks, list):
        return FixedLocator(ticks)

    return MaxNLocator()
