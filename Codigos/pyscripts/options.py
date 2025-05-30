########################################################################################################################
########################################################################################################################

###### USER NOTES:
# This script gathers some options used along the codes of this repository. 

########################################################################################################################
##################################### USER INPUT ##########################################################################
########################################################################################################################


import matplotlib.colors as mcolors
import numpy as np

    # Maps options

# CURRENTLY NOT USED IN CODE
extents = {
    "balears": [-1, 5, 37.7, 41],
    "med": [-6, 36.33, 30, 46],
}


    # MHWs statistics 

# Basic MHWs statistics
mhws_basic_stats = [
    'total_days',
    'duration',
    'total_icum',
    'intensity_max_max',
    'intensity_mean_byday',
    'severity_mean_byday',
]

# All MHWs statistics
mhws_stats = [
    # Counts of events/days
    'count',
    'total_days',
    'moderate_days',
    'strong_days',
    'severe_days',
    'extreme_days',

    # Duration statistic
    'duration',

    # Cumulative intensity statistics
    'total_icum',
    'intensity_cumulative',

    # Intensity statistics
    'intensity_max_max',
    'intensity_max',
    'intensity_mean',
    'intensity_mean_byday',
    'intensity_var',

    # Cumulative intensity statistics
    'total_scum',
    'severity_cumulative',

    # Intensity statistics
    'severity_max_max',
    'severity_max',
    'severity_mean',
    'severity_mean_byday',
    'severity_var',

    # Rate onset/decline statistics
    'rate_onset',
    'rate_decline',

    # Temperature statistics
    'temp_min',
    'temp_mean',
    'temp_max',
    'mean_thresh',
]

# Short names for MHWs statistics
mhws_stats_shortname = {
    "count":                "Annual MHW events",
    "total_days":           "Total days",
    "moderate_days":        "Annual moderate MHW days",
    "strong_days":          "Annual strong MHW days",
    "severe_days":          "Annual severe MHW days",
    "extreme_days":         "Annual extreme MHW days",

    "duration":             "Mean duration",

    "total_icum":           "Cumulative intensity",
    "intensity_cumulative": "Mean MHW cumulative intensity",

    "intensity_max_max":    "Maximum intensity",
    "intensity_max":        "Mean MHW maximum intensity",
    "intensity_mean":       "Mean MHW event intensity",
    'intensity_mean_byday': "Mean intensity",
    "intensity_var":        "Mean MHW intensity variability",

    'total_scum':           "Annual MHW cumulative severity",
    'severity_cumulative':  "Mean MHW cumulative severity",

    'severity_max_max':     "Maximum MHW severity",
    'severity_max':         "Mean MHW maximum severity",
    'severity_mean':        "Mean MHW event severity",
    'severity_mean_byday':  "Mean severity",
    'severity_var':         "Mean MHW severity variability",

    "rate_onset":           "Mean MHW onset rate",
    "rate_decline":         "Mean MHW decline rate",

    "temp_min":             "Minimum temperature",
    "temp_mean":            "Annual mean temperature",
    "temp_max":             "Maximal temperature",
    "mean_thresh":          "Mean 90th percentile",
}

# Long names for MHWs statistics (from Oliver's code)
mhws_stats_longname = {
    "count":                "Total MHW count per year",
    "total_days":           "Total number of MHW days per year",
    "moderate_days":        "Total number of moderate MHW days per year",
    "strong_days":          "Total number of strong MHW days per year",
    "severe_days":          "Total number of severe MHW days per year",
    "extreme_days":         "Total number of extreme MHW days per year",

    "duration":             "Average MHW duration per year",

    "total_icum":           "Total cumulative intensity over all MHWs per year",
    "intensity_cumulative": "Average MHW \"cumulative intensity\" per year",

    "intensity_max_max":    "Maximum MHW \"maximum (peak) intensity\" per year",
    "intensity_max":        "Average MHW \"maximum (peak) intensity\" per year",
    "intensity_mean":       "Average MHW event \"mean intensity\" per year",
    'intensity_mean_byday': "Average MHW day \"mean intensity\" per year",
    "intensity_var":        "Average MHW \"intensity variability\" per year",

    'total_scum':           "Annual MHW cumulative severity",
    'severity_cumulative':  "Mean MHW cumulative severity",

    'severity_max_max':     "Maximum MHW severity",
    'severity_max':         "Mean MHW maximum severity",
    'severity_mean':        "Mean MHW event severity",
    'severity_mean_byday':  "Mean MHW day severity",
    'severity_var':         "Mean MHW severity variability",

    "rate_onset":           "Average MHW onset rate per year",
    "rate_decline":         "Average MHW decline rate per year",

    "temp_min":             "Minimum temperature per year",
    "temp_mean":            "Mean temperature per year",
    "temp_max":             "Maximum temperature per year",
    "mean_thresh":          "Mean 90th threshold",
}

# MHWs statistics units
mhws_stats_units = {
    "count":                "count",
    "total_days":           "days",
    "moderate_days":        "days",
    "strong_days":          "days",
    "severe_days":          "days",
    "extreme_days":         "days",

    "duration":             "days",

    "total_icum":           "°C.days",
    "intensity_cumulative": "°C.days",

    "intensity_max_max":    "°C",
    "intensity_max":        "°C",
    "intensity_mean":       "°C",
    'intensity_mean_byday': "°C",
    "intensity_var":        "°C",

    'total_scum':           "Severity Index.day",
    'severity_cumulative':  "Severity Index.day",

    'severity_max_max':     "Category",
    'severity_max':         "Category",
    'severity_mean':        "Category",
    'severity_mean_byday':  "Category",
    'severity_var':         "Category",

    "rate_onset":           "°C/days",
    "rate_decline":         "°C/days",

    "temp_min":             "°C",
    "temp_mean":            "°C",
    "temp_max":             "°C",
    "mean_thresh":          "°C",
}

category_colors = ['#ffe7a7', '#fdb941', '#fe821f', '#e84e1c', '#be2e19', '#8b1c15', '#5b0b07', '#2b0403']
category_colormap_lin = mcolors.LinearSegmentedColormap.from_list("linear_category_cmap", category_colors)
category_colormap_seq = lambda n : mcolors.ListedColormap(category_colormap_lin(np.linspace(0, 1, n)))

# Category from Oliver
category_colors_oliver = { 1: '#ffd86e', 2: '#ff621f', 3: '#df391b', 4: '#861a15'}
category_colormap_oliver = mcolors.ListedColormap([category_colors_oliver[i] for i in category_colors_oliver])

# Colormaps to be used for each MHWs statistics
mhws_stats_cmaps = {
    "count":                "cmo.ice_r",                 # "Purples",
    "total_days":           "cmo.tempo",                 # "Blues",
    "moderate_days":        "cmo.tempo",                 # "Blues",
    "strong_days":          "cmo.tempo",                 # "Blues",
    "severe_days":          "cmo.tempo",                 # "Blues",
    "extreme_days":         "cmo.tempo",                 # "Blues",

    "duration":             "cmo.matter",                   # "Blues", "cmo.matter"

    "total_icum":           "cmo.solar_r",               # "Oranges",
    "intensity_cumulative": "cmo.solar_r",               # "Oranges",

    "intensity_max_max":    "cmo.amp",                   # "Reds",
    "intensity_max":        "cmo.amp",                   # "Reds",
    "intensity_mean":       "cmo.amp",                   # "Reds",
    'intensity_mean_byday': "cmo.amp",
    "intensity_var":        "cmo.amp",                   # "Reds",
    
    'total_scum':           category_colormap_lin,
    'severity_cumulative':  category_colormap_lin,

    'severity_max_max':     category_colormap_lin,
    'severity_max':         category_colormap_lin,
    'severity_mean':        category_colormap_lin,
    'severity_mean_byday':  category_colormap_lin,
    'severity_var':         category_colormap_lin,

    "rate_onset":           "cmo.speed",                 # "YlOrRd",
    "rate_decline":         "cmo.speed",                 # "YlOrRd",

    "temp_min":             "cmo.thermal",
    "temp_mean":            "cmo.thermal",
    "temp_max":             "cmo.thermal",
    "mean_thresh":          "cmo.amp",
}

# Description to add to generated MHWs dataset
mhw_dataset_description = "MHWs statistics computed using the marineHeatWaves " \
        "module for python developped by Eric C. J. Oliver."
mhw_yearly_dataset_description = "MHWs yearly statistics computed using the marineHeatWaves " \
        "module for python developped by Eric C. J. Oliver."

# Acknowledgment to add to generated MHWs dataset depending on the original dataset used
rep_acknowledgment = 'Generated using E.U. Copernicus Marine Service Information, ' \
        'Mediterranean Sea - High Resolution L4 Sea Surface Temperature Reprocessed (DOI: https://doi.org/10.48670/moi-00173)'
medrea_acknowledgment = 'Generated using E.U. Copernicus Marine Service Information, ' \
        'Mediterranean Sea Physics Reanalysis (DOI: https://doi.org/10.25423/CMCC/MEDSEA_MULTIYEAR_PHY_006_004_E3R1)'

    # Regions 

# Basic MHWs statistics
regions = [
    'continental_coast',
    'balearic_coast',
    'balearic_sea_deep',
    'west_algerian_deep',
]

# Short names for regions
region_shortname = {
    "continental_coast":    "Continental coast",
    "balearic_coast":       "Balearic Islands coast",
    "balearic_sea_deep":    "Balearic Sea deep",
    "west_algerian_deep":   "West Algerian Basin deep",
}