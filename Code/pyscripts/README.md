# Code/pyscripts folder

In this folder, Python scripts define functions that are used in the Python notebooks. The code here has been intended to be reusable.

## What is in here?

 - `load_save_dataset.py` :
     Functions for loading and saving datasets from or to the Data/ folder.

 - `mhws_computer.py` :
     Core logic for computing MHW metrics from temperature datasets.

 - `basic_plotter.py` :
     Functions for generating figures from datasets.

 - `options.py` :
     Centralised configuration and options used by other scripts.

 - `utils.py` :
     Helper functions for various tasks.

 - `marineHeatWaves.py` :
     Modified version of *marineHeatWaves* module for Python developed by Eric C. J. Oliver. (see License section)

## License

The *marineHeatWaves* module for python developed by Eric C. J. Oliver has been modified for the purpose of the thesis. The modifications are the following :

 1. **Add severity metrics**: The severity metric has been added as described in the report.
 2. **Calculate means by days and not by event**: This modification makes longer events have more impact on the annual mean (of intensity or severity).
 3. **Option to cut events between 31st December and 1st January**: This modification makes that for a given year, the annual metrics are only based on what happened this given year. This introduces a bias in the mean duration metric, as some events would be split and thus show a lower duration.
