"""
Defines the analysis and plotting functions for postprocessing
"""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import plotly
import plotly.graph_objects as go
from fastga.command import api as api_cs23
import os.path as pth
import pandas as pd
import csv
from plotly.subplots import make_subplots

from fastoad.io import VariableIO
from fastoad.openmdao.variables import VariableList

from .analysis_and_plots import aircraft_polar

COLS = plotly.colors.DEFAULT_PLOTLY_COLORS

current_path: str = r"C:/Users/Lucas/FAST-OAD-GA/FAST-GA/src/fastga/utils/postprocessing"
REF_XML_FILE = pth.join(current_path, 'TBM-900_reference.xml')
DAHER_POLAR_FILE = pth.join(current_path, 'daher_polar.csv')


def simple_comparison(
    aircraft_file_path: str, fig=None, file_formatter=None
) -> go.FigureWidget:
    """
    Returns a figure plot of the top view of the wing.
    Different designs can be superposed by providing an existing fig.
    Each design can be provided a name.

    :param aircraft_file_path: path of data file
    :param name: name to give to the trace added to the figure
    :param fig: existing figure to which add the plot
    :param file_formatter: the formatter that defines the format of data file. If not provided, default format will
                           be assumed.
    :return: wing plot figure
    """
    var_ref = VariableIO(REF_XML_FILE, file_formatter).read()
    var_computed = VariableIO(aircraft_file_path, file_formatter).read()
    var_computed_names = np.array([var.name for var in var_computed])

    # var_list = np.array([var for var in var_ref if var.name.split(":")[1] == "geometry"])

    computed_values = []
    real_values = []
    names = []
    error = []

    for var in var_ref:
        real_values.append(var.value[0])
        index_var = np.where(var_computed_names == var.name)
        computed_values.append(round(var_computed[index_var[0][0]].value[0],3))
        names.append(','.join(var.name.split(":")[2:]) + "  [" + var.units + "]")
        error.append(round(100 * abs((computed_values[-1] - real_values[-1]) / real_values[-1]), 2))

    fig = go.Figure(data=[go.Table(header=dict(values=['Quantity', 'Real values', 'Computed values', 'Relative error %']),
                                   cells=dict(values=[names, real_values, computed_values, error]))
                          ])

    return fig


def tbm_drag_polar_comparison(
    aircraft_file_path: str, fig=None, file_formatter=None, name=None
) -> go.FigureWidget:
    """
    Returns a figure plot of the top view of the wing.
    Different designs can be superposed by providing an existing fig.
    Each design can be provided a name.

    :param aircraft_file_path: path of data file
    :param name: name to give to the trace added to the figure
    :param fig: existing figure to which add the plot
    :param file_formatter: the formatter that defines the format of data file. If not provided, default format will
                           be assumed.
    :return: wing plot figure
    """

    var_computed = VariableIO(aircraft_file_path, file_formatter).read()

    mach_cruise = var_computed["data:aerodynamics:cruise:mach"].value[0]
    mach_low_speed = var_computed["data:aerodynamics:low_speed:mach"].value[0]

    # Retrieval of Daher data and interpolation to match the cruise mach number
    mach_array = [0, 0.4, 0.5, 0.6]

    cl_array_cruise = [[0.862, 0.877, 1.02, 1.076, 1.258], [1.019, 1.036, 1.192, 1.253, 1.452], [1.139, 1.157, 1.322, 1.386, 1.597], [1.334, 1.353, 1.532, 1.602, 1.83]]
    cd_array_cruise = [[0.0336, 0.0372, 0.0592, 0.0736, 0.1236], [0.0336, 0.0372, 0.0592, 0.0736, 0.1236], [0.0336, 0.0372, 0.0592, 0.0736, 0.1236], [0.0336, 0.0372, 0.0592, 0.0736, 0.1236]]
    cl_array_low_speed = [[0.234, 0.273, 0.624, 0.761, 1.21], [0.338, 0.381, 0.764, 0.914, 1.404], [0.421, 0.466, 0.872, 1.03, 1.549], [0.561, 0.609, 1.049, 1.22, 1.781]]
    cd_array_low_speed = [[0.0336, 0.0372, 0.0592, 0.0736, 0.1236], [0.0336, 0.0372, 0.0592, 0.0736, 0.1236], [0.0336, 0.0372, 0.0592, 0.0736, 0.1236], [0.0336, 0.0372, 0.0592, 0.0736, 0.1236]]

    cl_array_mach_cruise = []
    cl_array_mach_low_speed = []

    for i in range(len(cd_array_cruise[0])):
        cl_interp = [cl[i] for cl in cl_array_cruise]
        cl_array_mach_cruise.append(np.interp(mach_cruise,mach_array,cl_interp))

    for i in range(len(cd_array_low_speed[0])):
        cl_interp = [cl[i] for cl in cl_array_low_speed]
        cl_array_mach_low_speed.append(np.interp(mach_low_speed,mach_array,cl_interp))

    if fig is None:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Cruise", "Low Speed"))

    scatter = go.Scatter(x=cd_array_cruise[0], y=cl_array_mach_cruise, mode="lines+markers",name="Daher data")
    fig.add_trace(scatter, 1, 1)

    scatter = go.Scatter(x=cd_array_low_speed[0], y=cl_array_mach_low_speed, mode="lines+markers",name="Daher data")
    fig.add_trace(scatter, 1, 2)

    fig = aircraft_polar(aircraft_file_path,file_formatter,fig=fig)

    fig = go.FigureWidget(fig)

    return fig