"""
Defines the analysis and plotting functions for postprocessing of propeller performances.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
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
from plotly.subplots import make_subplots

from fastoad.io import VariableIO

COLS = plotly.colors.DEFAULT_PLOTLY_COLORS


def propeller_efficiency_map_plot(
    aircraft_file_path: str, file_formatter=None, sea_level=False
) -> go.FigureWidget:
    """
    Returns a contour plot of the propeller efficiency maps as they are used in FAST-OAD-GA.

    :param aircraft_file_path: path of data file
    :param file_formatter: the formatter that defines the format of data file. If not provided,
    default format will be assumed.
    :param sea_level: boolean to choose whether to plot the sea level maps or the cruise level map.
    :return: propeller efficiency map.
    """

    variables = VariableIO(aircraft_file_path, file_formatter).read()

    # Maps parameter
    if sea_level:
        efficiency = np.array(variables["data:aerodynamics:propeller:sea_level:efficiency"].value)
        speed = np.array(variables["data:aerodynamics:propeller:sea_level:speed"].value)
        thrust = np.array(variables["data:aerodynamics:propeller:sea_level:thrust"].value)
    else:
        efficiency = np.array(
            variables["data:aerodynamics:propeller:cruise_level:efficiency"].value
        )
        speed = np.array(variables["data:aerodynamics:propeller:cruise_level:speed"].value)
        thrust = np.array(variables["data:aerodynamics:propeller:cruise_level:thrust"].value)

    fig = go.Figure()
    prop_scatter = go.Contour(
        x=thrust,
        y=speed,
        z=efficiency,
        line_smoothing=0.85,
        ncontours=40,
        zmax=np.max(efficiency),
        zmin=0.0,
        colorscale="RdBu",
    )
    fig.add_trace(prop_scatter)

    if sea_level:
        fig.update_layout(
            title_text="Propeller efficiency map at sea level",
            title_x=0.5,
            yaxis_title="True airspeed [m/s]",
            xaxis_title="Thrust [N]",
            font_size=15,
        )
    else:
        fig.update_layout(
            title_text="Propeller efficiency map at cruise level",
            title_x=0.5,
            yaxis_title="True airspeed [m/s]",
            xaxis_title="Thrust [N]",
            font_size=15,
        )

    fig = go.FigureWidget(fig)

    return fig


def propeller_coeff_map_plot(
    aircraft_file_path: str, name="", fig=None, file_formatter=None
) -> go.FigureWidget:
    """
    Returns a two subplot figure of the thrust and power coefficient of the propeller.
    Different figure can be superposed by providing an existing fig.
    Each figure can be provided a name.

    :param aircraft_file_path: path of data file
    :param name: name to give to the trace added to the figure
    :param fig: existing figure to which add the plot
    :param file_formatter: the formatter that defines the format of data file. If not provided,
    default format will be assumed.
    :return: thrust and power coefficient graphs
    """

    variables = VariableIO(aircraft_file_path, file_formatter).read()

    advance_ratio = variables["data:aerodynamics:propeller:coefficient_map:advance_ratio"].value
    c_p = variables["data:aerodynamics:propeller:coefficient_map:power_coefficient"].value
    c_t = variables["data:aerodynamics:propeller:coefficient_map:thrust_coefficient"].value

    if fig is None:
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("Power coefficient", "Thrust coefficient")
        )

    trace_cp = go.Scatter(x=advance_ratio, y=c_p, mode="lines+markers", name=name + " - Cp")
    trace_ct = go.Scatter(x=advance_ratio, y=c_t, mode="lines+markers", name=name + " - Ct")

    fig.add_trace(trace_cp, row=1, col=1)
    fig.update_xaxes(title_text="Advance ratio J [-]", row=1, col=1)
    fig.update_yaxes(title_text="Power coefficient [-]", row=1, col=1)

    fig.add_trace(trace_ct, row=1, col=2)
    fig.update_xaxes(title_text="Advance ratio J [-]", row=1, col=2)
    fig.update_yaxes(title_text="Thrust coefficient [-]", row=1, col=2)

    fig.update_layout(
        title_text="Performance coefficient",
        title_x=0.5,
        font_size=15,
    )

    fig = go.FigureWidget(fig)

    return fig
