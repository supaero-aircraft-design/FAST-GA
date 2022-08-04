"""
Defines the analysis and plotting functions for postprocessing of load analysis.
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

    variables = VariableIO(aircraft_file_path, file_formatter).read()

    # Maps parameter
    if sea_level:
        efficiency = variables["data:aerodynamics:propeller:sea_level:efficiency"].value
        speed = variables["data:aerodynamics:propeller:sea_level:speed"].value
        thrust = variables["data:aerodynamics:propeller:sea_level:thrust"].value
    else:
        efficiency = variables["data:aerodynamics:propeller:cruise_level:efficiency"].value
        speed = variables["data:aerodynamics:propeller:cruise_level:speed"].value
        thrust = variables["data:aerodynamics:propeller:cruise_level:thrust"].value

    fig = go.Figure()
    prop_scatter = go.Contour(
        x=speed,
        y=thrust,
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
            xaxis_title="True airspeed [m/s]",
            yaxis_title="Thrust [N]",
            font_size=15,
        )
    else:
        fig.update_layout(
            title_text="Propeller efficiency map at cruise level",
            title_x=0.5,
            xaxis_title="True airspeed [m/s]",
            yaxis_title="Thrust [N]",
            font_size=15,
        )

    fig = go.FigureWidget(fig)

    return fig


def propeller_coeff_map_plot(
    aircraft_file_path: str, name="", fig=None, file_formatter=None
) -> go.FigureWidget:

    variables = VariableIO(aircraft_file_path, file_formatter).read()

    advance_ratio = variables["data:aerodynamics:propeller:coefficient_map:advance_ratio"].value
    cp = variables["data:aerodynamics:propeller:coefficient_map:power_coefficient"].value
    ct = variables["data:aerodynamics:propeller:coefficient_map:thrust_coefficient"].value

    if fig is None:
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("Power coefficient", "Thrust coefficient")
        )

    trace_cp = go.Scatter(x=advance_ratio, y=cp, mode="lines+markers", name=name + " - Cp")
    trace_ct = go.Scatter(x=advance_ratio, y=ct, mode="lines+markers", name=name + " - Ct")

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
