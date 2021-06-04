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

from fastoad.io import VariableIO

COLS = plotly.colors.DEFAULT_PLOTLY_COLORS


def evolution_diagram(
    aircraft_file_path: str, name=None, fig=None, file_formatter=None
) -> go.FigureWidget:
    """
    Returns a figure plot of the V-N diagram of the aircraft.
    Different designs can be superposed by providing an existing fig.
    Each design can be provided a name.

    :param aircraft_file_path: path of data file
    :param name: name to give to the trace added to the figure
    :param fig: existing figure to which add the plot
    :param file_formatter: the formatter that defines the format of data file. If not provided, default format will
                           be assumed.
    :return: V-N plot figure
    """
    variables = VariableIO(aircraft_file_path, file_formatter).read()

    velocity_array = list(variables["data:flight_domain:velocity"].value)
    load_factor_array = list(variables["data:flight_domain:load_factor"].value)

    # Save maneuver envelope
    x_maneuver_line = list(np.linspace(velocity_array[0], velocity_array[2], 10))
    y_maneuver_line = []
    x_maneuver_pts = [velocity_array[0], velocity_array[2]]
    y_maneuver_pts = [load_factor_array[0], load_factor_array[2]]
    for idx in range(len(x_maneuver_line)):
        y_maneuver_line.append(load_factor_array[0] * (x_maneuver_line[idx] / velocity_array[0])**2.0)
    x_maneuver_line.extend([velocity_array[9], velocity_array[10], velocity_array[6], velocity_array[3]])
    y_maneuver_line.extend([load_factor_array[9], load_factor_array[10], load_factor_array[6], load_factor_array[3]])
    x_local = list(np.linspace(velocity_array[3], velocity_array[1], 10))
    x_maneuver_line.extend(x_local)
    x_maneuver_pts.extend([velocity_array[9], velocity_array[10], velocity_array[6], velocity_array[3],
                           velocity_array[1]])
    y_maneuver_pts.extend([load_factor_array[9], load_factor_array[10], load_factor_array[6], load_factor_array[3],
                           load_factor_array[1]])
    for idx in range(len(x_local)):
        y_maneuver_line.append(load_factor_array[1] * (x_local[idx] / x_local[-1])**2.0)
    x_maneuver_line.extend([x_local[-1], velocity_array[0], velocity_array[0]])
    y_maneuver_line.extend([0.0, 0.0, load_factor_array[0]])

    # Save gust envelope
    x_gust = [0.0]
    y_gust = [0.0]
    if not(velocity_array[4] == 0.0):
        x_gust.append(velocity_array[4])
        y_gust.append(load_factor_array[4])
    x_gust.append(velocity_array[7])
    y_gust.append(load_factor_array[7])
    x_gust.append(velocity_array[11])
    y_gust.append(load_factor_array[11])
    x_gust.append(velocity_array[12])
    y_gust.append(load_factor_array[12])
    x_gust.append(velocity_array[8])
    y_gust.append(load_factor_array[8])
    if not (velocity_array[5] == 0.0):
        x_gust.append(velocity_array[5])
        y_gust.append(load_factor_array[5])
    x_gust.append(0.0)
    y_gust.append(0.0)


    if fig is None:
        fig = go.Figure()

    scatter = go.Scatter(x=x_maneuver_line, y=y_maneuver_line, mode="lines", name=name + ' - maneuver')

    fig.add_trace(scatter)

    scatter = go.Scatter(x=x_maneuver_pts, y=y_maneuver_pts, mode="markers", name=name + ' - maneuver [points]')

    fig.add_trace(scatter)

    scatter = go.Scatter(x=x_gust, y=y_gust, mode="lines+markers", name=name + ' - gust')

    fig.add_trace(scatter)

    fig = go.FigureWidget(fig)

    fig.update_layout(
        title_text="Evolution Diagram",
        title_x=0.5,
        xaxis=dict(range=[0.0, max(max(x_maneuver_line), max(x_gust)) * 1.1]),
        xaxis_title="speed [m/s]",
        yaxis=dict(range=[min(min(y_maneuver_line), min(y_gust)) * 1.1, max(max(y_maneuver_line), max(y_gust)) * 1.1]),
        yaxis_title="load [g]",
    )


    return fig
