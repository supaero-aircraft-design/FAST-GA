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
from fastoad.io import VariableIO

from fastga.models.load_analysis.wing.constants import (
    POINT_MASS_SPAN_RATIO,
    NB_POINTS_POINT_MASS,
)

COLS = plotly.colors.DEFAULT_PLOTLY_COLORS


def force_repartition_diagram(
    aircraft_file_path: str, name="", fig=None, file_formatter=None
) -> go.FigureWidget:
    """
    Returns a figure plot of the force repartition on the wing.
    Different designs can be superposed by providing an existing fig.
    Each design can be provided a name.

    :param aircraft_file_path: path of data file
    :param name: name to give to the trace added to the figure
    :param fig: existing figure to which add the plot
    :param file_formatter: the formatter that defines the format of data file. If not provided,
    default format will be assumed.
    :return: force repartition diagram.
    """
    variables = VariableIO(aircraft_file_path, file_formatter).read()

    y_vector = list(variables["data:loads:y_vector"].value)
    wing_weight = list(variables["data:loads:structure:ultimate:force_distribution:wing"].value)
    fuel_weight = list(variables["data:loads:structure:ultimate:force_distribution:fuel"].value)
    point_weight = list(
        variables["data:loads:structure:ultimate:force_distribution:point_mass"].value
    )
    lift = list(variables["data:loads:aerodynamic:ultimate:force_distribution"].value)
    span = variables["data:geometry:wing:span"].value
    semi_span = span[0] / 2.0

    interval_len = POINT_MASS_SPAN_RATIO * semi_span / NB_POINTS_POINT_MASS
    readjust_point = 1.5 * interval_len

    y_vector_tmp = np.array(y_vector)
    wing_weight_tmp = np.array(wing_weight)
    fuel_weight_tmp = np.array(fuel_weight)
    point_weight_tmp = np.array(point_weight)
    lift_tmp = np.array(lift)

    y_vector = _delete_additional_zeros(y_vector_tmp)
    index = len(y_vector) + 1
    wing_weight = wing_weight_tmp[: int(index)]
    fuel_weight = fuel_weight_tmp[: int(index)]
    # We have to readjust the point mass since because of the way we represented it (finite over
    # a small interval) the value of the array is artificially high
    point_weight = readjust_point * point_weight_tmp[: int(index)]
    lift = lift_tmp[0 : int(index)]

    if fig is None:
        fig = go.Figure()

    wing_weight_scatter = go.Scatter(
        x=y_vector, y=wing_weight, mode="lines", name=name + " - wing weight"
    )

    fig.add_trace(wing_weight_scatter)

    fuel_weight_scatter = go.Scatter(
        x=y_vector, y=fuel_weight, mode="lines", name=name + " - fuel weight"
    )

    fig.add_trace(fuel_weight_scatter)

    point_weight_scatter = go.Scatter(
        x=y_vector, y=point_weight, mode="lines", name=name + " - point masses weight"
    )

    fig.add_trace(point_weight_scatter)

    lift_scatter = go.Scatter(x=y_vector, y=lift, mode="lines", name=name + " - lift")

    fig.add_trace(lift_scatter)

    fig = go.FigureWidget(fig)

    fig.update_layout(
        title_text="Forces distribution on the wing",
        title_x=0.5,
        xaxis_title="spanwise position [m]",
        yaxis_title="distributed force [N/m]",
    )

    return fig


def shear_diagram(
    aircraft_file_path: str, name="", fig=None, file_formatter=None
) -> go.FigureWidget:
    """
    Returns a figure plot of the shear repartition on the wing.
    Different designs can be superposed by providing an existing fig.
    Each design can be provided a name.

    :param aircraft_file_path: path of data file
    :param name: name to give to the trace added to the figure
    :param fig: existing figure to which add the plot
    :param file_formatter: the formatter that defines the format of data file. If not provided,
    default format will be assumed.
    :return: force repartition diagram.
    """
    variables = VariableIO(aircraft_file_path, file_formatter).read()

    y_vector = list(variables["data:loads:y_vector"].value)
    wing_shear = list(variables["data:loads:structure:ultimate:shear:wing"].value)
    fuel_shear = list(variables["data:loads:structure:ultimate:shear:fuel"].value)
    point_shear = list(variables["data:loads:structure:ultimate:shear:point_mass"].value)
    lift_shear = list(variables["data:loads:max_shear:lift_shear"].value)

    y_vector_tmp = np.array(y_vector)
    wing_shear_tmp = np.array(wing_shear)
    fuel_shear_tmp = np.array(fuel_shear)
    point_shear_tmp = np.array(point_shear)
    lift_shear_tmp = np.array(lift_shear)

    y_vector = _delete_additional_zeros(y_vector_tmp)
    index = len(y_vector) + 1
    wing_shear = wing_shear_tmp[: int(index)]
    fuel_shear = fuel_shear_tmp[: int(index)]
    point_shear = point_shear_tmp[: int(index)]
    lift_shear = lift_shear_tmp[: int(index)]

    if fig is None:
        fig = go.Figure()

    wing_shear_scatter = go.Scatter(
        x=y_vector, y=wing_shear, mode="lines", name=name + " - wing weight shear"
    )

    fig.add_trace(wing_shear_scatter)

    fuel_shear_scatter = go.Scatter(
        x=y_vector, y=fuel_shear, mode="lines", name=name + " - fuel weight shear"
    )

    fig.add_trace(fuel_shear_scatter)

    point_shear_scatter = go.Scatter(
        x=y_vector, y=point_shear, mode="lines", name=name + " - point masses shear"
    )

    fig.add_trace(point_shear_scatter)

    lift_scatter = go.Scatter(x=y_vector, y=lift_shear, mode="lines", name=name + " - lift shear")

    fig.add_trace(lift_scatter)

    fig = go.FigureWidget(fig)

    fig.update_layout(
        title_text="Shear forces on the wing",
        title_x=0.5,
        xaxis_title="spanwise position [m]",
        yaxis_title="shear force [N]",
    )

    return fig


def rbm_diagram(aircraft_file_path: str, name="", fig=None, file_formatter=None) -> go.FigureWidget:
    """
    Returns a figure plot of the root bending moment repartition on the wing.
    Different designs can be superposed by providing an existing fig.
    Each design can be provided a name.

    :param aircraft_file_path: path of data file
    :param name: name to give to the trace added to the figure
    :param fig: existing figure to which add the plot
    :param file_formatter: the formatter that defines the format of data file. If not provided,
    default format will be assumed.
    :return: force repartition diagram.
    """
    variables = VariableIO(aircraft_file_path, file_formatter).read()

    y_vector = list(variables["data:loads:y_vector"].value)
    wing_rbm = list(variables["data:loads:structure:ultimate:root_bending:wing"].value)
    fuel_rbm = list(variables["data:loads:structure:ultimate:root_bending:fuel"].value)
    point_rbm = list(variables["data:loads:structure:ultimate:root_bending:point_mass"].value)
    lift_rbm = list(variables["data:loads:max_rbm:lift_rbm"].value)

    y_vector_tmp = np.array(y_vector)
    wing_rbm_tmp = np.array(wing_rbm)
    fuel_rbm_tmp = np.array(fuel_rbm)
    point_rbm_tmp = np.array(point_rbm)
    lift_rbm_tmp = np.array(lift_rbm)

    y_vector = _delete_additional_zeros(y_vector_tmp)
    index = len(y_vector) + 1
    wing_rbm = wing_rbm_tmp[: int(index)]
    fuel_rbm = fuel_rbm_tmp[: int(index)]
    point_rbm = point_rbm_tmp[: int(index)]
    lift_rbm = lift_rbm_tmp[: int(index)]

    if fig is None:
        fig = go.Figure()

    wing_rbm_scatter = go.Scatter(
        x=y_vector, y=wing_rbm, mode="lines", name=name + " - wing weight bending moment"
    )

    fig.add_trace(wing_rbm_scatter)

    fuel_rbm_scatter = go.Scatter(
        x=y_vector, y=fuel_rbm, mode="lines", name=name + " - fuel weight bending moment"
    )

    fig.add_trace(fuel_rbm_scatter)

    point_rbm_scatter = go.Scatter(
        x=y_vector, y=point_rbm, mode="lines", name=name + " - point masses bending moment"
    )

    fig.add_trace(point_rbm_scatter)

    lift_scatter = go.Scatter(
        x=y_vector, y=lift_rbm, mode="lines", name=name + " - lift bending moment"
    )

    fig.add_trace(lift_scatter)

    fig = go.FigureWidget(fig)

    fig.update_layout(
        title_text="Bending moments on the wing",
        title_x=0.5,
        xaxis_title="spanwise position [m]",
        yaxis_title="Root bending moment [Nm]",
    )

    return fig


def _delete_additional_zeros(array, length: int = None):
    """
    Function that delete the additional zeros we had to add to fit the format imposed by
    OpenMDAO

    @param array: an array with additional zeros we want to delete
    @param length: if len is specified leave zeros up until the length of the array is len
    @return: final_array an array containing the same elements of the initial array but with
    the additional zeros deleted
    """

    last_zero = np.amax(np.where(array != 0.0)) + 1
    if length is not None:
        final_array = array[: max(int(last_zero), length)]
    else:
        final_array = array[: int(last_zero)]

    return final_array
