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
import plotly.graph_objects as go
import matplotlib.colors as colour

import plotly

from plotly.subplots import make_subplots

from typing import Dict
from random import SystemRandom
from openmdao.utils.units import convert_units

from fastoad.io import VariableIO
from fastoad.openmdao.variables import VariableList

COLS = plotly.colors.DEFAULT_PLOTLY_COLORS


def aircraft_geometry_plot(
    aircraft_file_path: str, name=None, fig=None, plot_nacelle: bool = True, file_formatter=None
) -> go.FigureWidget:
    """
    Returns a figure plot of the top view of the wing.
    Different designs can be superposed by providing an existing fig.
    Each design can be provided a name.

    :param aircraft_file_path: path of data file
    :param name: name to give to the trace added to the figure
    :param fig: existing figure to which add the plot
    :param plot_nacelle: boolean to turn on or off the plotting of the nacelles
    :param file_formatter: the formatter that defines the format of data file. If not provided, default format will
                           be assumed.
    :return: wing plot figure
    """
    variables = VariableIO(aircraft_file_path, file_formatter).read()

    # Wing parameters
    wing_kink_leading_edge_x = variables["data:geometry:wing:kink:leading_edge:x:local"].value[0]
    wing_tip_leading_edge_x = variables["data:geometry:wing:tip:leading_edge:x:local"].value[0]
    wing_root_y = variables["data:geometry:wing:root:y"].value[0]
    wing_kink_y = variables["data:geometry:wing:kink:y"].value[0]
    wing_tip_y = variables["data:geometry:wing:tip:y"].value[0]
    wing_root_chord = variables["data:geometry:wing:root:chord"].value[0]
    wing_kink_chord = variables["data:geometry:wing:kink:chord"].value[0]
    wing_tip_chord = variables["data:geometry:wing:tip:chord"].value[0]

    y_wing = np.array(
        [0, wing_root_y, wing_kink_y, wing_tip_y, wing_tip_y, wing_kink_y, wing_root_y, 0, 0]
    )

    x_wing = np.array(
        [
            0,
            0,
            wing_kink_leading_edge_x,
            wing_tip_leading_edge_x,
            wing_tip_leading_edge_x + wing_tip_chord,
            wing_kink_leading_edge_x + wing_kink_chord,
            wing_root_chord,
            wing_root_chord,
            0,
        ]
    )

    # Horizontal Tail parameters
    ht_root_chord = variables["data:geometry:horizontal_tail:root:chord"].value[0]
    ht_tip_chord = variables["data:geometry:horizontal_tail:tip:chord"].value[0]
    ht_span = variables["data:geometry:horizontal_tail:span"].value[0]
    ht_sweep_0 = variables["data:geometry:horizontal_tail:sweep_0"].value[0]

    ht_tip_leading_edge_x = ht_span / 2.0 * np.tan(ht_sweep_0 * np.pi / 180.0)

    y_ht = np.array([0, ht_span / 2.0, ht_span / 2.0, 0.0, 0.0])

    x_ht = np.array(
        [0, ht_tip_leading_edge_x, ht_tip_leading_edge_x + ht_tip_chord, ht_root_chord, 0]
    )

    # Fuselage parameters
    fuselage_max_width = variables["data:geometry:fuselage:maximum_width"].value[0]
    fuselage_length = variables["data:geometry:fuselage:length"].value[0]
    fuselage_front_length = variables["data:geometry:fuselage:front_length"].value[0]
    fuselage_rear_length = variables["data:geometry:fuselage:rear_length"].value[0]

    x_fuselage = np.array(
        [
            0.0,
            0.0,
            fuselage_front_length,
            fuselage_length - fuselage_rear_length,
            fuselage_length,
            fuselage_length,
        ]
    )

    y_fuselage = np.array(
        [
            0.0,
            fuselage_max_width / 4.0,
            fuselage_max_width / 2.0,
            fuselage_max_width / 2.0,
            fuselage_max_width / 4.0,
            0.0,
        ]
    )

    # CGs
    wing_25mac_x = variables["data:geometry:wing:MAC:at25percent:x"].value[0]
    wing_mac_length = variables["data:geometry:wing:MAC:length"].value[0]
    local_wing_mac_le_x = variables["data:geometry:wing:MAC:leading_edge:x:local"].value[0]
    local_ht_25mac_x = variables["data:geometry:horizontal_tail:MAC:at25percent:x:local"].value[0]
    ht_distance_from_wing = variables[
        "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"
    ].value[0]

    x_wing = x_wing + wing_25mac_x - 0.25 * wing_mac_length - local_wing_mac_le_x
    x_ht = x_ht + wing_25mac_x + ht_distance_from_wing - local_ht_25mac_x

    # pylint: disable=invalid-name # that's a common naming
    x = np.concatenate((x_fuselage, x_wing, x_ht))
    # pylint: disable=invalid-name # that's a common naming
    y = np.concatenate((y_fuselage, y_wing, y_ht))

    # pylint: disable=invalid-name # that's a common naming
    y = np.concatenate((-y, y))
    # pylint: disable=invalid-name # that's a common naming
    x = np.concatenate((x, x))

    if fig is None:
        fig = go.Figure()

    scatter = go.Scatter(x=y, y=x, mode="lines+markers", name=name, showlegend=True)

    fig.add_trace(scatter)

    # Nacelle + propeller
    prop_layout = variables["data:geometry:propulsion:layout"].value[0]
    nac_width = variables["data:geometry:propulsion:nacelle:width"].value[0]
    nac_length = variables["data:geometry:propulsion:nacelle:length"].value[0]
    prop_diam = variables["data:geometry:propeller:diameter"].value[0]
    pos_y_nacelle = np.array(variables["data:geometry:propulsion:nacelle:y"].value)
    pos_x_nacelle = np.array(variables["data:geometry:propulsion:nacelle:x"].value)

    if prop_layout == 1.0:
        x_nacelle_plot = np.array([0.0, nac_length, nac_length, 0.0, 0.0, 0.0])
        y_nacelle_plot = np.array(
            [
                -nac_width / 2,
                -nac_width / 2,
                nac_width / 2,
                nac_width / 2,
                prop_diam / 2,
                -prop_diam / 2,
            ]
        )
    elif prop_layout == 3.0:
        x_nacelle_plot = np.array([0.0, nac_length, nac_length, 0.0, 0.0, 0.0])
        y_nacelle_plot = np.array(
            [
                max(-nac_width / 2, -fuselage_max_width / 4.0),
                -nac_width / 2,
                nac_width / 2,
                min(nac_width / 2, fuselage_max_width / 4.0),
                prop_diam / 2,
                -prop_diam / 2,
            ]
        )
    else:
        x_nacelle_plot = np.array([])
        y_nacelle_plot = np.array([])

    if plot_nacelle:
        if prop_layout == 1.0:

            all_colour = colour.CSS4_COLORS.keys()
            random_generator = SystemRandom()
            trace_colour = list(all_colour)[random_generator.randrange(0, len(list(all_colour)))]
            used_index = np.where(pos_y_nacelle >= 0.0)[0]
            show_legend = True

            for index in used_index:

                y_nacelle_local = pos_y_nacelle[index]
                x_nacelle_local = pos_x_nacelle[index]

                y_nacelle_left = y_nacelle_plot + y_nacelle_local
                y_nacelle_right = -y_nacelle_plot - y_nacelle_local
                x_nacelle = x_nacelle_local - x_nacelle_plot

                if show_legend:
                    scatter = go.Scatter(
                        x=y_nacelle_right,
                        y=x_nacelle,
                        mode="lines+markers",
                        line=dict(color=trace_colour),
                        name=name + " nacelle + propeller",
                    )
                    show_legend = False

                else:
                    scatter = go.Scatter(
                        x=y_nacelle_right,
                        y=x_nacelle,
                        mode="lines+markers",
                        line=dict(color=trace_colour),
                        showlegend=False,
                    )

                fig.add_trace(scatter)

                scatter = go.Scatter(
                    x=y_nacelle_left,
                    y=x_nacelle,
                    mode="lines+markers",
                    line=dict(color=trace_colour),
                    showlegend=False,
                )

                fig.add_trace(scatter)
        else:
            scatter = go.Scatter(
                x=y_nacelle_plot,
                y=x_nacelle_plot,
                mode="lines+markers",
                name=name + " nacelle + propeller",
            )
            fig.add_trace(scatter)

    fig.layout = go.Layout(yaxis=dict(scaleanchor="x", scaleratio=1))

    fig = go.FigureWidget(fig)

    fig.update_layout(
        title_text="Aircraft Geometry",
        title_x=0.5,
        xaxis_title="y",
        yaxis_title="x",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


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
        y_maneuver_line.append(
            load_factor_array[0] * (x_maneuver_line[idx] / velocity_array[0]) ** 2.0
        )
    x_maneuver_line.extend(
        [velocity_array[9], velocity_array[10], velocity_array[6], velocity_array[3]]
    )
    y_maneuver_line.extend(
        [load_factor_array[9], load_factor_array[10], load_factor_array[6], load_factor_array[3]]
    )
    x_local = list(np.linspace(velocity_array[3], velocity_array[1], 10))
    x_maneuver_line.extend(x_local)
    x_maneuver_pts.extend(
        [
            velocity_array[9],
            velocity_array[10],
            velocity_array[6],
            velocity_array[3],
            velocity_array[1],
        ]
    )
    y_maneuver_pts.extend(
        [
            load_factor_array[9],
            load_factor_array[10],
            load_factor_array[6],
            load_factor_array[3],
            load_factor_array[1],
        ]
    )
    for idx in range(len(x_local)):
        y_maneuver_line.append(load_factor_array[1] * (x_local[idx] / x_local[-1]) ** 2.0)
    x_maneuver_line.extend([x_local[-1], velocity_array[0], velocity_array[0]])
    y_maneuver_line.extend([0.0, 0.0, load_factor_array[0]])

    # Save gust envelope
    x_gust = [0.0]
    y_gust = [1.0]
    if not (velocity_array[4] == 0.0):
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
    y_gust.append(1.0)

    if fig is None:
        fig = go.Figure()

    scatter = go.Scatter(
        x=x_maneuver_line, y=y_maneuver_line, mode="lines", name=name + " - maneuver"
    )

    fig.add_trace(scatter)

    scatter = go.Scatter(
        x=x_maneuver_pts, y=y_maneuver_pts, mode="markers", name=name + " - maneuver [points]"
    )

    fig.add_trace(scatter)

    scatter = go.Scatter(x=x_gust, y=y_gust, mode="lines+markers", name=name + " - gust")

    fig.add_trace(scatter)

    fig = go.FigureWidget(fig)

    fig.update_layout(
        title_text="Evolution Diagram",
        title_x=0.5,
        xaxis=dict(range=[0.0, max(max(x_maneuver_line), max(x_gust)) * 1.1]),
        xaxis_title="speed [m/s]",
        yaxis=dict(
            range=[
                min(min(y_maneuver_line), min(y_gust)) * 1.1,
                max(max(y_maneuver_line), max(y_gust)) * 1.1,
            ]
        ),
        yaxis_title="load [g]",
    )

    return fig


def cl_wing_diagram(
    aircraft_ref_file_path: str,
    aircraft_mod_file_path: str,
    prop_on: bool = True,
    name_ref=None,
    name_mod=None,
    file_formatter=None,
) -> [go.FigureWidget, go.FigureWidget]:
    """
    Returns a figure plot of the CL distribution on the semi-wing, and highlights the delta_CL before the added part of
    the wing or before the reduced part of the wing.

    :param aircraft_ref_file_path: path of reference aircraft data file
    :param aircraft_mod_file_path: path of modified aircraft data file
    :param prop_on: boolean stating if the rotor is on or off (for single propeller plane)
    :param name_ref: name to give to the trace of the reference aircraft
    :param name_mod: name to give to the trace of the modified aircraft
    :param file_formatter: the formatter that defines the format of data file. If not provided, default format will
                           be assumed.
    :return: Cl distribution figure along the span
    """

    variables_ref = VariableIO(aircraft_ref_file_path, file_formatter).read()
    variables_mod = VariableIO(aircraft_mod_file_path, file_formatter).read()

    if prop_on:
        cl_array_ref = list(
            variables_ref["data:aerodynamics:slipstream:wing:cruise:prop_on:CL_vector"].value
        )
        span_array_ref = list(
            variables_ref["data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector"].value
        )
        cl_array_mod = list(
            variables_mod["data:aerodynamics:slipstream:wing:cruise:prop_on:CL_vector"].value
        )
        span_array_mod = list(
            variables_mod["data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector"].value
        )
    else:
        cl_array_ref = list(
            variables_ref["data:aerodynamics:slipstream:wing:cruise:prop_off:CL_vector"].value
        )
        span_array_ref = list(
            variables_ref["data:aerodynamics:slipstream:wing:cruise:prop_off:Y_vector"].value
        )
        cl_array_mod = list(
            variables_mod["data:aerodynamics:slipstream:wing:cruise:prop_off:CL_vector"].value
        )
        span_array_mod = list(
            variables_mod["data:aerodynamics:slipstream:wing:cruise:prop_off:Y_vector"].value
        )

    cl_array_ref = [i for i in cl_array_ref if i != 0]
    cl_array_ref.append(0)
    span_array_ref = [i for i in span_array_ref if i != 0]
    semi_span_ref = variables_ref["data:geometry:wing:span"].value[0] / 2

    span_array_ref.append(semi_span_ref)

    cl_array_mod = [i for i in cl_array_mod if i != 0]
    cl_array_mod.append(0)
    span_array_mod = [i for i in span_array_mod if i != 0]
    semi_span_mod = variables_mod["data:geometry:wing:span"].value[0] / 2

    span_array_mod.append(semi_span_mod)

    if span_array_mod[-1] >= span_array_ref[-1]:
        longer_wing = True
        span_array_short = span_array_ref
        span_array_long = span_array_mod
        cl_array_short = cl_array_ref
        cl_array_long = cl_array_mod
        name_short = name_ref
        name_long = name_mod
    else:
        longer_wing = False
        span_array_short = span_array_mod
        span_array_long = span_array_ref
        cl_array_short = cl_array_mod
        cl_array_long = cl_array_ref
        name_short = name_mod
        name_long = name_ref

    y = np.interp(span_array_short, span_array_long, cl_array_long)

    fig = go.Figure()
    scatter = go.Scatter(x=span_array_short, y=cl_array_short, name=name_short)
    fig.add_trace(scatter)
    scatter = go.Scatter(x=span_array_long, y=cl_array_long, name=name_long)
    fig.add_trace(scatter)
    scatter = go.Scatter(x=span_array_short, y=y, mode="markers", name="interpol")
    fig.add_trace(scatter)
    fig = go.FigureWidget(fig)

    if prop_on:
        name_diagram = "propeller ON"
    else:
        name_diagram = "propeller OFF"

    fig.update_layout(
        title_text="CL wing distribution with " + name_diagram,
        title_x=0.5,
        xaxis=dict(range=[0.0, max(span_array_long) * 1.1]),
        xaxis_title="Semi-Span [m]",
        yaxis=dict(range=[0, max(cl_array_long) * 1.1]),
        yaxis_title="CL [-]",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    fig1 = go.Figure()
    if longer_wing:
        y_scatter = y - cl_array_ref
    else:
        y_scatter = list(np.array(cl_array_mod) - y)

    scatter = go.Scatter(x=span_array_short, y=y_scatter, name="Delta_CL")
    fig1.add_trace(scatter)
    fig1 = go.FigureWidget(fig1)
    fig1.update_layout(
        title_text="Delta_CL wing Modified Configuration minus Reference Configuration with "
        + name_diagram,
        title_x=0.5,
        xaxis=dict(range=[0.0, max(span_array_short) * 1.1]),
        xaxis_title="Semi-Span of shortest configuration : " + name_short + " [m]",
        yaxis=dict(range=[min(y_scatter) * 1.1, max(y_scatter) * 1.2]),
        yaxis_title="Delta_CL [-]",
    )

    return fig, fig1


def cg_lateral_diagram(
    aircraft_file_path: str, name=None, fig=None, file_formatter=None, color=None
) -> go.FigureWidget:
    """
    Returns a figure plot of the lateral view of the plane.
    Different designs can be superposed by providing an existing fig.
    Each design can be provided a name.

    :param aircraft_file_path: path of data file
    :param name: name to give to the trace added to the figure
    :param color: color that we give to the aft, empty and fwd CGs of the aircraft
    :param fig: existing figure to which add the plot
    :param file_formatter: the formatter that defines the format of data file. If not provided, default format will
                           be assumed.
    :return: wing plot figure
    """
    variables = VariableIO(aircraft_file_path, file_formatter).read()

    # Fuselage parameters
    fuselage_max_height = variables["data:geometry:fuselage:maximum_height"].value[0]
    fuselage_length = variables["data:geometry:fuselage:length"].value[0]
    fuselage_front_length = variables["data:geometry:fuselage:front_length"].value[0]
    fuselage_rear_length = variables["data:geometry:fuselage:rear_length"].value[0]

    x_fuselage = np.array(
        [
            0.0,
            0.0,
            fuselage_front_length,
            fuselage_length - fuselage_rear_length,
            fuselage_length,
            fuselage_length,
        ]
    )

    z_fuselage = np.array(
        [
            0.0,
            fuselage_max_height / 5.0,
            fuselage_max_height / 2.0,
            fuselage_max_height / 2.5,
            fuselage_max_height / 5.0,
            0.0,
        ]
    )

    z_fuselage = np.concatenate((-z_fuselage, z_fuselage))
    x_fuselage = np.concatenate((x_fuselage, x_fuselage))

    # Vertical Tail parameters
    vt_root_chord = variables["data:geometry:vertical_tail:root:chord"].value[0]
    vt_tip_chord = variables["data:geometry:vertical_tail:tip:chord"].value[0]
    vt_span = variables["data:geometry:vertical_tail:span"].value[0]
    vt_sweep_0 = variables["data:geometry:vertical_tail:sweep_0"].value[0]

    vt_tip_leading_edge_x = vt_span * np.tan(vt_sweep_0 * np.pi / 180.0)

    x_vt = np.array(
        [0, vt_tip_leading_edge_x, vt_tip_leading_edge_x + vt_tip_chord, vt_root_chord, 0]
    )

    z_vt = np.array([0, vt_span, vt_span, 0, 0])

    wing_25mac_x = variables["data:geometry:wing:MAC:at25percent:x"].value[0]
    local_vt_25mac_x = variables["data:geometry:vertical_tail:MAC:at25percent:x:local"].value[0]
    vt_distance_from_wing = variables[
        "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"
    ].value[0]
    x_vt = x_vt + wing_25mac_x + vt_distance_from_wing - local_vt_25mac_x
    z_vt = z_vt + fuselage_max_height / 4.0

    # CGs

    cg_aft_x = variables["data:weight:aircraft:CG:aft:x"].value[0]
    cg_fwd_x = variables["data:weight:aircraft:CG:fwd:x"].value[0]
    cg_empty_x = variables["data:weight:aircraft_empty:CG:x"].value[0]
    cg_empty_z = variables["data:weight:aircraft_empty:CG:z"].value[0]
    lg_height = variables["data:geometry:landing_gear:height"].value[0]

    x_cg = np.array([cg_fwd_x, cg_empty_x, cg_aft_x])
    z_cg = np.array([cg_empty_z, cg_empty_z, cg_empty_z])
    z_cg = z_cg - lg_height - fuselage_max_height / 2.0

    # Stability

    l0 = variables["data:geometry:wing:MAC:length"].value[0]
    mac_position = variables["data:geometry:wing:MAC:at25percent:x"].value[0]
    stick_fixed_sm = variables["data:handling_qualities:stick_fixed_static_margin"].value[0]
    stick_free_sm = variables["data:handling_qualities:stick_free_static_margin"].value[0]
    ac_ratio_fixed = variables["data:aerodynamics:cruise:neutral_point:stick_fixed:x"].value[0]
    ac_ratio_free = variables["data:aerodynamics:cruise:neutral_point:stick_free:x"].value[0]

    ac_fixed_x = mac_position + (ac_ratio_fixed - 0.25) * l0
    ac_free_x = mac_position + (ac_ratio_free - 0.25) * l0

    if fig is None:
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.7, 0.3],
            subplot_titles=("Aircraft Lateral View : Barycenter Position", "Zoom"),
        )
        scatter = go.Scatter(
            x=x_fuselage,
            y=z_fuselage,
            mode="lines+markers",
            name=name + " geometry",
            line=dict(color=color),
        )
        fig.add_trace(scatter, 1, 1)
        scatter = go.Scatter(
            x=x_vt,
            y=z_vt,
            mode="lines+markers",
            name=name,
            line=dict(color=color),
            showlegend=False,
        )
        fig.add_trace(scatter, 1, 1)
    else:
        scatter = go.Scatter(
            x=x_fuselage,
            y=z_fuselage,
            mode="lines+markers",
            name=name + " geometry",
            line=dict(color=color),
        )
        fig.add_trace(scatter, 1, 1)
        scatter = go.Scatter(
            x=x_vt,
            y=z_vt,
            mode="lines+markers",
            name=name,
            line=dict(color=color),
            showlegend=False,
        )
        fig.add_trace(scatter, 1, 1)

    scatter = go.Scatter(
        x=x_cg,
        y=z_cg,
        mode="lines+markers",
        name=name + " CG positions",
        line=dict(color=color, width=2),
        marker_line=dict(width=2),
    )
    fig.add_trace(scatter, 1, 1)
    scatter = go.Scatter(
        x=x_cg,
        y=z_cg,
        text=["fwd CG", "empty CG", "aft CG"],
        mode="lines+markers+text",
        textposition=["bottom center", "top center", "top center"],
        name=name + " CG positions",
        line={"dash": "dash"},
        marker_line=dict(width=2),
        line_color=color,
        showlegend=False,
    )
    fig.add_trace(scatter, 1, 2)

    scatter = go.Scatter(
        x=[ac_fixed_x],
        y=[z_cg[0]],
        text=" Neutral Point"
        + "<br>"
        + "Stick Fixed"
        + "<br>"
        + "Static Margin = "
        + str(round(stick_fixed_sm, 3)),
        textposition="bottom center",
        mode="markers+text",
        line=dict(color="DarkRed"),
        showlegend=False,
        marker_line=dict(width=2),
    )
    fig.add_trace(scatter, 1, 2)

    scatter = go.Scatter(
        x=[ac_free_x],
        y=[z_cg[0]],
        text="Neutral Point"
        + "<br>"
        + "Stick Free"
        + "<br>"
        + "Static Margin = "
        + str(round(stick_free_sm, 3)),
        textposition="bottom center",
        mode="markers+text",
        line=dict(color="DodgerBlue"),
        showlegend=False,
        marker_line=dict(width=2),
    )
    fig.add_trace(scatter, 1, 2)

    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_xaxes(title_text="X", row=1, col=2)
    fig.update_yaxes(title_text="Z", row=1, col=1)
    fig.update_yaxes(title_text="Z", row=1, col=2)

    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

    return fig


def _get_variable_values_with_new_units(
    variables: VariableList, var_names_and_new_units: Dict[str, str]
):
    """
    Returns the value of the requested variable names with respect to their new units in the order
    in which their were given. This function works only for variable of value with shape=1 or float.

    :param variables: instance containing variables information
    :param var_names_and_new_units: dictionary of the variable names as keys and units as value
    :return: values of the requested variables with respect to their new units
    """
    new_values = []
    for variable_name, unit in var_names_and_new_units.items():
        new_values.append(
            convert_units(variables[variable_name].value[0], variables[variable_name].units, unit,)
        )

    return new_values


def _data_weight_decomposition(variables: VariableList, owe=None):
    """
    Returns the two level weight decomposition of MTOW and optionally the decomposition of owe
    subcategories.

    :param variables: instance containing variables information
    :param owe: value of OWE, if provided names of owe subcategories will be provided
    :return: variable values, names and optionally owe subcategories names
    """
    category_values = []
    category_names = []
    owe_subcategory_names = []
    for variable in variables.names():
        name_split = variable.split(":")
        if isinstance(name_split, list) and len(name_split) == 4:
            if name_split[0] + name_split[1] + name_split[3] == "dataweightmass" and not (
                "aircraft" in name_split[2]
            ):
                category_values.append(
                    convert_units(variables[variable].value[0], variables[variable].units, "kg")
                )
                category_names.append(name_split[2])
                if owe:
                    owe_subcategory_names.append(
                        name_split[2]
                        + "<br>"
                        + str(int(variables[variable].value[0]))
                        + " [kg] ("
                        + str(round(variables[variable].value[0] / owe * 100, 1))
                        + "%)"
                    )
    if owe:
        result = category_values, category_names, owe_subcategory_names
    else:
        result = category_values, category_names, None

    return result


def mass_breakdown_bar_plot(
    aircraft_file_path: str, name=None, fig=None, file_formatter=None
) -> go.FigureWidget:
    """
    Returns a figure plot of the aircraft mass breakdown using bar plots.
    Different designs can be superposed by providing an existing fig.
    Each design can be provided a name.

    :param aircraft_file_path: path of data file
    :param name: name to give to the trace added to the figure
    :param fig: existing figure to which add the plot
    :param file_formatter: the formatter that defines the format of data file. If not provided,
                           default format will be assumed.
    :return: bar plot figure
    """
    variables = VariableIO(aircraft_file_path, file_formatter).read()

    var_names_and_new_units = {
        "data:weight:aircraft:MTOW": "kg",
        "data:weight:aircraft:OWE": "kg",
        "data:weight:aircraft:payload": "kg",
        "data:mission:sizing:fuel": "kg",
    }

    # pylint: disable=unbalanced-tuple-unpacking # It is balanced for the parameters provided
    mtow, owe, payload, fuel_mission = _get_variable_values_with_new_units(
        variables, var_names_and_new_units
    )

    if fig is None:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Maximum Take-Off Weight Breakdown", "Overall Weight Empty Breakdown"),
        )

    # Same color for each aircraft configuration
    i = len(fig.data)

    weight_labels = ["MTOW", "OWE", "Fuel - Mission", "Payload"]
    weight_values = [mtow, owe, fuel_mission, payload]
    fig.add_trace(
        go.Bar(name="", x=weight_labels, y=weight_values, marker_color=COLS[i], showlegend=False),
        row=1,
        col=1,
    )

    # Get data:weight decomposition
    main_weight_values, main_weight_names, _ = _data_weight_decomposition(variables, owe=None)
    fig.add_trace(
        go.Bar(name=name, x=main_weight_names, y=main_weight_values, marker_color=COLS[i]),
        row=1,
        col=2,
    )

    fig.update_layout(yaxis_title="[kg]")

    return fig


def mass_breakdown_sun_plot(aircraft_file_path: str, file_formatter=None):
    """
    Returns a figure sunburst plot of the mass breakdown.
    On the left a MTOW sunburst and on the right a OWE sunburst.
    Different designs can be superposed by providing an existing fig.
    Each design can be provided a name.

    :param aircraft_file_path: path of data file
    :param file_formatter: the formatter that defines the format of data file. If not provided,
                           default format will be assumed.
    :return: sunburst plot figure
    """
    variables = VariableIO(aircraft_file_path, file_formatter).read()

    var_names_and_new_units = {
        "data:weight:aircraft:MTOW": "kg",
        "data:weight:aircraft:OWE": "kg",
        "data:weight:aircraft:payload": "kg",
        "data:mission:sizing:fuel": "kg",
    }

    # pylint: disable=unbalanced-tuple-unpacking # It is balanced for the parameters provided
    mtow, owe, payload, onboard_fuel_at_takeoff = _get_variable_values_with_new_units(
        variables, var_names_and_new_units
    )

    # TODO: Deal with this in a more generic manner ?
    if round(mtow, 0) == round(owe + payload + onboard_fuel_at_takeoff, 0):
        mtow = owe + payload + onboard_fuel_at_takeoff

    fig = make_subplots(1, 2, specs=[[{"type": "domain"}, {"type": "domain"}]],)

    fig.add_trace(
        go.Sunburst(
            labels=[
                "MTOW" + "<br>" + str(int(mtow)) + " [kg]",
                "payload"
                + "<br>"
                + str(int(payload))
                + " [kg] ("
                + str(round(payload / mtow * 100, 1))
                + "%)",
                "onboard_fuel_at_takeoff"
                + "<br>"
                + str(int(onboard_fuel_at_takeoff))
                + " [kg] ("
                + str(round(onboard_fuel_at_takeoff / mtow * 100, 1))
                + "%)",
                "OWE" + "<br>" + str(int(owe)) + " [kg] (" + str(round(owe / mtow * 100, 1)) + "%)",
            ],
            parents=[
                "",
                "MTOW" + "<br>" + str(int(mtow)) + " [kg]",
                "MTOW" + "<br>" + str(int(mtow)) + " [kg]",
                "MTOW" + "<br>" + str(int(mtow)) + " [kg]",
            ],
            values=[mtow, payload, onboard_fuel_at_takeoff, owe],
            branchvalues="total",
        ),
        1,
        1,
    )

    # Get data:weight 2-levels decomposition
    categories_values, categories_names, categories_labels = _data_weight_decomposition(
        variables, owe=owe
    )

    sub_categories_values = []
    sub_categories_names = []
    sub_categories_parent = []
    for variable in variables.names():
        name_split = variable.split(":")
        if isinstance(name_split, list) and len(name_split) >= 5:
            parent_name = name_split[2]
            if parent_name in categories_names and name_split[-1] == "mass":
                variable_name = "_".join(name_split[3:-1])
                sub_categories_values.append(
                    convert_units(variables[variable].value[0], variables[variable].units, "kg")
                )
                sub_categories_parent.append(categories_labels[categories_names.index(parent_name)])
                sub_categories_names.append(variable_name)

    # Define figure data
    figure_labels = ["OWE" + "<br>" + str(int(owe)) + " [kg]"]
    figure_labels.extend(categories_labels)
    figure_labels.extend(sub_categories_names)
    figure_parents = [""]
    for _ in categories_names:
        figure_parents.append("OWE" + "<br>" + str(int(owe)) + " [kg]")
    figure_parents.extend(sub_categories_parent)
    figure_values = [owe]
    figure_values.extend(categories_values)
    figure_values.extend(sub_categories_values)

    # Plot figure
    fig.add_trace(
        go.Sunburst(
            labels=figure_labels,
            parents=figure_parents,
            values=figure_values,
            branchvalues="total",
        ),
        1,
        2,
    )

    fig.update_layout(title_text="Mass Breakdown", title_x=0.5)

    return fig


def payload_range(
    aircraft_file_path: str, name=None, fig=None, file_formatter=None
) -> go.FigureWidget:
    """
    Returns a figure plot of the top view of the plane.
    Different designs can be superposed by providing an existing fig.
    Each design can be provided a name.

    :param aircraft_file_path: path of data file
    :param name: name to give to the trace added to the figure
    :param fig: existing figure to which add the plot
    :param file_formatter: the formatter that defines the format of data file. If not provided, default format will
                           be assumed.
    :return: wing plot figure
    """
    variables = VariableIO(aircraft_file_path, file_formatter).read()

    payload_array = list(variables["data:payload_range:payload_array"].value)
    range_array = list(variables["data:payload_range:range_array"].value)
    sr_array = list(variables["data:payload_range:specific_range_array"].value)

    # If point D does not not exist, remove it
    if range_array[3] == 0:
        range_array = range_array[0:3] + [range_array[4]]
        payload_array = payload_array[0:3] + [payload_array[4]]
        sr_array = sr_array[0:3] + [sr_array[4]]
        text_plot = ["A" + "<br>" + "SR = " + str(round(sr_array[0], 1)), "B", "E"]
    else:
        text_plot = [
            "A" + "<br>" + "SR = " + str(round(sr_array[0], 1)),
            "B" + "<br>" + "SR = " + str(round(sr_array[1], 1)),
            "D" + "<br>" + "SR = " + str(round(sr_array[3], 1)),
            "E" + "<br>" + "SR = " + str(round(sr_array[4], 1)),
        ]

    # Plotting of the diagram
    if fig is None:
        fig = go.Figure()
    scatter = go.Scatter(
        x=range_array[0:2] + range_array[3:],
        y=payload_array[0:2] + payload_array[3:],
        mode="lines+markers+text",
        name=name + " Computed Points",
        text=text_plot,
        textposition="bottom right",
        textfont=dict(size=14),
    )
    fig.add_trace(scatter)
    scatter = go.Scatter(
        x=[range_array[2]],
        y=[payload_array[2]],
        mode="lines+markers+text",
        name=name + " Design Point",
        text=["C" + "<br>" + "SR = " + str(round(sr_array[2], 1))],
        textposition="bottom left",
        textfont=dict(size=14),
    )
    fig.add_trace(scatter)

    fig = go.FigureWidget(fig)

    fig.update_layout(
        title_text="Payload Range",
        title_x=0.5,
        xaxis_title="Range [NM]",
        yaxis_title="Payload [kg]",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def drag_breakdown_diagram(
    aircraft_file_path: str, fig=None, file_formatter=None,
) -> go.FigureWidget:
    """

    """
    variables = VariableIO(aircraft_file_path, file_formatter).read()

    parasite_drag_cruise = variables["data:aerodynamics:aircraft:cruise:CD0"].value[0]
    induced_drag_cruise = variables["data:aerodynamics:wing:cruise:induced_drag_coefficient"].value[
        0
    ]
    fuselage_drag_cruise = variables["data:aerodynamics:fuselage:cruise:CD0"].value[0]
    wing_parasite_drag_cruise = variables["data:aerodynamics:wing:cruise:CD0"].value[0]
    htp_drag_cruise = variables["data:aerodynamics:horizontal_tail:cruise:CD0"].value[0]
    vtp_drag_cruise = variables["data:aerodynamics:vertical_tail:cruise:CD0"].value[0]
    lg_drag_cruise = variables["data:aerodynamics:landing_gear:cruise:CD0"].value[0]
    nacelle_drag_cruise = variables["data:aerodynamics:nacelles:cruise:CD0"].value[0]
    other_drag_cruise = variables["data:aerodynamics:other:cruise:CD0"].value[0]

    parasite_drag_low_speed = variables["data:aerodynamics:aircraft:low_speed:CD0"].value[0]
    induced_drag_low_speed = variables[
        "data:aerodynamics:wing:low_speed:induced_drag_coefficient"
    ].value[0]
    fuselage_drag_low_speed = variables["data:aerodynamics:fuselage:low_speed:CD0"].value[0]
    wing_parasite_drag_low_speed = variables["data:aerodynamics:wing:low_speed:CD0"].value[0]
    htp_drag_low_speed = variables["data:aerodynamics:horizontal_tail:low_speed:CD0"].value[0]
    vtp_drag_low_speed = variables["data:aerodynamics:vertical_tail:low_speed:CD0"].value[0]
    lg_drag_low_speed = variables["data:aerodynamics:landing_gear:low_speed:CD0"].value[0]
    nacelle_drag_low_speed = variables["data:aerodynamics:nacelles:low_speed:CD0"].value[0]
    other_drag_low_speed = variables["data:aerodynamics:other:low_speed:CD0"].value[0]

    # CRUD (other undesirable drag). Factor from Gudmundsson book. Introduced in aerodynamics.components.cd0_total.py.
    crud_factor = 1.25

    if fig is None:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Drag coefficient breakdown in cruise conditions",
                "Drag coefficient breakdown in low_speed conditions",
            ),
            specs=[[{"type": "domain"}, {"type": "domain"}]],
        )

    fig.add_trace(
        go.Sunburst(
            labels=[
                "Parasite Drag",
                "Induced Drag",
                "Fuselage",
                "Wing",
                "Horizontal Tail",
                "Vertical Tail",
                "Landing Gears",
                "Nacelle",
                "Other",
            ],
            parents=[
                "",
                "",
                "Parasite Drag",
                "Parasite Drag",
                "Parasite Drag",
                "Parasite Drag",
                "Parasite Drag",
                "Parasite Drag",
                "Parasite Drag",
            ],
            values=[
                parasite_drag_cruise,
                induced_drag_cruise,
                crud_factor * fuselage_drag_cruise,
                crud_factor * wing_parasite_drag_cruise,
                crud_factor * htp_drag_cruise,
                crud_factor * vtp_drag_cruise,
                crud_factor * lg_drag_cruise,
                crud_factor * nacelle_drag_cruise,
                crud_factor * other_drag_cruise,
            ],
            branchvalues="total",
        ),
        1,
        1,
    )

    fig.add_trace(
        go.Sunburst(
            labels=[
                "Parasite Drag",
                "Induced Drag",
                "Fuselage",
                "Wing",
                "Horizontal Tail",
                "Vertical Tail",
                "Landing Gears",
                "Nacelle",
                "Other",
            ],
            parents=[
                "",
                "",
                "Parasite Drag",
                "Parasite Drag",
                "Parasite Drag",
                "Parasite Drag",
                "Parasite Drag",
                "Parasite Drag",
                "Parasite Drag",
            ],
            values=[
                parasite_drag_low_speed,
                induced_drag_low_speed,
                crud_factor * fuselage_drag_low_speed,
                crud_factor * wing_parasite_drag_low_speed,
                crud_factor * htp_drag_low_speed,
                crud_factor * vtp_drag_low_speed,
                crud_factor * lg_drag_low_speed,
                crud_factor * nacelle_drag_low_speed,
                crud_factor * other_drag_low_speed,
            ],
            branchvalues="total",
        ),
        1,
        2,
    )

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0),)

    fig = go.FigureWidget(fig)

    return fig
