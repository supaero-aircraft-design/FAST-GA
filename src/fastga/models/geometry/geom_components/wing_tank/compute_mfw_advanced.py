"""Estimation of max fuel weight."""
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

import warnings

import numpy as np

from scipy.integrate import trapz

from openmdao.core.explicitcomponent import ExplicitComponent

import fastoad.api as oad

from ...constants import SUBMODEL_MFW

POINTS_NB_WING = 50


@oad.RegisterSubmodel(SUBMODEL_MFW, "fastga.submodel.geometry.mfw.advanced")
class ComputeMFWAdvanced(ExplicitComponent):
    """
    Max fuel weight estimation based on Jenkinson 'Aircraft Design projects for Engineering
    Students' p.65. discretize the fuel tank in the wings along the span. Only works for linear
    chord and thickness profiles. The xml quantities
    "data:geometry:propulsion:tank:LE_chord_percentage",
    "data:geometry:propulsion:tank:TE_chord_percentage",
    "data:geometry:propulsion:tank:y_ratio_tank_beginning" and
    "data:geometry:propulsion:tank:y_ratio_tank_end" have to be determined as close to possible
    as the real aircraft quantities. The quantity "settings:geometry:fuel_tanks:depth" allows to
    calibrate the model for each aircraft. WARNING : If this class is updated, update_wing_area
    will have to be updated as well as it uses the same approach.
    """

    def setup(self):

        self.add_input("data:propulsion:fuel_type", val=np.nan)
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:flap:chord_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aileron:chord_ratio", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:y_ratio_tank_beginning", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:y_ratio_tank_end", val=np.nan)
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input(
            "data:geometry:propulsion:engine:y_ratio",
            shape_by_conn=True,
        )
        self.add_input("data:geometry:propulsion:tank:LE_chord_percentage", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:TE_chord_percentage", val=np.nan)
        self.add_input("data:geometry:propulsion:nacelle:width", val=np.nan, units="m")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:type", val=np.nan)
        self.add_input("data:geometry:landing_gear:y", val=np.nan, units="m")
        self.add_input("settings:geometry:fuel_tanks:depth", val=np.nan)

        self.add_output("data:weight:aircraft:MFW", units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fuel_type = inputs["data:propulsion:fuel_type"]
        y_ratio_tank_beginning = inputs["data:geometry:propulsion:tank:y_ratio_tank_beginning"]
        y_ratio_tank_end = inputs["data:geometry:propulsion:tank:y_ratio_tank_end"]
        span = inputs["data:geometry:wing:span"]

        if fuel_type == 1.0:
            m_vol_fuel = 718.9  # gasoline volume-mass [kg/m**3], cold worst case, Avgas
        elif fuel_type == 2.0:
            m_vol_fuel = 860.0  # Diesel volume-mass [kg/m**3], cold worst case
        elif fuel_type == 3.0:
            m_vol_fuel = 804.0  # Jet-A1 volume mass [kg/m**3], cold worst case
        else:
            m_vol_fuel = 718.9
            warnings.warn("Fuel type {} does not exist, replaced by type 1!".format(fuel_type))

        semi_span = span / 2
        y_tank_beginning = semi_span * y_ratio_tank_beginning
        y_tank_end = semi_span * y_ratio_tank_end
        y_array = np.linspace(y_tank_beginning, y_tank_end, POINTS_NB_WING)

        # Computation of the fuel volume available in one wing. The 0.85 coefficient represents
        # the internal obstructions caused by the structural and system components within the
        # tank, typical of integral tanks.

        area_array = tank_volume_distribution(inputs, y_array)

        tank_volume_one_wing = trapz(area_array, y_array.flatten())

        tank_volume = tank_volume_one_wing * 2

        maximum_fuel_weight = tank_volume * m_vol_fuel

        outputs["data:weight:aircraft:MFW"] = maximum_fuel_weight


def tank_volume_distribution(inputs, y_array_orig):

    root_chord = inputs["data:geometry:wing:root:chord"]
    tip_chord = inputs["data:geometry:wing:tip:chord"]
    root_y = inputs["data:geometry:wing:root:y"]
    tip_y = inputs["data:geometry:wing:tip:y"]
    root_tc = inputs["data:geometry:wing:root:thickness_ratio"]
    tip_tc = inputs["data:geometry:wing:tip:thickness_ratio"]
    flap_chord_ratio = inputs["data:geometry:flap:chord_ratio"]
    aileron_chord_ratio = inputs["data:geometry:wing:aileron:chord_ratio"]
    y_ratio_tank_beginning = inputs["data:geometry:propulsion:tank:y_ratio_tank_beginning"]
    y_ratio_tank_end = inputs["data:geometry:propulsion:tank:y_ratio_tank_end"]
    engine_config = inputs["data:geometry:propulsion:engine:layout"]
    le_chord_percentage = inputs["data:geometry:propulsion:tank:LE_chord_percentage"]
    te_chord_percentage = inputs["data:geometry:propulsion:tank:TE_chord_percentage"]
    nacelle_width = inputs["data:geometry:propulsion:nacelle:width"]
    span = inputs["data:geometry:wing:span"]
    lg_type = inputs["data:geometry:landing_gear:type"]
    y_lg = inputs["data:geometry:landing_gear:y"]
    k = inputs["settings:geometry:fuel_tanks:depth"]

    # Create the array which will contain the tank cross section area at each section

    y_array = y_array_orig.flatten()

    semi_span = span / 2
    y_tank_beginning = semi_span * y_ratio_tank_beginning
    y_tank_end = semi_span * y_ratio_tank_end

    y_in_tank_index = np.where((y_array >= y_tank_beginning) & (y_array <= y_tank_end))[0]
    y_in_tank_array = y_array[y_in_tank_index]

    slope_chord = (tip_chord - root_chord) / (tip_y - root_y)

    chord_array = np.zeros_like(y_array)
    for y in y_in_tank_array:
        idx = np.where(y_array == y)[0]
        if y < root_y:
            chord_array[idx] = root_chord
        else:
            chord_array[idx] = slope_chord * y + root_chord

    # Computation of the thickness ratio profile along the span, as tc = slope * y +
    # tc_fuselage_center.
    slope_tc = (tip_tc - root_tc) / (tip_y - root_y)
    # fuselage_center_virtual_tc = 0.5 * (root_tc + tip_tc - slope_tc * (root_y + tip_y))
    thickness_ratio_array = np.zeros_like(y_array)
    for y in y_in_tank_array:
        idx = np.where(y_array == y)[0]
        if y < root_y:
            thickness_ratio_array[idx] = root_tc
        else:
            thickness_ratio_array[idx] = slope_tc * y + root_tc
    # thickness_ratio_array = slope_tc * y_array + fuselage_center_virtual_tc

    # The k factor stating the depth of the fuel tanks is included here.
    thickness_array = k * chord_array * thickness_ratio_array

    # Distributed propulsion / single engine on the wing / nose or rear mounted engine
    if engine_config != 1.0:
        y_ratio = 0.0
    else:
        y_ratio = inputs["data:geometry:propulsion:engine:y_ratio"]

    y_eng_array = semi_span * np.array(y_ratio)

    in_eng_nacelle = np.full(len(y_in_tank_array), False)
    for y_eng in y_eng_array:
        for i in np.where(abs(y_in_tank_array - y_eng) <= nacelle_width / 2.0):
            in_eng_nacelle[i] = True
    where_engine = np.where(in_eng_nacelle)

    width_array = (
        1.0 - le_chord_percentage - te_chord_percentage - max(flap_chord_ratio, aileron_chord_ratio)
    ) * chord_array
    if engine_config == 1.0:
        for i in where_engine:
            # For now 50% size reduction in the fuel tank capacity due to the engine
            width_array[i] *= 0.5
    if lg_type == 1.0:
        for i in np.where(y_array < y_lg):
            # For now 80% size reduction in the fuel tank capacity due to the landing gear
            width_array[i] *= 0.2

    tank_cross_section = thickness_array * width_array * 0.85

    return tank_cross_section
