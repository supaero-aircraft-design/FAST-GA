"""
    Estimation of max fuel weight
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
from openmdao.core.explicitcomponent import ExplicitComponent
import warnings

POINTS_NB_WING = 50


class ComputeMFW(ExplicitComponent):

    """
    Max fuel weight estimation based on Jenkinson 'Aircraft Design projects for Engineering Students' p.65
    Only works for linear chord and thickness profiles.
    """

    def setup(self):

        self.add_input("data:propulsion:IC_engine:fuel_type", val=np.nan)
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:flap:chord_ratio", val=np.nan)
        self.add_input("data:geometry:flap:span_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aileron:chord_ratio", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", units="m")
        self.add_input("data:geometry:propulsion:y_ratio_tank_end", val=np.nan)
        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:geometry:propulsion:y_ratio", val=np.nan)
        self.add_input("data:geometry:propulsion:nacelle:width", val=np.nan, units="m")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:type", val=np.nan)
        self.add_input("data:geometry:landing_gear:y", val=np.nan, units="m")

        self.add_output("data:weight:aircraft:new_MFW", units="kg")

        self.declare_partials("data:weight:aircraft:new_MFW", [
            "data:geometry:wing:root:chord",
            "data:geometry:wing:tip:chord",
            "data:geometry:wing:root:thickness_ratio",
            "data:geometry:wing:tip:thickness_ratio",
            ], method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fuel_type = inputs["data:propulsion:IC_engine:fuel_type"]
        root_chord = inputs["data:geometry:wing:root:chord"]
        tip_chord = inputs["data:geometry:wing:tip:chord"]
        root_thickness_ratio = inputs["data:geometry:wing:root:thickness_ratio"]
        tip_thickness_ratio = inputs["data:geometry:wing:tip:thickness_ratio"]
        flap_chord_ratio = inputs["data:geometry:flap:chord_ratio"]
        flap_span_ratio = inputs["data:geometry:flap:span_ratio"]
        aileron_chord_ratio = inputs["data:geometry:wing:aileron:chord_ratio"]
        fuselage_max_width = inputs["data:geometry:fuselage:maximum_width"]
        y_ratio_tank_end = inputs["data:geometry:propulsion:y_ratio_tank_end"]
        engine_config = inputs["data:geometry:propulsion:layout"]
        y_ratio_engine = inputs["data:geometry:propulsion:y_ratio"]
        nacelle_width = inputs["data:geometry:propulsion:nacelle:width"]
        lg_type = inputs["data:geometry:landing_gear:type"]
        y_lg = inputs["data:geometry:landing_gear:y"]

        span = inputs["data:geometry:wing:span"]

        if fuel_type == 1.0:
            m_vol_fuel = 718.9  # gasoline volume-mass [kg/m**3], cold worst case, Avgas
        elif fuel_type == 2.0:
            m_vol_fuel = 860.  # Diesel volume-mass [kg/m**3], cold worst case
        elif fuel_type == 3.0:
            m_vol_fuel = 804.  # Jet-A1 volume mass [kg/m**3], cold worst case
        else:
            m_vol_fuel = 718.9
            warnings.warn("Fuel type {} does not exist, replaced by type 1!".format(fuel_type))

        # Factor to relate the average tank depth to the max wing profile depth. The value depends on the shape of
        # the section profile and the allowance made for structure. Typical values lie between 0.5 and 0.8
        k = 0.6

        semi_span = span / 2
        y_tank_end = semi_span * y_ratio_tank_end
        length_tank = y_tank_end - fuselage_max_width / 2
        y_flap_end = fuselage_max_width / 2 + flap_span_ratio * semi_span
        y_array = np.linspace(fuselage_max_width / 2, y_tank_end, POINTS_NB_WING)
        #
        virtual_chord_center = (tip_chord - root_chord) / (semi_span - fuselage_max_width / 2) *\
                               (-fuselage_max_width / 2) + root_chord
        virtual_thickness_ratio_center = (tip_thickness_ratio - root_thickness_ratio) / (semi_span - fuselage_max_width / 2) * \
                                         (-fuselage_max_width / 2) + root_thickness_ratio

        chord_array = (tip_chord - root_chord) / (semi_span - fuselage_max_width / 2) * y_array + virtual_chord_center
        thickness_ratio_array = (tip_thickness_ratio - root_thickness_ratio) / (semi_span - fuselage_max_width / 2) *\
                                y_array + virtual_thickness_ratio_center
        thickness_array = k * chord_array * thickness_ratio_array
        width_array = []

        if engine_config == 1.0:
            y_engine = y_ratio_engine * semi_span
        else:
            y_engine = 0

        for i in range(len(y_array)):
            if y_array[i] > y_flap_end:
                width_i = (1 - 0.11 - 0.05 - aileron_chord_ratio) * chord_array[i]
            else:
                width_i = (1 - 0.11 - 0.05 - flap_chord_ratio) * chord_array[i]
            if engine_config == 1.0:
                if abs(y_array[i] - y_engine) <= nacelle_width / 2:
                    # For now 50% size reduction in the fuel tank capacity due to the engine
                    width_i = width_i * 0.5
            if lg_type == 1.0:
                if y_array[i] < y_lg:
                    # For now 80% size reduction in the fuel tank capacity due to the landing gear
                    width_i = width_i * 0.2
            width_array.append(width_i)

        # Computation of the three areas
        area_array = thickness_array * width_array

        # Computation of the fuel volume available in one wing. The 0.85 coefficient represents the internal
        # obstructions caused by the structural and system components within the tank

        tank_volume_one_wing = 0.85 * length_tank / (2 * (POINTS_NB_WING - 1)) *\
                               (area_array[0] + 2 * np.sum(area_array[1:-1]) + area_array[-1])

        tank_volume = tank_volume_one_wing * 2

        maximum_fuel_weight = tank_volume * m_vol_fuel

        outputs["data:weight:aircraft:new_MFW"] = maximum_fuel_weight
