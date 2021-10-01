"""
Computation of wing area
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
import openmdao.api as om
import logging
from scipy.constants import g
import warnings

from fastoad.module_management.service_registry import RegisterOpenMDAOSystem
from fastoad.module_management.constants import ModelDomain
from scipy.optimize import fsolve

from fastga.models.aerodynamics.constants import ENGINE_COUNT

POINTS_NB_WING = 50

_LOGGER = logging.getLogger(__name__)


@RegisterOpenMDAOSystem("fastga.loop.wing_area", domain=ModelDomain.OTHER)
class UpdateWingArea(om.Group):
    """
    Computes needed wing area to:
      - have enough lift at required approach speed
      - be able to load enough fuel to achieve the sizing mission
    """

    def setup(self):
        self.add_subsystem("wing_area", _UpdateWingArea(), promotes=["*"])
        self.add_subsystem("constraints", _ComputeWingAreaConstraints(), promotes=["*"])


class _UpdateWingArea(om.ExplicitComponent):
    """ Computation of wing area from needed approach speed and mission fuel. For the mission wing area the code uses
     the fsolve algorithm on a function that computes the wing area following the same approach as in compute_mfw."""

    def setup(self):
        self.add_input("data:mission:sizing:fuel", val=np.nan, units="kg")
        self.add_input("data:propulsion:IC_engine:fuel_type", val=np.nan)
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:flap:chord_ratio", val=np.nan)
        self.add_input("data:geometry:flap:span_ratio", val=np.nan)
        self.add_input("data:geometry:aileron:chord_ratio", val=np.nan)
        self.add_input("data:geometry:aileron:span_ratio", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", units="m")
        self.add_input("data:geometry:propulsion:y_ratio_tank_beginning", val=np.nan)
        self.add_input("data:geometry:propulsion:y_ratio_tank_end", val=np.nan)
        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:geometry:propulsion:y_ratio", shape=ENGINE_COUNT, val=np.nan)
        self.add_input("data:geometry:propulsion:LE_chord_percentage", val=np.nan)
        self.add_input("data:geometry:propulsion:TE_chord_percentage", val=np.nan)
        self.add_input("data:geometry:propulsion:nacelle:width", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:type", val=np.nan)
        self.add_input("data:geometry:landing_gear:y", val=np.nan, units="m")
        self.add_input("settings:geometry:fuel_tanks:depth", val=np.nan)

        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)

        self.add_output("data:geometry:wing:area", val=10.0, units="m**2")

        self.declare_partials(
            "data:geometry:wing:area",
            [
                "data:mission:sizing:fuel",
                "data:geometry:wing:root:chord",
                "data:geometry:wing:tip:chord",
                "data:geometry:wing:root:thickness_ratio",
                "data:geometry:wing:tip:thickness_ratio",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mfw_mission = inputs["data:mission:sizing:fuel"]
        fuel_type = inputs["data:propulsion:IC_engine:fuel_type"]
        root_chord = inputs["data:geometry:wing:root:chord"]
        tip_chord = inputs["data:geometry:wing:tip:chord"]
        root_thickness_ratio = inputs["data:geometry:wing:root:thickness_ratio"]
        tip_thickness_ratio = inputs["data:geometry:wing:tip:thickness_ratio"]

        if fuel_type == 1.0:
            m_vol_fuel = 730  # gasoline volume-mass [kg/m**3], cold worst case
        elif fuel_type == 2.0:
            m_vol_fuel = 860  # gasoil volume-mass [kg/m**3], cold worst case
        elif fuel_type == 3.0:
            m_vol_fuel = 804.0  # Jet-A1 volume mass [kg/m**3], cold worst case
        else:
            m_vol_fuel = 730
            warnings.warn("Fuel type {} does not exist, replaced by type 1!".format(fuel_type))

        # Initial guess of the wing area value. For this guess the fuel tanks are assumed to be between 1st (30% MAC)
        # and 3rd (60% MAC) longeron: 30% of the wing.
        ave_thickness = (
            0.7 * (root_chord * root_thickness_ratio + tip_chord * tip_thickness_ratio) / 2.0
        )
        wing_area_mission_initial = (mfw_mission / m_vol_fuel) / (0.3 * ave_thickness)

        fuel_volume_one_wing_mission = mfw_mission / m_vol_fuel / 2
        wing_area_mission, dic, _, _ = fsolve(
            self.compute_wing_area,
            wing_area_mission_initial,
            args=(fuel_volume_one_wing_mission, inputs),
            xtol=0.01,
            full_output=True,
        )

        stall_speed = inputs["data:TLAR:v_approach"] / 1.3
        mlw = inputs["data:weight:aircraft:MLW"]
        max_cl = inputs["data:aerodynamics:aircraft:landing:CL_max"]
        wing_area_approach = 2 * mlw * g / (stall_speed ** 2) / (1.225 * max_cl)

        _LOGGER.info(
            "Looping on wing area with new value equal to {}".format(
                max(wing_area_mission, wing_area_approach)
            )
        )
        _LOGGER.info("Wing Area for mission fuel equal to {}".format(wing_area_mission))
        outputs["data:geometry:wing:area"] = max(wing_area_mission, wing_area_approach)

    @staticmethod
    def compute_wing_area(wing_area_parameter, fuel_volume_one_wing_mission, inputs):
        taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        lambda_wing = inputs["data:geometry:wing:aspect_ratio"]
        root_tc = inputs["data:geometry:wing:root:thickness_ratio"]
        tip_tc = inputs["data:geometry:wing:tip:thickness_ratio"]
        y_lg = inputs["data:geometry:landing_gear:y"]
        lg_type = inputs["data:geometry:landing_gear:type"]
        flap_chord_ratio = inputs["data:geometry:flap:chord_ratio"]
        flap_span_ratio = inputs["data:geometry:flap:span_ratio"]
        aileron_chord_ratio = inputs["data:geometry:aileron:chord_ratio"]
        fuselage_max_width = inputs["data:geometry:fuselage:maximum_width"]
        y_ratio_tank_beginning = inputs["data:geometry:propulsion:y_ratio_tank_beginning"]
        y_ratio_tank_end = inputs["data:geometry:propulsion:y_ratio_tank_end"]
        engine_config = inputs["data:geometry:propulsion:layout"]
        nacelle_width = inputs["data:geometry:propulsion:nacelle:width"]
        le_chord_percentage = inputs["data:geometry:propulsion:LE_chord_percentage"]
        te_chord_percentage = inputs["data:geometry:propulsion:TE_chord_percentage"]
        k = inputs["settings:geometry:fuel_tanks:depth"]

        # Computation of the wing needed geometry for compute_mfw
        span = np.sqrt(lambda_wing * wing_area_parameter)
        tip_y = span / 2.0
        root_y = fuselage_max_width / 2.0
        root_chord = wing_area_parameter / (2.0 * root_y + (tip_y - root_y) * (1.0 + taper_ratio))
        tip_chord = root_chord * taper_ratio

        semi_span = span / 2
        y_tank_beginning = semi_span * y_ratio_tank_beginning
        y_tank_end = semi_span * y_ratio_tank_end
        length_tank = y_tank_end - y_tank_beginning
        y_flap_end = fuselage_max_width / 2 + flap_span_ratio * semi_span
        y_array = np.linspace(y_tank_beginning, y_tank_end, POINTS_NB_WING)

        # Computation of the chord profile along the span, as chord = slope * y + chord_fuselage_center.
        slope_chord = (tip_chord - root_chord) / (tip_y - root_y)
        fuselage_center_virtual_chord = 0.5 * (
            root_chord + tip_chord - slope_chord * (root_y + tip_y)
        )
        chord_array = slope_chord * y_array + fuselage_center_virtual_chord

        # Computation of the thickness ratio profile along the span, as tc = slope * y + tc_fuselage_center.
        slope_tc = (tip_tc - root_tc) / (tip_y - root_y)
        fuselage_center_virtual_tc = 0.5 * (root_tc + tip_tc - slope_tc * (root_y + tip_y))
        thickness_ratio_array = slope_tc * y_array + fuselage_center_virtual_tc

        # The k factor stating the depth of the fuel tanks is included here.
        thickness_array = k * chord_array * thickness_ratio_array

        if engine_config != 1.0:
            y_ratio = 0.0
        else:
            y_ratio_data = inputs["data:geometry:propulsion:y_ratio"]
            used_index = np.where(y_ratio_data >= 0.0)[0]
            y_ratio = y_ratio_data[used_index]

        y_eng_array = semi_span * np.array(y_ratio)

        in_eng_nacelle = np.full(len(y_array), False)
        for y_eng in y_eng_array:
            for i in np.where(abs(y_array - y_eng) <= nacelle_width / 2.0):
                in_eng_nacelle[i] = True
        where_engine = np.where(in_eng_nacelle)

        # Computation of the fuel distribution along the span, taking in account the elements restricting it.
        # The fuel tanks go from the leading edge with a chord percentage offset to the nearest control surface
        # with another chord percentage offset.
        width_array = np.zeros((len(y_array), 1))
        for i in range(len(width_array)):
            if y_array[i] > y_flap_end:
                width_array[i] = (
                    1 - le_chord_percentage - te_chord_percentage - aileron_chord_ratio
                ) * chord_array[i]
            else:
                width_array[i] = (
                    1 - le_chord_percentage - te_chord_percentage - flap_chord_ratio
                ) * chord_array[i]
        if engine_config == 1.0:
            for i in where_engine:
                # For now 50% size reduction in the fuel tank capacity due to the engine
                width_array[i] = width_array[i] * 0.5
        if lg_type == 1.0:
            for i in np.where(y_array < y_lg):
                # For now 80% size reduction in the fuel tank capacity due to the landing gear
                width_array[i] = width_array[i] * 0.2

        area_array = thickness_array * width_array

        # Computation of the fuel volume available in one wing. The 0.85 coefficient represents the internal
        # obstructions caused by the structural and system components within the tank, typical of integral tankage.

        tank_volume_one_wing = (
            0.85
            * length_tank
            / (2 * (POINTS_NB_WING - 1))
            * (area_array[0] + 2 * np.sum(area_array[1:-1]) + area_array[-1])
        )

        return tank_volume_one_wing - fuel_volume_one_wing_mission


class _ComputeWingAreaConstraints(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:mission:sizing:fuel", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MFW", val=np.nan, units="kg")

        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("data:constraints:wing:additional_fuel_capacity", units="kg")
        self.add_output("data:constraints:wing:additional_CL_capacity")

        self.declare_partials(
            "data:constraints:wing:additional_fuel_capacity",
            ["data:weight:aircraft:MFW", "data:mission:sizing:fuel"],
            method="fd",
        )
        self.declare_partials(
            "data:constraints:wing:additional_CL_capacity",
            [
                "data:TLAR:v_approach",
                "data:weight:aircraft:MLW",
                "data:aerodynamics:aircraft:landing:CL_max",
                "data:geometry:wing:area",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mfw = inputs["data:weight:aircraft:MFW"]
        mission_fuel = inputs["data:mission:sizing:fuel"]
        v_stall = inputs["data:TLAR:v_approach"] / 1.3
        cl_max = inputs["data:aerodynamics:aircraft:landing:CL_max"]
        mlw = inputs["data:weight:aircraft:MLW"]
        wing_area = inputs["data:geometry:wing:area"]

        outputs["data:constraints:wing:additional_fuel_capacity"] = mfw - mission_fuel
        outputs["data:constraints:wing:additional_CL_capacity"] = cl_max - mlw * g / (
            0.5 * 1.225 * v_stall ** 2 * wing_area
        )
