"""FAST - Copyright (c) 2022 ONERA ISAE."""

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
import math

import numpy as np
import openmdao.api as om
import fastoad.api as oad
from openmdao.core.explicitcomponent import ExplicitComponent
from stdatm import Atmosphere

from ..constants import SUBMODEL_CD0_INLETS


@oad.RegisterSubmodel(SUBMODEL_CD0_INLETS, "fastga.submodel.aerodynamics.inlets.cd0.legacy")
class Cd0Inlets(ExplicitComponent):
    """
    Drag estimation for the inlets

    Based on : Drag and pressure recovery characteristics of auxiliary air inlets at
    subsonic speeds, 2013.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:propulsion:fuelcell:air_flow", val=np.nan, units="slug/s")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="ft/s")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="ft")
        self.add_input("data:aerodynamics:wing:cruise:reynolds", val=np.nan)
        self.add_input("data:geometry:fuselage:number_of_inlets", val=np.nan)
        if self.options["low_speed_aero"]:
            self.add_output("data:aerodynamics:inlets:low_speed:CD0")
        else:
            self.add_output("data:aerodynamics:inlets:cruise:CD0")

            self.add_output("data:geometry:inlets:width", units="ft")
            self.add_output("data:geometry:inlets:throat:length", units="ft")
            self.add_output("data:geometry:inlets:throat:width", units="ft")
            self.add_output("data:geometry:inlets:maximum_height", units="ft")
            self.add_output("data:geometry:inlets:lip_height", units="ft")
            self.add_output("data:geometry:inlets:area", units="ft**2")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]
        rho_cruise = Atmosphere(cruise_alt, altitude_in_feet=True).density
        v_cruise = inputs["data:TLAR:v_cruise"]
        reynolds_number = inputs["data:geometry:wing:cruise:reynolds"]
        fuselage_length = inputs["data:geometry:fuselage:length"]
        fuelcell_airflow = inputs["data:geometry:propulsion:fuelcell:air_flow"]
        inlets = inputs["data:geometry:fuselage:number_of_inlets"]

        speed_of_sound = Atmosphere(cruise_alt, altitude_in_feet=False).speed_of_sound
        mach = v_cruise / speed_of_sound

        alpha = 7  # inlet ramp angle is set at 7 for max. efficiency

        boundary_layer_thickness = (0.37 * (0.6 * fuselage_length)) / reynolds_number ** (1/5)
        momentum_thickness = boundary_layer_thickness / 6   # approximation

        drag_parameter = fuelcell_airflow / (rho_cruise * v_cruise * momentum_thickness ** 2)
        dt = 1.203 * momentum_thickness * (drag_parameter ** 0.415)

        inlet_width = 4 * dt  # aspect ratio is 4
        throat_length = 0.25 * dt
        throat_thickness = 0.25 * dt
        max_external_height = dt + throat_thickness - throat_length * math.tan(alpha)
        lip_height = dt + 0.5 * throat_thickness - throat_length * math.tan(alpha)
        inlet_area = lip_height * inlet_width

        # Drag Computation

        k_sigma = 0.8   # value for alpha = 7 deg
        mass_flow_ratio = 0.1651 * (dt / lip_height) * (momentum_thickness / dt) ** -0.4068
        thickness_ratio = boundary_layer_thickness / lip_height
        momentum_flow_ratio = (1.01456 - 0.26997 * thickness_ratio - 0.02344 * thickness_ratio * mach
                               + 0.03991 * thickness_ratio ** 2)
        ram_drag = 2 * k_sigma * momentum_flow_ratio * mass_flow_ratio

        k_alpha = 1  # from figure 12 for alpha = 7
        if mach <= 0.55:
            k_m = 1
        else:
            k_m = 1.1

        if mass_flow_ratio < 0.2:
            k_spfl = 1
        elif mass_flow_ratio >= 0.4:
            k_spfl = 0
        elif 0.2 <= mass_flow_ratio < 0.3:
            k_spfl = 0.2
        elif 0.3 <= mass_flow_ratio < 0.4:
            k_spfl = 0.1

        cd_fl = 0.159  # from figure 11
        spillage_drag = k_alpha * k_m * k_spfl * cd_fl

        incremental_drag = 0  # approximation from figure 15 for general aviation aircraft

        total_drag = ram_drag + spillage_drag + incremental_drag
        inlet_drag = total_drag * inlets

        outputs["data:geometry:inlets:width"] = inlet_width
        outputs["data:geometry:inlets:maximum_height"] = max_external_height
        outputs["data:geometry:inlets:lip_height"] = lip_height
        outputs["data:geometry:inlets:area"] = inlet_area
        outputs["data:geometry:inlets:throat:length"] = throat_length
        outputs["data:geometry:inlets:throat:width"] = throat_thickness

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:inlets:low_speed:CD0"] = inlet_drag
        else:
            outputs["data:aerodynamics:inlets:cruise:CD0"] = inlet_drag





