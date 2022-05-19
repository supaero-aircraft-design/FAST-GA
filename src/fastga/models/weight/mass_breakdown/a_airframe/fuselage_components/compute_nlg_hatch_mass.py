"""Computes the mass of the nose landing gear, adapted from TASOPT by Lucas REMOND."""
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

import openmdao.api as om

import numpy as np


class ComputeNLGHatch(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:geometry:landing_gear:height", val=np.nan, units="m")
        self.add_input(
            "data:geometry:cabin:pressurized",
            val=0.0,
            desc="Cabin pressurization; 0.0 for no pressurization, 1.0 for pressurization",
        )
        self.add_input(
            "data:weight:airframe:fuselage:shell:area_density", val=np.nan, units="kg/m**2"
        )

        self.add_output("data:weight:airframe:fuselage:nlg_hatch:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        lg_height = inputs["data:geometry:landing_gear:height"]
        pressurized = inputs["data:geometry:cabin:pressurized"]
        shell_mass_area_density = inputs["data:weight:airframe:fuselage:shell:area_density"]

        door_height = 1.1 * lg_height
        door_width = 0.3
        if pressurized:
            mass_door_component = 22.0 * door_height * door_width
        else:
            mass_door_component = 16.1 * door_height * door_width

        mass_surrounds = 9.97 * (door_width * door_height) ** 0.5
        mass_removed = door_width * door_height * shell_mass_area_density
        mass_nlg_door = max(mass_door_component + mass_surrounds - mass_removed, 0.0)

        outputs["data:weight:airframe:fuselage:nlg_hatch:mass"] = mass_nlg_door
