"""Computes the mass of the doors, adapted from TASOPT by Lucas REMOND."""
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


class ComputeDoors(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:geometry:cabin:doors:number", val=2.0)
        self.add_input("data:geometry:cabin:doors:height", val=1.0, units="m")
        self.add_input("data:geometry:cabin:doors:width", val=0.61, units="m")
        self.add_input("data:geometry:cabin:max_differential_pressure", val=np.nan, units="hPa")
        self.add_input(
            "data:weight:airframe:fuselage:shell:area_density", val=np.nan, units="kg/m**2"
        )
        self.add_input(
            "data:geometry:cabin:pressurized",
            val=0.0,
            desc="Cabin pressurization; 0.0 for no pressurization, 1.0 for pressurization",
        )

        self.add_output("data:weight:airframe:fuselage:doors:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        doors_number = inputs["data:geometry:cabin:doors:number"]
        doors_height = inputs["data:geometry:cabin:doors:height"]
        doors_width = inputs["data:geometry:cabin:doors:width"]
        # Converting to kg/cm**2, can't be done by OpenMDAO
        delta_p_max = inputs["data:geometry:cabin:max_differential_pressure"] * 0.00102
        shell_area_density = inputs["data:weight:airframe:fuselage:shell:area_density"]
        pressurized = inputs["data:geometry:cabin:pressurized"]

        door_area = doors_height * doors_width

        if pressurized:
            door_filling_mass = 44.2 * delta_p_max ** 0.5 * door_area
            door_surround_mass = 22.3 * door_area ** 0.5
            removed_mass = shell_area_density * door_area
            door_weight = max(door_surround_mass + door_filling_mass - removed_mass, 0.0)
        else:
            door_filling_mass = 9.765 * door_area
            door_surround_mass = 14.9 * door_area ** 0.5
            removed_mass = shell_area_density * door_area
            door_weight = max(door_surround_mass + door_filling_mass - removed_mass, 0.0)

        doors_weight = door_weight * doors_number

        outputs["data:weight:airframe:fuselage:doors:mass"] = doors_weight
