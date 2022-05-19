"""Computes the mass of the tail cone, adapted from TASOPT by Lucas REMOND."""
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


class ComputeFloor(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

        self.add_input(
            "settings:weight:airframe:fuselage:floor:area_density", units="kg/m**2", val=4.62
        )

        self.add_output("data:weight:airframe:fuselage:floor:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Floor width is not exactly equal to the fuselage max width
        floor_width = inputs["data:geometry:fuselage:maximum_width"] * 0.9
        cabin_length = inputs["data:geometry:cabin:length"]
        floor_density = inputs["settings:weight:airframe:fuselage:floor:area_density"]

        floor_area = floor_width * cabin_length
        floor_weight = floor_density * floor_area ** 1.045

        outputs["data:weight:airframe:fuselage:floor:mass"] = floor_weight
