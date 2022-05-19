"""Computes the mass of the insulation, adapted from TASOPT by Lucas REMOND."""
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


class ComputeInsulation(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:geometry:fuselage:wet_area", val=np.nan, units="m**2")
        self.add_input("settings:materials:insulation:area_density", val=0.488243, units="kg/m**2")

        self.add_output("data:weight:airframe:fuselage:insulation:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Default value of insulation comes from https://www.fire.tc.faa.gov/pdf/insulate.pdf

        outputs["data:weight:airframe:fuselage:insulation:mass"] = (
            inputs["data:geometry:fuselage:wet_area"]
            * inputs["settings:materials:insulation:area_density"]
        )
