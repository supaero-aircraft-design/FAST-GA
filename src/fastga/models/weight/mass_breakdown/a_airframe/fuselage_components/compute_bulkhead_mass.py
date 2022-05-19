"""Computes the mass of the bulkhead, adapted from Torenbeek by Lucas REMOND."""
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


class ComputeBulkhead(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:max_differential_pressure", val=np.nan, units="hPa")
        self.add_input(
            "data:geometry:cabin:pressurized",
            val=0.0,
            desc="Cabin pressurization; 0.0 for no pressurization, 1.0 for pressurization",
        )

        self.add_output("data:weight:airframe:fuselage:bulkhead:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuselage_max_width = inputs["data:geometry:fuselage:maximum_width"]
        fuselage_max_height = inputs["data:geometry:fuselage:maximum_height"]
        pressurized = inputs["data:geometry:cabin:pressurized"]
        # Converting to kg/cm**2, can't be done by OpenMDAO
        delta_p_max = inputs["data:geometry:cabin:max_differential_pressure"] * 0.00102

        fuselage_radius = np.sqrt(fuselage_max_height * fuselage_max_width) / 2.0

        if pressurized:
            mass_bulkhead = 9.1 + 12.48 * delta_p_max ** 0.8 * np.pi * fuselage_radius ** 2
        else:
            mass_bulkhead = 0
        mass_bulkheads = 2 * mass_bulkhead

        outputs["data:weight:airframe:fuselage:bulkhead:mass"] = mass_bulkheads
