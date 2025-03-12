"""
Python module for fuselage master cross-section calculation, part of the fuselage dimension.
"""

#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
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


class ComputeFuselageMasterCrossSection(om.ExplicitComponent):
    """
    Computes master cross-section of the fuselage by multiplying the maximum height and width of
    fuselage.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")

        self.add_output("data:geometry:fuselage:master_cross_section", units="m**2")

        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]

        master_cross_section = np.pi * (np.sqrt(b_f * h_f) / 2.0) ** 2.0

        outputs["data:geometry:fuselage:master_cross_section"] = master_cross_section

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]

        partials[
            "data:geometry:fuselage:master_cross_section", "data:geometry:fuselage:maximum_width"
        ] = (np.pi * h_f) / 4.0
        partials[
            "data:geometry:fuselage:master_cross_section", "data:geometry:fuselage:maximum_height"
        ] = (np.pi * b_f) / 4.0
