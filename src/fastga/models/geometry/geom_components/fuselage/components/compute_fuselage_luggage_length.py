"""
Python module for fuselage luggage compartment length calculation, part of the fuselage dimension.
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


class ComputeFuselageLuggageLength(om.ExplicitComponent):
    """
    Computes luggage length.

    80% of internal radius section can be filled with luggage.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:luggage:mass_max", val=np.nan, units="kg")

        self.add_output("data:geometry:fuselage:luggage_length", units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        w_cabin = inputs["data:geometry:fuselage:maximum_width"] / 1.06
        luggage_mass_max = inputs["data:geometry:cabin:luggage:mass_max"]

        luggage_density = 161.0  # In kg/m3
        l_lug = (luggage_mass_max / luggage_density) / (0.2 * np.pi * w_cabin**2.0)

        outputs["data:geometry:fuselage:luggage_length"] = l_lug

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        w_cabin = inputs["data:geometry:fuselage:maximum_width"] / 1.06
        luggage_mass_max = inputs["data:geometry:cabin:luggage:mass_max"]

        luggage_density = 161.0  # In kg/m3
        l_lug = (luggage_mass_max / luggage_density) / (0.2 * np.pi * w_cabin**2.0)

        partials[
            "data:geometry:fuselage:luggage_length", "data:geometry:cabin:luggage:mass_max"
        ] = l_lug / luggage_mass_max

        partials[
            "data:geometry:fuselage:luggage_length", "data:geometry:fuselage:maximum_width"
        ] = -2.0 * l_lug / (w_cabin * 1.06)
