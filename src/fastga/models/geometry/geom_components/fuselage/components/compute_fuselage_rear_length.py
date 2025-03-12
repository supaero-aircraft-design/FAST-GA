"""
Python module for fuselage rear length calculation, part of the fuselage dimension with FD.
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


class ComputeFuselageRearLength(om.ExplicitComponent):
    """
    Computes fuselage rear length, this calculation is only applied while calculating fuselage
    geometry with constant distance between wing MAC and tail MAC.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

        self.add_output("data:geometry:fuselage:rear_length", units="m")

        self.declare_partials(of="*", wrt="data:geometry:fuselage:length", val=1.0)
        self.declare_partials(
            of="*",
            wrt=["data:geometry:fuselage:front_length", "data:geometry:cabin:length"],
            val=-1.0,
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        cabin_length = inputs["data:geometry:cabin:length"]

        lar = fus_length - (lav + cabin_length)

        outputs["data:geometry:fuselage:rear_length"] = lar
