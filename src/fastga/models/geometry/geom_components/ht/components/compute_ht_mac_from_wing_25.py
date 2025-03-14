"""
Python module for horizontal tail MAC distance from the 25% wing MAC calculation with fixed
rear fuselage length, part of the horizontal tail geometry.
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


class ComputeHTMACFromWing25(om.ExplicitComponent):
    """
    Horizontal tail MAC distance from the 25% wing MAC estimation with (F)ixed rear fuselage
    (L)ength.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:local", val=np.nan, units="m"
        )
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:has_T_tail", val=np.nan)
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:absolute", val=np.nan, units="m"
        )
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        self.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", units="m")

        self.declare_partials("*", "*", method="exact")
        self.declare_partials(of="*", wrt="data:geometry:has_T_tail", method="fd")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        fus_length = inputs["data:geometry:fuselage:length"]
        tail_type = inputs["data:geometry:has_T_tail"]
        x_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:absolute"]
        x_wing25 = inputs["data:geometry:wing:MAC:at25percent:x"]
        x0_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:local"]

        if tail_type == 1.0:
            ht_lp = (x_ht + x0_ht) - x_wing25
        else:
            ht_lp = (fus_length - root_chord + x0_ht) - x_wing25

        outputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"] = ht_lp

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        tail_type = inputs["data:geometry:has_T_tail"]

        if tail_type == 1.0:
            partials[
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:horizontal_tail:root:chord",
            ] = 0.0
            partials[
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:fuselage:length",
            ] = 0.0
            partials[
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:horizontal_tail:MAC:at25percent:x:absolute",
            ] = 1.0
            partials[
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            ] = 1.0
            partials[
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:wing:MAC:at25percent:x",
            ] = -1.0
        else:
            partials[
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:horizontal_tail:root:chord",
            ] = -1.0
            partials[
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:horizontal_tail:MAC:at25percent:x:absolute",
            ] = 0.0
            partials[
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:fuselage:length",
            ] = 1.0
            partials[
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:wing:MAC:at25percent:x",
            ] = -1.0
            partials[
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            ] = 1.0
