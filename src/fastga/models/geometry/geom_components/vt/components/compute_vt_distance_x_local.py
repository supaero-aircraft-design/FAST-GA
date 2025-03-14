"""
Python module for the calculation of vertical tail distance from 25% wing MAC with fixed fuselage
length, part of the vertical tail geometry.
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


class ComputeVTMACDistanceXLocal(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Vertical tail mean aerodynamic chord X-position estimation of vertical tail w.r.t. leading
    edge of root chord.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:vertical_tail:MAC:at25percent:x:local", units="m")

        self.declare_partials("*", "*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        x0_vt = (
            (root_chord * 0.25 + b_v * np.tan(sweep_25_vt) - tip_chord * 0.25)
            * (root_chord + 2.0 * tip_chord)
            / (3.0 * (root_chord + tip_chord))
        )

        outputs["data:geometry:vertical_tail:MAC:at25percent:x:local"] = x0_vt

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        chord_sum = tip_chord + root_chord

        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:local",
            "data:geometry:vertical_tail:root:chord",
        ] = (
            chord_sum**2.0 + 2.0 * tip_chord**2.0 - 4.0 * b_v * tip_chord * np.tan(sweep_25_vt)
        ) / (12.0 * chord_sum**2.0)
        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:local",
            "data:geometry:vertical_tail:tip:chord",
        ] = (
            2.0 * root_chord * b_v * np.tan(sweep_25_vt)
            - tip_chord**2.0
            - 2.0 * root_chord * tip_chord
        ) / (6.0 * chord_sum**2.0)

        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:local",
            "data:geometry:vertical_tail:sweep_25",
        ] = b_v * (chord_sum + tip_chord) / (3.0 * chord_sum) / np.cos(sweep_25_vt) ** 2.0

        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:local",
            "data:geometry:vertical_tail:span",
        ] = (np.tan(sweep_25_vt) * (chord_sum + tip_chord)) / (3.0 * chord_sum)
