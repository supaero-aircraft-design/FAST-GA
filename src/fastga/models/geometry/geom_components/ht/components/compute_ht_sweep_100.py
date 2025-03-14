"""
Python module for horizontal tail sweep angle calculation at 100% of the MAC, part of the
horizontal tail sweep angle.
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


class ComputeHTSweep100(om.ExplicitComponent):
    """Estimation of horizontal tail sweep at 100% of the MAC."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")

        self.add_output("data:geometry:horizontal_tail:sweep_100", units="rad")

        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        b_h = inputs["data:geometry:horizontal_tail:span"]
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]

        half_span = b_h / 2.0
        sweep_100 = np.pi / 2.0 - np.arctan(
            half_span / (half_span * np.tan(sweep_25) - 0.75 * root_chord + 0.75 * tip_chord)
        )

        outputs["data:geometry:horizontal_tail:sweep_100"] = sweep_100

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        b_h = inputs["data:geometry:horizontal_tail:span"]
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]

        half_span = b_h / 2.0

        common_denominator = (
            b_h**2.0 + (b_h * np.tan(sweep_25) - 1.5 * root_chord + 1.5 * tip_chord) ** 2.0
        )

        partials[
            "data:geometry:horizontal_tail:sweep_100", "data:geometry:horizontal_tail:span"
        ] = (1.5 * root_chord - 1.5 * tip_chord) / common_denominator

        partials[
            "data:geometry:horizontal_tail:sweep_100", "data:geometry:horizontal_tail:root:chord"
        ] = -(3.0 * half_span) / common_denominator
        partials[
            "data:geometry:horizontal_tail:sweep_100", "data:geometry:horizontal_tail:tip:chord"
        ] = (3.0 * half_span) / common_denominator
        partials[
            "data:geometry:horizontal_tail:sweep_100", "data:geometry:horizontal_tail:sweep_25"
        ] = b_h**2.0 / np.cos(sweep_25) ** 2.0 / common_denominator
