"""
    Estimation of horizontal tail sweep at l/c=0%.
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


class ComputeHTSweep0(om.ExplicitComponent):
    """Estimation of horizontal tail sweep at l/c=0%"""

    def setup(self):

        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")

        self.add_output("data:geometry:horizontal_tail:sweep_0", units="rad")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        b_h = inputs["data:geometry:horizontal_tail:span"]
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]

        half_span = b_h / 2.0

        sweep_0 = np.pi / 2.0 - np.arctan2(
            half_span, (0.25 * root_chord - 0.25 * tip_chord + half_span * np.tan(sweep_25))
        )

        outputs["data:geometry:horizontal_tail:sweep_0"] = sweep_0

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        b_h = inputs["data:geometry:horizontal_tail:span"]
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]
        chord_diff = root_chord-tip_chord
        length_constant = 4.0*b_h**2.0 + (chord_diff + 2.0*b_h* np.tan(sweep_25))**2

        partials["data:geometry:horizontal_tail:sweep_0", "data:geometry:horizontal_tail:span"] = (
            - 2.0 * chord_diff / length_constant
        )
        partials[
            "data:geometry:horizontal_tail:sweep_0", "data:geometry:horizontal_tail:root:chord"
        ] = 2.0 * b_h / length_constant
        partials[
            "data:geometry:horizontal_tail:sweep_0", "data:geometry:horizontal_tail:tip:chord"
        ] = -2.0 * b_h / length_constant
        partials[
            "data:geometry:horizontal_tail:sweep_0", "data:geometry:horizontal_tail:sweep_25"
        ] = 4.0 * b_h**2 * np.cos(sweep_25)**-2.0 / length_constant