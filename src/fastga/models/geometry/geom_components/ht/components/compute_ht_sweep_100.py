"""
    Estimation of horizontal tail sweep at l/c=100%.
"""

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

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from ..constants import SUBMODEL_HT_SWEEP_100


@oad.RegisterSubmodel(
    SUBMODEL_HT_SWEEP_100, "fastga.submodel.geometry.horizontal_tail.sweep_100.legacy"
)
class ComputeHTSweep100(om.ExplicitComponent):
    """Estimation of horizontal tail sweep at l/c=100%"""

    def setup(self):

        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")

        self.add_output("data:geometry:horizontal_tail:sweep_100", units="rad")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        b_h = inputs["data:geometry:horizontal_tail:span"]
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]

        half_span = b_h / 2.0
        sweep_100 = np.pi / 2 - np.arctan(
            half_span / (half_span * np.tan(sweep_25) - 0.75 * root_chord + 0.75 * tip_chord)
        )

        outputs["data:geometry:horizontal_tail:sweep_100"] = sweep_100

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        b_h = inputs["data:geometry:horizontal_tail:span"]
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]

        half_span = b_h / 2.0

        tmp = (3 * tip_chord) / 4 - (3 * root_chord) / 4 + half_span * np.tan(sweep_25)

        partials[
            "data:geometry:horizontal_tail:sweep_100", "data:geometry:horizontal_tail:span"
        ] = (
            -(1 / tmp - (half_span * np.tan(sweep_25)) / tmp ** 2)
            / (half_span ** 2 / tmp ** 2 + 1)
            / 2
        )
        partials[
            "data:geometry:horizontal_tail:sweep_100", "data:geometry:horizontal_tail:root:chord"
        ] = -(3 * half_span) / (4 * (half_span ** 2 / tmp ** 2 + 1) * tmp ** 2)
        partials[
            "data:geometry:horizontal_tail:sweep_100", "data:geometry:horizontal_tail:tip:chord"
        ] = (3 * half_span) / (4 * (half_span ** 2 / tmp ** 2 + 1) * tmp ** 2)
        partials[
            "data:geometry:horizontal_tail:sweep_100", "data:geometry:horizontal_tail:sweep_25"
        ] = (half_span ** 2 * (np.tan(sweep_25) ** 2 + 1)) / (
            (half_span ** 2 / tmp ** 2 + 1) * tmp ** 2
        )
