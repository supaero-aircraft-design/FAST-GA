"""
    Estimation of horizontal tail sweep at l/c=50%.
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

from ..constants import SUBMODEL_HT_SWEEP_50


@oad.RegisterSubmodel(
    SUBMODEL_HT_SWEEP_50, "fastga.submodel.geometry.horizontal_tail.sweep_50.legacy"
)
class ComputeHTSweep50(om.ExplicitComponent):
    """Estimation of horizontal tail sweep at l/c=50%"""

    def setup(self):

        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:sweep_0", val=np.nan, units="rad")

        self.add_output("data:geometry:horizontal_tail:sweep_50", units="rad")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ar_ht = inputs["data:geometry:horizontal_tail:aspect_ratio"]
        taper_ht = inputs["data:geometry:horizontal_tail:taper_ratio"]
        sweep_0 = inputs["data:geometry:horizontal_tail:sweep_0"]

        sweep_50 = np.arctan(np.tan(sweep_0) - 2 / ar_ht * ((1 - taper_ht) / (1 + taper_ht)))

        outputs["data:geometry:horizontal_tail:sweep_50"] = sweep_50

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        ar_ht = inputs["data:geometry:horizontal_tail:aspect_ratio"]
        taper_ht = inputs["data:geometry:horizontal_tail:taper_ratio"]
        sweep_0 = inputs["data:geometry:horizontal_tail:sweep_0"]

        partials[
            "data:geometry:horizontal_tail:sweep_50", "data:geometry:horizontal_tail:aspect_ratio"
        ] = -(2 * (taper_ht - 1)) / (
            ar_ht ** 2
            * ((np.tan(sweep_0) + (2 * (taper_ht - 1)) / (ar_ht * (taper_ht + 1))) ** 2 + 1)
            * (taper_ht + 1)
        )
        partials[
            "data:geometry:horizontal_tail:sweep_50", "data:geometry:horizontal_tail:taper_ratio"
        ] = (
            2 / (ar_ht * (taper_ht + 1)) - (2 * (taper_ht - 1)) / (ar_ht * (taper_ht + 1) ** 2)
        ) / (
            (np.tan(sweep_0) + (2 * (taper_ht - 1)) / (ar_ht * (taper_ht + 1))) ** 2 + 1
        )
        partials[
            "data:geometry:horizontal_tail:sweep_50", "data:geometry:horizontal_tail:sweep_0"
        ] = (np.tan(sweep_0) ** 2 + 1) / (
            (np.tan(sweep_0) + (2 * (taper_ht - 1)) / (ar_ht * (taper_ht + 1))) ** 2 + 1
        )
