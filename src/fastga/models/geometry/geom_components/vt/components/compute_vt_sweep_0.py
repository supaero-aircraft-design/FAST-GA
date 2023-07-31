"""Estimation of vertical tail sweeps."""
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

from ..constants import SUBMODEL_VT_SWEEP_0


# TODO: HT and VT components are similar --> factorize
@oad.RegisterSubmodel(SUBMODEL_VT_SWEEP_0, "fastga.submodel.geometry.vertical_tail.sweep_0.legacy")
class ComputeVTSweep0(om.ExplicitComponent):
    """Estimation of vertical tail sweep at l/c=0%"""

    def setup(self):

        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="deg")

        self.add_output("data:geometry:vertical_tail:sweep_0", units="deg")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        sweep_25 = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        sweep_0 = (
            (
                np.pi / 2
                - np.arctan2(
                    b_v,
                    (0.25 * root_chord - 0.25 * tip_chord + b_v * np.tan(sweep_25 / 180.0 * np.pi)),
                )
            )
            / np.pi
            * 180.0
        )

        outputs["data:geometry:vertical_tail:sweep_0"] = sweep_0

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        sweep_25 = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        tmp = root_chord / 4 - tip_chord / 4 + b_v * np.tan((sweep_25 * np.pi) / 180)

        partials[
            "data:geometry:vertical_tail:sweep_0", "data:geometry:vertical_tail:root:chord"
        ] = (45 * b_v) / (np.pi * (b_v ** 2 / tmp ** 2 + 1) * tmp ** 2)
        partials[
            "data:geometry:vertical_tail:sweep_0", "data:geometry:vertical_tail:tip:chord"
        ] = -(45 * b_v) / (np.pi * (b_v ** 2 / tmp ** 2 + 1) * tmp ** 2)
        partials["data:geometry:vertical_tail:sweep_0", "data:geometry:vertical_tail:sweep_25"] = (
            b_v ** 2 * (np.tan((np.pi * sweep_25) / 180) ** 2 + 1)
        ) / ((b_v ** 2 / tmp ** 2 + 1) * tmp ** 2)
        partials["data:geometry:vertical_tail:sweep_0", "data:geometry:vertical_tail:span"] = -(
            180
            * (
                1 / (root_chord / 4 - tip_chord / 4 + b_v * np.tan((np.pi * sweep_25) / 180))
                - (b_v * np.tan((np.pi * sweep_25) / 180)) / tmp ** 2
            )
        ) / (np.pi * (b_v ** 2 / tmp ** 2 + 1))
