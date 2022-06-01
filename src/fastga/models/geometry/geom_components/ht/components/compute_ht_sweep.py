"""
    Estimation of horizontal tail sweeps and aspect ratio.
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

import math
import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent
import fastoad.api as oad

from ..constants import SUBMODEL_HT_SWEEP


@oad.RegisterSubmodel(SUBMODEL_HT_SWEEP, "fastga.submodel.geometry.horizontal_tail.sweep.legacy")
class ComputeHTSweep(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Horizontal tail sweeps and aspect ratio estimation"""

    def setup(self):
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")

        self.add_output("data:geometry:horizontal_tail:sweep_0", units="rad")
        self.add_output("data:geometry:horizontal_tail:sweep_100", units="rad")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        b_h = inputs["data:geometry:horizontal_tail:span"]
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]

        half_span = b_h / 2.0
        # TODO: The unit conversion can be handled by OpenMDAO
        sweep_0 = math.pi / 2 - math.atan(
            half_span / (0.25 * root_chord - 0.25 * tip_chord + half_span * math.tan(sweep_25))
        )
        sweep_100 = math.pi / 2 - math.atan(
            half_span / (half_span * math.tan(sweep_25) - 0.75 * root_chord + 0.75 * tip_chord)
        )

        outputs["data:geometry:horizontal_tail:sweep_0"] = sweep_0
        outputs["data:geometry:horizontal_tail:sweep_100"] = sweep_100

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        b_h = inputs["data:geometry:horizontal_tail:span"]
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]

        half_span = b_h / 2.0

        tmp_0 = half_span / (0.25 * root_chord - 0.25 * tip_chord + half_span * math.tan(sweep_25))
        tmp_100 = half_span / (
            half_span * math.tan(sweep_25) - 0.75 * root_chord + 0.75 * tip_chord
        )

        d_tmp_0_d_half_span = (
            (0.25 * root_chord - 0.25 * tip_chord + half_span * math.tan(sweep_25))
            - half_span * math.tan(sweep_25)
        ) / tmp_0 ** 2.0
        d_tmp_0_d_rc = -(tmp_0 ** 2.0) / half_span * 0.25
        d_tmp_0_d_tc = (tmp_0 ** 2.0) / half_span * 0.25
        d_tmp_0_d_sweep = -(tmp_0 ** 2.0) * (1.0 + math.tan(sweep_25) ** 2.0)

        d_tmp_100_d_half_span = (
            (half_span * math.tan(sweep_25) - 0.75 * root_chord + 0.75 * tip_chord)
            - half_span * math.tan(sweep_25)
        ) / tmp_100 ** 2.0
        d_tmp_100_d_rc = (tmp_100 ** 2.0) / half_span * 0.75
        d_tmp_100_d_tc = -(tmp_100 ** 2.0) / half_span * 0.75
        d_tmp_100_d_sweep = -(tmp_100 ** 2.0) * (1.0 + math.tan(sweep_25) ** 2.0)

        partials["data:geometry:horizontal_tail:sweep_0", "data:geometry:horizontal_tail:span"] = (
            -1.0 / (1.0 + tmp_0 ** 2.0) * d_tmp_0_d_half_span / 2.0
        )
        partials[
            "data:geometry:horizontal_tail:sweep_0", "data:geometry:horizontal_tail:root:chord"
        ] = (-1.0 / (1.0 + tmp_0 ** 2.0) * d_tmp_0_d_rc)
        partials[
            "data:geometry:horizontal_tail:sweep_0", "data:geometry:horizontal_tail:tip:chord"
        ] = (-1.0 / (1.0 + tmp_0 ** 2.0) * d_tmp_0_d_tc)
        partials[
            "data:geometry:horizontal_tail:sweep_0", "data:geometry:horizontal_tail:sweep_25"
        ] = (-1.0 / (1.0 + tmp_0 ** 2.0) * d_tmp_0_d_sweep)

        partials[
            "data:geometry:horizontal_tail:sweep_100", "data:geometry:horizontal_tail:span"
        ] = (-1.0 / (1.0 + tmp_100 ** 2.0) * d_tmp_100_d_half_span / 2.0)
        partials[
            "data:geometry:horizontal_tail:sweep_100", "data:geometry:horizontal_tail:root:chord"
        ] = (-1.0 / (1.0 + tmp_100 ** 2.0) * d_tmp_100_d_rc)
        partials[
            "data:geometry:horizontal_tail:sweep_100", "data:geometry:horizontal_tail:tip:chord"
        ] = (-1.0 / (1.0 + tmp_100 ** 2.0) * d_tmp_100_d_tc)
        partials[
            "data:geometry:horizontal_tail:sweep_100", "data:geometry:horizontal_tail:sweep_25"
        ] = (-1.0 / (1.0 + tmp_100 ** 2.0) * d_tmp_100_d_sweep)
