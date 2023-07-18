"""
    Estimation of vertical tail root chord
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

from ..constants import SUBMODEL_VT_SPAN


@oad.RegisterSubmodel(SUBMODEL_VT_SPAN, "fastga.submodel.geometry.vertical_tail.span.legacy")
class ComputeVTSpan(om.ExplicitComponent):
    """
    Estimates vertical tail span
    """

    def setup(self):

        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="m**2")

        self.add_output("data:geometry:vertical_tail:span", units="m")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        lambda_vt = float(inputs["data:geometry:vertical_tail:aspect_ratio"])
        s_v = float(inputs["data:geometry:vertical_tail:area"])

        # Give a minimum value to avoid 0 division later if s_v initialised to 0
        b_v = np.sqrt(max(lambda_vt * s_v, 0.1))

        outputs["data:geometry:vertical_tail:span"] = b_v

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        lambda_vt = float(inputs["data:geometry:vertical_tail:aspect_ratio"])
        s_v = float(inputs["data:geometry:vertical_tail:area"])

        if lambda_vt * s_v >= 0.1:

            partials[
                "data:geometry:vertical_tail:span", "data:geometry:vertical_tail:aspect_ratio"
            ] = s_v / (2 * np.sqrt(lambda_vt * s_v))
            partials[
                "data:geometry:vertical_tail:span", "data:geometry:vertical_tail:area"
            ] = lambda_vt / (2 * np.sqrt(lambda_vt * s_v))

        else:

            partials[
                "data:geometry:vertical_tail:span", "data:geometry:vertical_tail:aspect_ratio"
            ] = 0.0
            partials["data:geometry:vertical_tail:span", "data:geometry:vertical_tail:area"] = 0.0
