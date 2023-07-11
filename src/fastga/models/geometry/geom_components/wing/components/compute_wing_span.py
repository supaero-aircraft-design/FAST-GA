"""Estimation of wing span."""
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

from ..constants import SUBMODEL_WING_SPAN


@oad.RegisterSubmodel(SUBMODEL_WING_SPAN, "fastga.submodel.geometry.wing.span.legacy")
class ComputeWingSpan(om.ExplicitComponent):
    """Wing span estimation."""

    # TODO: Document equations. Cite sources

    def setup(self):

        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("data:geometry:wing:span", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        lambda_wing = inputs["data:geometry:wing:aspect_ratio"]
        wing_area = inputs["data:geometry:wing:area"]

        span = np.sqrt(lambda_wing * wing_area)

        outputs["data:geometry:wing:span"] = span
