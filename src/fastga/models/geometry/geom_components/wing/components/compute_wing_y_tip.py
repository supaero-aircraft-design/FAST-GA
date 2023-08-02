"""Estimation of wing tip Y (sections span)."""
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

from ..constants import SUBMODEL_WING_Y_TIP


@oad.RegisterSubmodel(SUBMODEL_WING_Y_TIP, "fastga.submodel.geometry.wing.y.tip.legacy")
class ComputeWingYTip(om.ExplicitComponent):
    """Wing tip Y estimation."""

    # TODO: Document equations. Cite sources

    def setup(self):

        self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        self.add_output("data:geometry:wing:tip:y", units="m")

        self.declare_partials(of="*", wrt="*", val=0.5)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        span = inputs["data:geometry:wing:span"]

        y4_wing = span / 2.0

        outputs["data:geometry:wing:tip:y"] = y4_wing