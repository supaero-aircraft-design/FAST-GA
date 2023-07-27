"""Estimation of wing kink Y (sections span)."""
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

from ..constants import SUBMODEL_WING_Y_KINK


@oad.RegisterSubmodel(SUBMODEL_WING_Y_KINK, "fastga.submodel.geometry.wing.y.kink.legacy")
class ComputeWingYKink(om.ExplicitComponent):
    """Wing kink Ys estimation."""

    # TODO: Document equations. Cite sources

    def setup(self):

        self.add_input("data:geometry:wing:kink:span_ratio", val=0.5)
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")

        self.add_output("data:geometry:wing:kink:y", units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_break = inputs["data:geometry:wing:kink:span_ratio"]
        y4_wing = inputs["data:geometry:wing:tip:y"]

        y3_wing = y4_wing * wing_break

        outputs["data:geometry:wing:kink:y"] = y3_wing

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        wing_break = inputs["data:geometry:wing:kink:span_ratio"]
        y4_wing = inputs["data:geometry:wing:tip:y"]

        partials["data:geometry:wing:kink:y", "data:geometry:wing:kink:span_ratio"] = y4_wing
        partials["data:geometry:wing:kink:y", "data:geometry:wing:tip:y"] = wing_break
