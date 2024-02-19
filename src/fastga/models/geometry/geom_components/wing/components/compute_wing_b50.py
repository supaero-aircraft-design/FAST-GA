"""Estimation of wing B50."""
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

from ..constants import SUBMODEL_WING_B50


@oad.RegisterSubmodel(SUBMODEL_WING_B50, "fastga.submodel.geometry.wing.half_span.legacy")
class ComputeWingB50(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Wing B50 estimation."""

    def setup(self):
        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        self.add_output("data:geometry:wing:b_50", units="m")

        self.declare_partials("data:geometry:wing:b_50", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        span = inputs["data:geometry:wing:span"]

        sweep_50 = np.arctan2((x4_wing + l4_wing * 0.5 - 0.5 * l1_wing), (y4_wing - y2_wing))
        b_50 = span / np.cos(sweep_50)

        outputs["data:geometry:wing:b_50"] = b_50
