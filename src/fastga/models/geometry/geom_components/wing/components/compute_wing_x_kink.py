"""Estimation of wing kink X local."""
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

from ..constants import SUBMODEL_WING_X_LOCAL_KINK


@oad.RegisterSubmodel(
    SUBMODEL_WING_X_LOCAL_KINK, "fastga.submodel.geometry.wing.x_local.kink.legacy"
)
class ComputeWingXKink(om.ExplicitComponent):
    """Wing kink X local estimation."""

    # TODO: Document equations. Cite sources

    def setup(self):

        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")

        self.add_output("data:geometry:wing:kink:leading_edge:x:local", units="m")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        y2_wing = inputs["data:geometry:wing:root:y"]
        y3_wing = inputs["data:geometry:wing:kink:y"]
        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]
        l3_wing = inputs["data:geometry:wing:kink:chord"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]

        x3_wing = 1.0 / 4.0 * l1_wing + (y3_wing - y2_wing) * np.tan(sweep_25) - 1.0 / 4.0 * l3_wing

        outputs["data:geometry:wing:kink:leading_edge:x:local"] = x3_wing

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        y2_wing = inputs["data:geometry:wing:root:y"]
        y3_wing = inputs["data:geometry:wing:kink:y"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]

        partials[
            "data:geometry:wing:kink:leading_edge:x:local", "data:geometry:wing:root:y"
        ] = -np.tan(sweep_25)
        partials[
            "data:geometry:wing:kink:leading_edge:x:local", "data:geometry:wing:kink:y"
        ] = np.tan(sweep_25)
        partials[
            "data:geometry:wing:kink:leading_edge:x:local", "data:geometry:wing:root:virtual_chord"
        ] = 0.25
        partials[
            "data:geometry:wing:kink:leading_edge:x:local", "data:geometry:wing:kink:chord"
        ] = -0.25
        partials["data:geometry:wing:kink:leading_edge:x:local", "data:geometry:wing:sweep_25"] = -(
            np.tan(sweep_25) ** 2 + 1
        ) * (y2_wing - y3_wing)
