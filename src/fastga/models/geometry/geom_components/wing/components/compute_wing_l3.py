"""Estimation of wing L3 chords."""
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

from ..constants import SUBMODEL_WING_L3


@oad.RegisterSubmodel(SUBMODEL_WING_L3, "fastga.submodel.geometry.wing.l3")
class ComputeWingL3(om.ExplicitComponent):
    """Estimate l3 wing chord."""

    def setup(self):

        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")

        self.add_output("data:geometry:wing:kink:chord", units="m")

        self.declare_partials(
            "data:geometry:wing:kink:chord", "data:geometry:wing:root:chord", val=1.0
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        l2_wing = inputs["data:geometry:wing:root:chord"]

        l3_wing = l2_wing

        outputs["data:geometry:wing:kink:chord"] = l3_wing
