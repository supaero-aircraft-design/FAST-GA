"""Estimation of wing L4 chords."""
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

from ..constants import SUBMODEL_WING_L4


@oad.RegisterSubmodel(SUBMODEL_WING_L4, "fastga.submodel.geometry.wing.l4.legacy")
class ComputeWingL4(om.ExplicitComponent):
    """Estimate l4 wing chord."""

    def setup(self):

        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")

        self.add_output("data:geometry:wing:tip:chord", units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]

        l4_wing = l1_wing * taper_ratio

        outputs["data:geometry:wing:tip:chord"] = l4_wing

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]

        partials["data:geometry:wing:tip:chord", "data:geometry:wing:taper_ratio"] = l1_wing
        partials[
            "data:geometry:wing:tip:chord", "data:geometry:wing:root:virtual_chord"
        ] = taper_ratio
