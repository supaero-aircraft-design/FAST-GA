"""Estimation of wing outer area."""
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

from ..constants import SUBMODEL_WING_OUTER_AREA


@oad.RegisterSubmodel(SUBMODEL_WING_OUTER_AREA, "fastga.submodel.geometry.wing.outer_area.legacy")
class ComputeWingOuterArea(om.ExplicitComponent):
    """Wing outer area estimation."""

    def setup(self):

        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")

        self.add_output("data:geometry:wing:outer_area", units="m**2")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_area = inputs["data:geometry:wing:area"]
        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]
        y1_wing = inputs["data:geometry:fuselage:maximum_width"] / 2.0

        s_pf = wing_area - 2.0 * l1_wing * y1_wing

        outputs["data:geometry:wing:outer_area"] = s_pf

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]
        y1_wing = inputs["data:geometry:fuselage:maximum_width"] / 2.0

        partials["data:geometry:wing:outer_area", "data:geometry:wing:area"] = 1.0
        partials["data:geometry:wing:outer_area", "data:geometry:wing:root:virtual_chord"] = (
            -2.0 * y1_wing
        )
        partials["data:geometry:wing:outer_area", "data:geometry:fuselage:maximum_width"] = -l1_wing
