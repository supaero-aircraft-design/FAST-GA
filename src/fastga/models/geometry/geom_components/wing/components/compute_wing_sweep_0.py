"""Estimation of wing sweep at l/c=0%."""
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

from ..constants import SUBMODEL_WING_SWEEP_0


@oad.RegisterSubmodel(SUBMODEL_WING_SWEEP_0, "fastga.submodel.geometry.wing.sweep.0")
class ComputeWingSweep0(om.ExplicitComponent):
    """Estimation of wing sweep at l/c=0%"""

    def setup(self):

        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")

        self.add_output("data:geometry:wing:sweep_0", units="rad")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]

        sweep_0 = np.arctan2(x4_wing, (y4_wing - y2_wing))

        outputs["data:geometry:wing:sweep_0"] = sweep_0

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]

        tmp = y2_wing - y4_wing

        partials[
            "data:geometry:wing:sweep_0", "data:geometry:wing:tip:leading_edge:x:local"
        ] = -1 / ((x4_wing ** 2 / tmp ** 2 + 1) * tmp)
        partials["data:geometry:wing:sweep_0", "data:geometry:wing:root:y"] = x4_wing / (
            (x4_wing ** 2 / tmp ** 2 + 1) * tmp ** 2
        )
        partials["data:geometry:wing:sweep_0", "data:geometry:wing:tip:y"] = -x4_wing / (
            (x4_wing ** 2 / tmp ** 2 + 1) * tmp ** 2
        )
