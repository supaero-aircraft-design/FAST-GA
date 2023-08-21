"""Estimation of wing sweep at l/c=50%."""
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

from ..constants import SUBMODEL_WING_SWEEP_50


@oad.RegisterSubmodel(SUBMODEL_WING_SWEEP_50, "fastga.submodel.geometry.wing.sweep.50.legacy")
class ComputeWingSweep50(om.ExplicitComponent):
    """Estimation of wing sweep at l/c=50%"""

    def setup(self):

        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_0", val=np.nan, units="rad")

        self.add_output("data:geometry:wing:sweep_50", units="rad")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        taper_ratio_wing = inputs["data:geometry:wing:taper_ratio"]
        sweep_0 = inputs["data:geometry:wing:sweep_0"]

        outputs["data:geometry:wing:sweep_50"] = np.arctan(
            np.tan(sweep_0) - 2.0 / wing_ar * ((1.0 - taper_ratio_wing) / (1.0 + taper_ratio_wing))
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        taper_ratio_wing = inputs["data:geometry:wing:taper_ratio"]
        sweep_0 = inputs["data:geometry:wing:sweep_0"]

        partials["data:geometry:wing:sweep_50", "data:geometry:wing:aspect_ratio"] = -(
            2.0 * (taper_ratio_wing - 1.0)
        ) / (
            wing_ar ** 2.0
            * (
                (
                    np.tan(sweep_0)
                    + (2.0 * (taper_ratio_wing - 1.0)) / (wing_ar * (taper_ratio_wing + 1.0))
                )
                ** 2.0
                + 1.0
            )
            * (taper_ratio_wing + 1.0)
        )
        partials["data:geometry:wing:sweep_50", "data:geometry:wing:taper_ratio"] = (
            2.0 / (wing_ar * (taper_ratio_wing + 1.0))
            - (2.0 * (taper_ratio_wing - 1.0)) / (wing_ar * (taper_ratio_wing + 1.0) ** 2.0)
        ) / (
            (
                np.tan(sweep_0)
                + (2.0 * (taper_ratio_wing - 1.0)) / (wing_ar * (taper_ratio_wing + 1.0))
            )
            ** 2.0
            + 1.0
        )
        partials["data:geometry:wing:sweep_50", "data:geometry:wing:sweep_0"] = (
            np.tan(sweep_0) ** 2.0 + 1.0
        ) / (
            (
                np.tan(sweep_0)
                + (2.0 * (taper_ratio_wing - 1.0)) / (wing_ar * (taper_ratio_wing + 1.0))
            )
            ** 2.0
            + 1.0
        )
