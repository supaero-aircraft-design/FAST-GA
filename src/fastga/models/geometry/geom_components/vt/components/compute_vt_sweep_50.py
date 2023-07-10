"""Estimation of vertical tail sweeps."""
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

from ..constants import SUBMODEL_VT_SWEEP_50


# TODO: HT and VT components are similar --> factorize
@oad.RegisterSubmodel(SUBMODEL_VT_SWEEP_50, "fastga.submodel.geometry.vertical_tail.sweep_50.legacy")
class ComputeVTSweep50(om.ExplicitComponent):
    """Estimation of vertical tail sweep at l/c=50%"""

    def setup(self):

        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:sweep_0", val=np.nan, units="deg")

        self.add_output("data:geometry:vertical_tail:sweep_50", units="rad")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]
        sweep_0 = inputs["data:geometry:vertical_tail:sweep_0"]

        sweep_50 = np.arctan(
            np.tan(sweep_0 * np.pi / 180) - 2 / ar_vt * ((1 - taper_vt) / (1 + taper_vt))
        )

        outputs["data:geometry:vertical_tail:sweep_50"] = sweep_50
