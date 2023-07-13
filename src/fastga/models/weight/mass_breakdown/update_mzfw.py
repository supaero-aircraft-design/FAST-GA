"""
Maximum Zero Fuel Weight (MZFW) estimation.
"""
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

from .constants import SUBMODEL_MZFW


@oad.RegisterSubmodel(SUBMODEL_MZFW, "fastga.submodel.weight.mass.mzfw.legacy")
class ComputeMZFW(om.ExplicitComponent):
    """
    Computes Maximum Zero Fuel Weight from Overall Empty Weight and Maximum Payload.
    """

    def setup(self):

        self.add_input("data:weight:aircraft:OWE", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:max_payload", val=np.nan, units="kg")

        self.add_output("data:weight:aircraft:MZFW", units="kg")

        self.declare_partials("data:weight:aircraft:MZFW", "data:weight:aircraft:OWE", val=1.0)
        self.declare_partials(
            "data:weight:aircraft:MZFW", "data:weight:aircraft:max_payload", val=1.0
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        owe = inputs["data:weight:aircraft:OWE"]
        max_pl = inputs["data:weight:aircraft:max_payload"]

        mzfw = owe + max_pl

        outputs["data:weight:aircraft:MZFW"] = mzfw
