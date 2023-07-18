"""
    Estimation of payload (pax) center of gravity.
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

from .constants import SUBMODEL_PAYLOAD_PAX_CG


@oad.RegisterSubmodel(SUBMODEL_PAYLOAD_PAX_CG, "fastga.submodel.weight.cg.payload.pax.legacy")
class ComputePaxCG(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Pasenger center of gravity estimation.

    Passengers gravity center identical to seats
    """

    def setup(self):

        self.add_input("data:weight:furniture:passenger_seats:CG:x", val=np.nan, units="m")

        self.add_output("data:weight:payload:PAX:CG:x", units="m")

        self.declare_partials("*", "*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        x_cg_d2 = inputs["data:weight:furniture:passenger_seats:CG:x"]

        x_cg_pax = x_cg_d2

        outputs["data:weight:payload:PAX:CG:x"] = x_cg_pax
