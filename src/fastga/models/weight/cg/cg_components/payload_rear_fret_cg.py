"""
    Estimation of payload (rear fret) center of gravity.
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

from .constants import SUBMODEL_PAYLOAD_REAR_FRET_CG


@oad.RegisterSubmodel(
    SUBMODEL_PAYLOAD_REAR_FRET_CG, "fastga.submodel.weight.cg.payload.rear_fret.legacy"
)
class ComputeRearFretCG(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Rear fret center of gravity estimation"""

    def setup(self):

        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:PAX_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:luggage_length", val=np.nan, units="m")

        self.add_output("data:weight:payload:rear_fret:CG:x", units="m")

        self.declare_partials(
            "data:weight:payload:rear_fret:CG:x", "data:geometry:fuselage:front_length", val=1.0
        )
        self.declare_partials(
            "data:weight:payload:rear_fret:CG:x", "data:geometry:fuselage:PAX_length", val=1.0
        )
        self.declare_partials(
            "data:weight:payload:rear_fret:CG:x", "data:geometry:fuselage:luggage_length", val=0.5
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        lav = inputs["data:geometry:fuselage:front_length"]
        lpax = inputs["data:geometry:fuselage:PAX_length"]
        l_lug = inputs["data:geometry:fuselage:luggage_length"]

        # Instruments length
        l_instr = 0.7

        x_cg_r_fret = lav + l_instr + lpax + l_lug / 2

        outputs["data:weight:payload:rear_fret:CG:x"] = x_cg_r_fret
