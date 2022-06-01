"""
    Estimation of payload center(s) of gravity.
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
from openmdao.core.explicitcomponent import ExplicitComponent

import fastoad.api as oad

from .constants import SUBMODEL_PAYLOAD_CG


@oad.RegisterSubmodel(SUBMODEL_PAYLOAD_CG, "fastga.submodel.weight.cg.payload.legacy")
class ComputePayloadCG(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Payload center(s) of gravity estimation"""

    def setup(self):

        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:weight:furniture:passenger_seats:CG:x", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:PAX_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:luggage_length", val=np.nan, units="m")

        self.add_output("data:weight:payload:PAX:CG:x", units="m")
        self.add_output("data:weight:payload:rear_fret:CG:x", units="m")
        self.add_output("data:weight:payload:front_fret:CG:x", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        lav = inputs["data:geometry:fuselage:front_length"]
        x_cg_d2 = inputs["data:weight:furniture:passenger_seats:CG:x"]
        lpax = inputs["data:geometry:fuselage:PAX_length"]
        l_lug = inputs["data:geometry:fuselage:luggage_length"]

        # Passengers gravity center identical to seats
        x_cg_pax = x_cg_d2
        # Instruments length
        l_instr = 0.7
        # Fret center of gravity
        x_cg_f_fret = lav * 0.0  # ???: should be defined somewhere in the CAB
        x_cg_r_fret = lav + l_instr + lpax + l_lug / 2

        outputs["data:weight:payload:PAX:CG:x"] = x_cg_pax
        outputs["data:weight:payload:rear_fret:CG:x"] = x_cg_r_fret
        outputs["data:weight:payload:front_fret:CG:x"] = x_cg_f_fret
