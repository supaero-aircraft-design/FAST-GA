"""
Estimation of geometry of fuselage part A - Cabin (Commercial). 
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

from ..constants import SUBMODEL_FUSELAGE_CABIN_LENGTH


@oad.RegisterSubmodel(
    SUBMODEL_FUSELAGE_CABIN_LENGTH,
    "fastga.submodel.geometry.fuselage.dimensions.cabin_length.legacy",
)
class ComputeFuselageCabinLength(om.ExplicitComponent):
    """
    Computes cabin total length.
    """

    def setup(self):

        self.add_input("data:geometry:fuselage:PAX_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:luggage_length", val=np.nan, units="m")

        self.add_output("data:geometry:cabin:length", units="m")

        self.declare_partials("*", "data:geometry:fuselage:PAX_length", val=1.0)
        self.declare_partials("*", "data:geometry:fuselage:luggage_length", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        l_pax = inputs["data:geometry:fuselage:PAX_length"]
        l_lug = inputs["data:geometry:fuselage:luggage_length"]

        # Length of instrument panel
        l_instr = 0.7

        cabin_length = l_instr + l_pax + l_lug

        outputs["data:geometry:cabin:length"] = cabin_length
