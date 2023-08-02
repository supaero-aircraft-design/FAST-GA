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

from ..constants import SUBMODEL_FUSELAGE_NPAX


@oad.RegisterSubmodel(
    SUBMODEL_FUSELAGE_NPAX, "fastga.submodel.geometry.fuselage.dimensions.npax.legacy"
)
class ComputeFuselageNPAX(om.ExplicitComponent):
    """
    Computes number of pax cabin.
    """

    def setup(self):

        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)

        self.add_output("data:geometry:cabin:NPAX")

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        npax_max = inputs["data:geometry:cabin:seats:passenger:NPAX_max"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]

        # noinspection PyBroadException
        npax = np.ceil(float(npax_max) / float(seats_p_row)) * float(seats_p_row)

        outputs["data:geometry:cabin:NPAX"] = npax
