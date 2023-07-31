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

from ..constants import SUBMODEL_FUSELAGE_DIMENSIONS_BASIC


@oad.RegisterSubmodel(
    SUBMODEL_FUSELAGE_DIMENSIONS_BASIC, "fastga.submodel.geometry.fuselage.dimensions.basic.legacy"
)
class ComputeFuselageGeometryBasic(om.ExplicitComponent):
    """
    Geometry of fuselage - Cabin length defined with total fuselage length input (no sizing).
    """

    def setup(self):

        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")

        self.add_output("data:geometry:cabin:length", units="m")

        self.declare_partials(of="*", wrt="data:geometry:fuselage:length", val=1.0)
        self.declare_partials(
            of="*",
            wrt=["data:geometry:fuselage:front_length", "data:geometry:fuselage:rear_length"],
            val=-1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]

        # Cabin total length
        cabin_length = fus_length - (lav + lar)

        outputs["data:geometry:cabin:length"] = cabin_length
