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

from ..constants import SUBMODEL_FUSELAGE_LENGTH_FL


@oad.RegisterSubmodel(
    SUBMODEL_FUSELAGE_LENGTH_FL,
    "fastga.submodel.geometry.fuselage.dimensions.length.fl.legacy",
)
class ComputeFuselageLengthFL(om.ExplicitComponent):
    """
    Computes fuselage length.
    """

    def setup(self):

        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

        self.add_output("data:geometry:fuselage:length", val=10.0, units="m")

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        lar = inputs["data:geometry:fuselage:rear_length"]
        cabin_length = inputs["data:geometry:cabin:length"]
        lav = inputs["data:geometry:fuselage:front_length"]

        fus_length = lav + cabin_length + lar

        outputs["data:geometry:fuselage:length"] = fus_length
