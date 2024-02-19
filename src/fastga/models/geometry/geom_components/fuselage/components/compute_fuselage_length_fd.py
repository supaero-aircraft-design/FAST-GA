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

from ..constants import SUBMODEL_FUSELAGE_LENGTH_FD


@oad.RegisterSubmodel(
    SUBMODEL_FUSELAGE_LENGTH_FD,
    "fastga.submodel.geometry.fuselage.dimensions.length.fd.legacy",
)
class ComputeFuselageLengthFD(om.ExplicitComponent):
    """
    Computes fuselage length.
    """

    def setup(self):

        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:length", val=np.nan, units="m")

        self.add_output("data:geometry:fuselage:length", val=10.0, units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        ht_lp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        vt_lp = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        ht_length = inputs["data:geometry:horizontal_tail:MAC:length"]
        vt_length = inputs["data:geometry:vertical_tail:MAC:length"]

        fus_length = fa_length + max(ht_lp + 0.75 * ht_length, vt_lp + 0.75 * vt_length)

        outputs["data:geometry:fuselage:length"] = fus_length

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        ht_lp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        vt_lp = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        ht_length = inputs["data:geometry:horizontal_tail:MAC:length"]
        vt_length = inputs["data:geometry:vertical_tail:MAC:length"]

        partials["data:geometry:fuselage:length", "data:geometry:wing:MAC:at25percent:x"] = 1.0

        if (ht_lp + 0.75 * ht_length) > (vt_lp + 0.75 * vt_length):

            partials[
                "data:geometry:fuselage:length",
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
            ] = 1.0
            partials[
                "data:geometry:fuselage:length", "data:geometry:horizontal_tail:MAC:length"
            ] = 0.75
            partials[
                "data:geometry:fuselage:length",
                "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            ] = 0.0
            partials[
                "data:geometry:fuselage:length", "data:geometry:vertical_tail:MAC:length"
            ] = 0.0

        elif (ht_lp + 0.75 * ht_length) < (vt_lp + 0.75 * vt_length):

            partials[
                "data:geometry:fuselage:length",
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
            ] = 0.0
            partials[
                "data:geometry:fuselage:length", "data:geometry:horizontal_tail:MAC:length"
            ] = 0.0
            partials[
                "data:geometry:fuselage:length",
                "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            ] = 1.0
            partials[
                "data:geometry:fuselage:length", "data:geometry:vertical_tail:MAC:length"
            ] = 0.75
