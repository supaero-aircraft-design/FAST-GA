"""Estimation of horizontal tail volume coefficient."""
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

from ..constants import SUBMODEL_HT_VOLUME_COEFF


@oad.RegisterSubmodel(
    SUBMODEL_HT_VOLUME_COEFF, "fastga.submodel.geometry.horizontal_tail.volume_coefficient.legacy"
)
class ComputeHTVolumeCoefficient(om.ExplicitComponent):
    """
    Computation of the Volume coefficient for the horizontal tail. It is a result and not an
    input of the sizing of the HTP.
    """

    def setup(self):

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )

        self.add_output("data:geometry:horizontal_tail:volume_coefficient")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        l0_wing = inputs["data:geometry:wing:MAC:length"]
        wing_area = inputs["data:geometry:wing:area"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        ht_area = inputs["data:geometry:horizontal_tail:area"]

        outputs["data:geometry:horizontal_tail:volume_coefficient"] = (ht_area * lp_ht) / (
            wing_area * l0_wing
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        l0_wing = inputs["data:geometry:wing:MAC:length"]
        wing_area = inputs["data:geometry:wing:area"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        ht_area = inputs["data:geometry:horizontal_tail:area"]

        partials[
            "data:geometry:horizontal_tail:volume_coefficient", "data:geometry:wing:MAC:length"
        ] = -(ht_area * lp_ht) / (wing_area * l0_wing ** 2.0)
        partials["data:geometry:horizontal_tail:volume_coefficient", "data:geometry:wing:area"] = -(
            ht_area * lp_ht
        ) / (wing_area ** 2.0 * l0_wing)
        partials[
            "data:geometry:horizontal_tail:volume_coefficient",
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
        ] = ht_area / (wing_area * l0_wing)
        partials[
            "data:geometry:horizontal_tail:volume_coefficient", "data:geometry:horizontal_tail:area"
        ] = lp_ht / (wing_area * l0_wing)
