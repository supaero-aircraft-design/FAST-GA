"""Estimation of electric power systems center of gravity."""

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

from ..constants import SUBMODEL_ELECTRIC_POWER_SYSTEMS_CG


@oad.RegisterSubmodel(
    SUBMODEL_ELECTRIC_POWER_SYSTEMS_CG,
    "fastga.submodel.weight.cg.system.power_system.electric.legacy",
)
class ComputeElectricPowerSystemCG(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Electric system gravity center estimation.

    Formula based on the fact that on a Cirrus SR22, one battery is in front of the
    firewall while the other one is behind the pressure bulkhead.
    """

    def setup(self):

        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")

        self.add_output("data:weight:systems:power:electric_systems:CG:x", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]

        x_cg_c12 = lav + 0.5 * (fus_length - (lav + lar))

        outputs["data:weight:systems:power:electric_systems:CG:x"] = x_cg_c12
