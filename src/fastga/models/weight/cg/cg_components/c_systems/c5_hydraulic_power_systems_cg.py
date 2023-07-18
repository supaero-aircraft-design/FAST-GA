"""Estimation of hydraulic power systems center of gravity."""

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

from ..constants import SUBMODEL_HYDRAULIC_POWER_SYSTEMS_CG


@oad.RegisterSubmodel(
    SUBMODEL_HYDRAULIC_POWER_SYSTEMS_CG,
    "fastga.submodel.weight.cg.system.power_system.hydraulic.legacy",
)
class ComputeHydraulicPowerSystemCG(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Hydraulic system gravity center estimation.
    """

    def setup(self):

        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")

        self.add_output("data:weight:systems:power:hydraulic_systems:CG:x", units="m")

        self.declare_partials(
            "data:weight:systems:power:hydraulic_systems:CG:x",
            "data:geometry:fuselage:length",
            val=0.5,
        )
        self.declare_partials(
            "data:weight:systems:power:hydraulic_systems:CG:x",
            "data:geometry:fuselage:front_length",
            val=0.5,
        )
        self.declare_partials(
            "data:weight:systems:power:hydraulic_systems:CG:x",
            "data:geometry:fuselage:rear_length",
            val=-0.5,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]

        x_cg_c13 = lav + 0.5 * (fus_length - (lav + lar))

        outputs["data:weight:systems:power:hydraulic_systems:CG:x"] = x_cg_c13
