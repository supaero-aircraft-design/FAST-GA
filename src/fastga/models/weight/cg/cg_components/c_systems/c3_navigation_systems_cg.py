"""Estimation of navigation systems center of gravity."""
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

from ..constants import SUBMODEL_NAVIGATION_SYSTEMS_CG


@oad.RegisterSubmodel(
    SUBMODEL_NAVIGATION_SYSTEMS_CG, "fastga.submodel.weight.cg.system.navigation_system.legacy"
)
class ComputeNavigationSystemsCG(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Navigation systems center of gravity estimation."""

    def setup(self):

        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")

        self.add_output("data:weight:systems:avionics:CG:x", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        lav = inputs["data:geometry:fuselage:front_length"]

        # Instruments length
        l_instr = 0.7
        x_cg_c3 = lav + l_instr / 2.0

        outputs["data:weight:systems:avionics:CG:x"] = x_cg_c3
