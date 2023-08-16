"""
Estimation of hydraulic power system weight.
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

from .constants import SUBMODEL_HYDRAULIC_POWER_SYSTEM_MASS


@oad.RegisterSubmodel(
    SUBMODEL_HYDRAULIC_POWER_SYSTEM_MASS,
    "fastga.submodel.weight.mass.system.power_system.hydraulic.legacy",
)
class ComputeHydraulicWeight(om.ExplicitComponent):
    """
    Weight estimation for hydraulic power systems (generation and distribution)

    Based on a statistical analysis. See :cite:`roskampart5:1985` USAF method for the electric
    system weight and hydraulic weight.
    """

    def setup(self):

        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")

        self.add_output("data:weight:systems:power:hydraulic_systems:mass", units="lb")

        self.declare_partials(of="*", wrt="*", val=0.007)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mtow = inputs["data:weight:aircraft:MTOW"]

        c13 = 0.007 * mtow  # mass formula in lb

        outputs["data:weight:systems:power:hydraulic_systems:mass"] = c13
