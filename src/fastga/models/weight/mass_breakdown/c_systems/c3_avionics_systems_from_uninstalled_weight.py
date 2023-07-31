"""Estimation of navigation systems weight."""
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

import openmdao.api as om
import fastoad.api as oad

from .constants import SUBMODEL_AVIONICS_SYSTEM_MASS


@oad.RegisterSubmodel(
    SUBMODEL_AVIONICS_SYSTEM_MASS,
    "fastga.submodel.weight.mass.system.avionics_systems.from_uninstalled",
)
class ComputeAvionicsSystemsWeightFromUninstalled(om.ExplicitComponent):
    """
    Weight estimation for avionics systems. Takes into account the weight of:
    - Instrumentation
    - Avionics
    - Navigation

    Based on a statistical analysis. See :cite:`gudmundsson:2013`.
    """

    def setup(self):

        self.add_input(
            "data:weight:systems:avionics:mass_uninstalled",
            val=45.0,
            units="lbm",
            desc="Weight of the uninstalled avionics system, default correspond to the value of "
            "the Garmin G1000",
        )

        self.add_output("data:weight:systems:avionics:mass", units="lbm")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        uninstalled_avionics = inputs["data:weight:systems:avionics:mass_uninstalled"]

        outputs["data:weight:systems:avionics:mass"] = 2.11 * uninstalled_avionics ** 0.933

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        uninstalled_avionics = inputs["data:weight:systems:avionics:mass_uninstalled"]

        partials[
            "data:weight:systems:avionics:mass", "data:weight:systems:avionics:mass_uninstalled"
        ] = 1.96863 / (uninstalled_avionics ** (0.067))
