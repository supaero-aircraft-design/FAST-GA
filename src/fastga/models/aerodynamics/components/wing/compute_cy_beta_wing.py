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

from fastga.models.aerodynamics.constants import SUBMODEL_CY_BETA_WING


@oad.RegisterSubmodel(
    SUBMODEL_CY_BETA_WING, "fastga.submodel.aerodynamics.wing.side_force_beta.legacy"
)
class ComputeCyBetaWing(om.ExplicitComponent):
    """
    Class to compute the contribution of the wing to the side force coefficient due to sideslip.
    Based on :cite:`roskampart6:1985` section 10.2.4.1
    """

    def setup(self):

        self.add_input("data:geometry:wing:dihedral", val=np.nan, units="deg")

        self.add_output("data:aerodynamics:wing:Cy_beta", units="rad**-1")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["data:aerodynamics:wing:Cy_beta"] = -0.00573 * inputs["data:geometry:wing:dihedral"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["data:aerodynamics:wing:Cy_beta", "data:geometry:wing:dihedral"] = -0.00573
