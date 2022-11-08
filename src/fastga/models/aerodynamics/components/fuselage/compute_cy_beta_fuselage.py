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

from fastga.models.aerodynamics.constants import SUBMODEL_CY_BETA_FUSELAGE


@oad.RegisterSubmodel(
    SUBMODEL_CY_BETA_FUSELAGE, "fastga.submodel.aerodynamics.fuselage.side_force_beta.legacy"
)
class ComputeCyBetaFuselage(om.ExplicitComponent):
    """
    Class to compute the contribution of the fuselage to the side force coefficient due to sideslip.
    Based on :cite:`roskampart6:1985` section 10.2.4.1
    """

    def setup(self):
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:root:z", val=np.nan, units="m")

        self.add_output("data:aerodynamics:fuselage:Cy_beta", units="rad**-1")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        ave_fuse_diameter = np.sqrt(b_f * h_f)

        wing_area = inputs["data:geometry:wing:area"]
        z2_wing = inputs["data:geometry:wing:root:z"]

        z2_ratio = 2.0 * z2_wing / ave_fuse_diameter

        if z2_ratio >= 0:
            k_i = 1 + 0.49 * z2_ratio
        else:
            k_i = 1 - 0.85 * z2_ratio

        # Station x0 is assumed to be in the cylindrical part of the fuselage
        s_0_fus = np.pi * (ave_fuse_diameter / 2) ** 2

        cy_beta_fus = -2.0 * k_i * s_0_fus / wing_area

        outputs["data:aerodynamics:fuselage:Cy_beta"] = cy_beta_fus

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]

        wing_area = inputs["data:geometry:wing:area"]
        z2_wing = inputs["data:geometry:wing:root:z"]

        if z2_wing >= 0:

            partials[
                "data:aerodynamics:fuselage:Cy_beta", "data:geometry:fuselage:maximum_width"
            ] = (-np.pi / (2.0 * wing_area) * (h_f + 0.49 * z2_wing / np.sqrt(b_f / h_f)))
            partials[
                "data:aerodynamics:fuselage:Cy_beta", "data:geometry:fuselage:maximum_height"
            ] = (-np.pi / (2.0 * wing_area) * (b_f + 0.49 * z2_wing / np.sqrt(h_f / b_f)))
            partials["data:aerodynamics:fuselage:Cy_beta", "data:geometry:wing:root:z"] = (
                -0.49 * np.pi * np.sqrt(h_f * b_f) / wing_area
            )
            partials["data:aerodynamics:fuselage:Cy_beta", "data:geometry:wing:area"] = (
                np.pi
                / (2.0 * wing_area ** 2.0)
                * (h_f * b_f + 2.0 * 0.49 * z2_wing * np.sqrt(h_f * b_f))
            )

        else:

            partials[
                "data:aerodynamics:fuselage:Cy_beta", "data:geometry:fuselage:maximum_width"
            ] = (-np.pi / (2.0 * wing_area) * (h_f - 0.85 * z2_wing / np.sqrt(b_f / h_f)))
            partials[
                "data:aerodynamics:fuselage:Cy_beta", "data:geometry:fuselage:maximum_height"
            ] = (-np.pi / (2.0 * wing_area) * (b_f + -0.85 * z2_wing / np.sqrt(h_f / b_f)))
            partials["data:aerodynamics:fuselage:Cy_beta", "data:geometry:wing:root:z"] = (
                +0.85 * np.pi * np.sqrt(h_f * b_f) / wing_area
            )
            partials["data:aerodynamics:fuselage:Cy_beta", "data:geometry:wing:area"] = (
                np.pi
                / (2.0 * wing_area ** 2.0)
                * (h_f * b_f - 2.0 * 0.85 * z2_wing * np.sqrt(h_f * b_f))
            )
