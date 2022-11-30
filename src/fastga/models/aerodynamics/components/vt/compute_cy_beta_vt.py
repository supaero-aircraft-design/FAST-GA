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

from fastga.models.aerodynamics.constants import SUBMODEL_CY_BETA_VT


@oad.RegisterSubmodel(
    SUBMODEL_CY_BETA_VT, "fastga.submodel.aerodynamics.vertical_tail.side_force_beta.legacy"
)
class ComputeCyBetaVerticalTail(om.ExplicitComponent):
    """
    Class to compute the contribution of the vertical tail to the side force coefficient due to
    sideslip.
    Based on :cite:`roskampart6:1985` section 10.2.4.1
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:root:z", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:average_depth", val=np.nan, units="m")

        self.add_input("data:aerodynamics:vertical_tail:efficiency", val=0.95)

        if self.options["low_speed_aero"]:
            self.add_input(
                "data:aerodynamics:vertical_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1"
            )
            self.add_output("data:aerodynamics:vertical_tail:low_speed:Cy_beta", units="rad**-1")
        else:
            self.add_input(
                "data:aerodynamics:vertical_tail:cruise:CL_alpha", val=np.nan, units="rad**-1"
            )
            self.add_output("data:aerodynamics:vertical_tail:cruise:Cy_beta", units="rad**-1")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        wing_area = inputs["data:geometry:wing:area"]
        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]
        z_w = inputs["data:geometry:wing:root:z"]
        z_f = inputs["data:geometry:fuselage:maximum_height"]
        vt_area = inputs["data:geometry:vertical_tail:area"]
        vt_span = inputs["data:geometry:vertical_tail:span"]
        avg_fus_depth = inputs["data:geometry:fuselage:average_depth"]

        eta_v = inputs["data:aerodynamics:vertical_tail:efficiency"]

        if self.options["low_speed_aero"]:
            cl_alpha_vt = inputs["data:aerodynamics:vertical_tail:low_speed:CL_alpha"]
        else:
            cl_alpha_vt = inputs["data:aerodynamics:vertical_tail:cruise:CL_alpha"]

        if vt_span / avg_fus_depth < 2.0:
            k_v = 0.75
        elif vt_span / avg_fus_depth < 3.5:
            k_v = 0.418 + 0.166 * vt_span / avg_fus_depth
        else:
            k_v = 1.0

        k_sigma = (
            0.724
            + 0.4 * z_w / z_f
            + 0.009 * wing_ar
            + 3.06 / (1.0 + np.cos(wing_sweep_25)) * vt_area / wing_area
        )

        cy_beta_vt = -k_v * cl_alpha_vt * k_sigma * eta_v * vt_area / wing_area

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:vertical_tail:low_speed:Cy_beta"] = cy_beta_vt
        else:
            outputs["data:aerodynamics:vertical_tail:cruise:Cy_beta"] = cy_beta_vt

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        wing_area = inputs["data:geometry:wing:area"]
        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]
        z_w = inputs["data:geometry:wing:root:z"]
        z_f = inputs["data:geometry:fuselage:maximum_height"]
        vt_area = inputs["data:geometry:vertical_tail:area"]
        vt_span = inputs["data:geometry:vertical_tail:span"]
        avg_fus_depth = inputs["data:geometry:fuselage:average_depth"]

        eta_v = inputs["data:aerodynamics:vertical_tail:efficiency"]

        if self.options["low_speed_aero"]:
            cl_alpha_vt = inputs["data:aerodynamics:vertical_tail:low_speed:CL_alpha"]
        else:
            cl_alpha_vt = inputs["data:aerodynamics:vertical_tail:cruise:CL_alpha"]

        if vt_span / avg_fus_depth < 2.0:
            k_v = 0.75
            d_k_v_d_span = 0.0
            d_k_v_d_hf = 0.0
        elif vt_span / avg_fus_depth < 3.5:
            k_v = 0.418 + 0.166 * vt_span / avg_fus_depth
            d_k_v_d_span = 0.166
            d_k_v_d_hf = -0.166 * vt_span / avg_fus_depth ** 2.0
        else:
            k_v = 1.0
            d_k_v_d_span = 0.0
            d_k_v_d_hf = 0.0

        k_sigma = (
            0.724
            + 0.4 * z_w / z_f
            + 0.009 * wing_ar
            + 3.06 / (1.0 + np.cos(wing_sweep_25)) * vt_area / wing_area
        )

        d_k_sigma_d_z_w = 0.4 / z_f
        d_k_sigma_d_z_f = -0.4 * z_w / z_f ** 2.0
        d_k_sigma_d_wing_ar = 0.009
        d_k_sigma_d_sweep_25 = (
            3.06
            * np.sin(wing_sweep_25)
            / (1.0 + np.cos(wing_sweep_25)) ** 2.0
            * vt_area
            / wing_area
        )
        d_k_sigma_d_wing_area = -3.06 / (1.0 + np.cos(wing_sweep_25)) * vt_area / wing_area ** 2.0
        d_k_sigma_d_vt_area = 3.06 / (1.0 + np.cos(wing_sweep_25)) / wing_area

        if self.options["low_speed_aero"]:
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cy_beta",
                "data:geometry:wing:aspect_ratio",
            ] = (
                -k_v * cl_alpha_vt * d_k_sigma_d_wing_ar * eta_v * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cy_beta",
                "data:geometry:wing:area",
            ] = (
                -k_v
                * cl_alpha_vt
                * eta_v
                * (
                    d_k_sigma_d_wing_area * vt_area / wing_area
                    - k_sigma * vt_area / wing_area ** 2.0
                )
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cy_beta",
                "data:geometry:wing:sweep_25",
            ] = (
                -k_v * cl_alpha_vt * d_k_sigma_d_sweep_25 * eta_v * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cy_beta", "data:geometry:wing:root:z"
            ] = (-k_v * cl_alpha_vt * d_k_sigma_d_z_w * eta_v * vt_area / wing_area)
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cy_beta",
                "data:geometry:fuselage:maximum_height",
            ] = (
                -k_v * cl_alpha_vt * d_k_sigma_d_z_f * eta_v * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cy_beta",
                "data:aerodynamics:vertical_tail:efficiency",
            ] = (
                -k_v * cl_alpha_vt * k_sigma * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cy_beta",
                "data:aerodynamics:vertical_tail:low_speed:CL_alpha",
            ] = (
                -k_v * k_sigma * eta_v * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cy_beta",
                "data:geometry:vertical_tail:area",
            ] = (
                -k_v
                * cl_alpha_vt
                * eta_v
                * (d_k_sigma_d_vt_area * vt_area / wing_area + k_sigma / wing_area)
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cy_beta",
                "data:geometry:vertical_tail:span",
            ] = (
                cl_alpha_vt * eta_v * k_sigma * vt_area / wing_area * d_k_v_d_span
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cy_beta",
                "data:geometry:fuselage:average_depth",
            ] = (
                cl_alpha_vt * eta_v * k_sigma * vt_area / wing_area * d_k_v_d_hf
            )

        else:
            partials[
                "data:aerodynamics:vertical_tail:cruise:Cy_beta",
                "data:geometry:wing:aspect_ratio",
            ] = (
                -k_v * cl_alpha_vt * d_k_sigma_d_wing_ar * eta_v * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:Cy_beta",
                "data:geometry:wing:area",
            ] = (
                -k_v
                * cl_alpha_vt
                * eta_v
                * (
                    d_k_sigma_d_wing_area * vt_area / wing_area
                    - k_sigma * vt_area / wing_area ** 2.0
                )
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:Cy_beta",
                "data:geometry:wing:sweep_25",
            ] = (
                -k_v * cl_alpha_vt * d_k_sigma_d_sweep_25 * eta_v * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:Cy_beta", "data:geometry:wing:root:z"
            ] = (-k_v * cl_alpha_vt * d_k_sigma_d_z_w * eta_v * vt_area / wing_area)
            partials[
                "data:aerodynamics:vertical_tail:cruise:Cy_beta",
                "data:geometry:fuselage:maximum_height",
            ] = (
                -k_v * cl_alpha_vt * d_k_sigma_d_z_f * eta_v * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:Cy_beta",
                "data:aerodynamics:vertical_tail:efficiency",
            ] = (
                -k_v * cl_alpha_vt * k_sigma * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:Cy_beta",
                "data:aerodynamics:vertical_tail:cruise:CL_alpha",
            ] = (
                -k_v * k_sigma * eta_v * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:Cy_beta",
                "data:geometry:vertical_tail:area",
            ] = (
                -k_v
                * cl_alpha_vt
                * eta_v
                * (d_k_sigma_d_vt_area * vt_area / wing_area + k_sigma / wing_area)
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:Cy_beta",
                "data:geometry:vertical_tail:span",
            ] = (
                cl_alpha_vt * eta_v * k_sigma * vt_area / wing_area * d_k_v_d_span
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:Cy_beta",
                "data:geometry:fuselage:average_depth",
            ] = (
                cl_alpha_vt * eta_v * k_sigma * vt_area / wing_area * d_k_v_d_hf
            )
