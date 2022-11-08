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

from ...constants import SUBMODEL_CL_P_VT


@oad.RegisterSubmodel(
    SUBMODEL_CL_P_VT, "fastga.submodel.aerodynamics.vertical_tail.roll_moment_roll_rate.legacy"
)
class ComputeClRollRateVerticalTail(om.ExplicitComponent):
    """
    Class to compute the contribution of the vertical tail to the roll moment coefficient due to
    roll rate (roll damping). The convention from :cite:`roskampart6:1985` are used, meaning that
    for lateral derivative, the reference length is the wing span. Another important point is
    that, for the derivative with respect to yaw and roll, the rotation speed are made
    dimensionless by multiplying them by the wing span and dividing them by 2 times the airspeed.

    Based on :cite:`roskampart6:1985` section 10.2.6
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:z", units="m", val=np.nan)
        self.add_input("data:geometry:vertical_tail:MAC:z", units="m", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_height", units="m", val=np.nan)

        if self.options["low_speed_aero"]:
            self.add_input(
                "data:aerodynamics:vertical_tail:low_speed:Cy_beta", val=np.nan, units="rad**-1"
            )

            self.add_output("data:aerodynamics:vertical_tail:low_speed:Cl_p", units="rad**-1")
        else:
            self.add_input(
                "data:aerodynamics:vertical_tail:cruise:Cy_beta", val=np.nan, units="rad**-1"
            )
            self.add_output("data:aerodynamics:vertical_tail:cruise:Cl_p", units="rad**-1")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        z_v = (
            inputs["data:geometry:wing:root:z"]
            + 0.5 * inputs["data:geometry:fuselage:maximum_height"]
            + inputs["data:geometry:vertical_tail:MAC:z"]
        )
        wing_span = inputs["data:geometry:wing:span"]

        if self.options["low_speed_aero"]:
            cy_beta_vt = inputs["data:aerodynamics:vertical_tail:low_speed:Cy_beta"]
            outputs["data:aerodynamics:vertical_tail:low_speed:Cl_p"] = (
                2.0 * (z_v / wing_span) ** 2.0 * cy_beta_vt
            )
        else:
            cy_beta_vt = inputs["data:aerodynamics:vertical_tail:cruise:Cy_beta"]
            outputs["data:aerodynamics:vertical_tail:cruise:Cl_p"] = (
                2.0 * (z_v / wing_span) ** 2.0 * cy_beta_vt
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        z_v = (
            inputs["data:geometry:wing:root:z"]
            + 0.5 * inputs["data:geometry:fuselage:maximum_height"]
            + inputs["data:geometry:vertical_tail:MAC:z"]
        )
        wing_span = inputs["data:geometry:wing:span"]

        d_z_v_d_z_w = 1.0
        d_z_v_d_h_f = 0.5
        d_z_v_d_z_mac = 1.0

        if self.options["low_speed_aero"]:
            cy_beta_vt = inputs["data:aerodynamics:vertical_tail:low_speed:Cy_beta"]

            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cl_p",
                "data:aerodynamics:vertical_tail:low_speed:Cy_beta",
            ] = (
                2.0 * (z_v / wing_span) ** 2.0
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cl_p",
                "data:geometry:wing:root:z",
            ] = (
                4.0 * (1.0 / wing_span) ** 2.0 * cy_beta_vt * z_v * d_z_v_d_z_w
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cl_p",
                "data:geometry:fuselage:maximum_height",
            ] = (
                4.0 * (1.0 / wing_span) ** 2.0 * cy_beta_vt * z_v * d_z_v_d_h_f
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cl_p",
                "data:geometry:vertical_tail:MAC:z",
            ] = (
                4.0 * (1.0 / wing_span) ** 2.0 * cy_beta_vt * z_v * d_z_v_d_z_mac
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cl_p",
                "data:geometry:wing:span",
            ] = (
                -4.0 * z_v ** 2.0 / wing_span ** 3.0 * cy_beta_vt
            )

        else:
            cy_beta_vt = inputs["data:aerodynamics:vertical_tail:cruise:Cy_beta"]

            partials[
                "data:aerodynamics:vertical_tail:cruise:Cl_p",
                "data:aerodynamics:vertical_tail:cruise:Cy_beta",
            ] = (
                2.0 * (z_v / wing_span) ** 2.0
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:Cl_p",
                "data:geometry:wing:root:z",
            ] = (
                4.0 * (1.0 / wing_span) ** 2.0 * cy_beta_vt * z_v * d_z_v_d_z_w
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:Cl_p",
                "data:geometry:fuselage:maximum_height",
            ] = (
                4.0 * (1.0 / wing_span) ** 2.0 * cy_beta_vt * z_v * d_z_v_d_h_f
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:Cl_p",
                "data:geometry:vertical_tail:MAC:z",
            ] = (
                4.0 * (1.0 / wing_span) ** 2.0 * cy_beta_vt * z_v * d_z_v_d_z_mac
            )
            partials["data:aerodynamics:vertical_tail:cruise:Cl_p", "data:geometry:wing:span",] = (
                -4.0 * z_v ** 2.0 / wing_span ** 3.0 * cy_beta_vt
            )
