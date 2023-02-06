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

from fastga.models.aerodynamics.constants import SUBMODEL_CL_Q_HT


@oad.RegisterSubmodel(
    SUBMODEL_CL_Q_HT, "fastga.submodel.aerodynamics.horizontal_tail.cl_pitch_velocity.legacy"
)
class ComputeCLPitchVelocityHorizontalTail(om.ExplicitComponent):
    """
    Computation of the contribution of the horizontal tail to the increase in lift due to a pitch
    velocity. The convention from :cite:`roskampart6:1985` are used, meaning that,
    for the derivative with respect to yaw and roll, the rotation speed are made dimensionless by
    multiplying them by the wing span and dividing them by 2 times the airspeed.


    Based on :cite:`roskampart6:1985` section 10.2.7
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:wing:area", units="m**2", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:area", units="m**2", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:volume_coefficient", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:efficiency", val=np.nan)

        if self.options["low_speed_aero"]:
            self.add_input(
                "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1"
            )

            self.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_q", units="rad**-1")

            self.declare_partials(
                of="data:aerodynamics:horizontal_tail:low_speed:CL_q",
                wrt=[
                    "data:geometry:wing:area",
                    "data:geometry:horizontal_tail:area",
                    "data:geometry:horizontal_tail:volume_coefficient",
                    "data:aerodynamics:horizontal_tail:low_speed:CL_alpha",
                    "data:aerodynamics:horizontal_tail:efficiency",
                ],
                method="exact",
            )
        else:
            self.add_input(
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan, units="rad**-1"
            )

            self.add_output("data:aerodynamics:horizontal_tail:cruise:CL_q", units="rad**-1")

            self.declare_partials(
                of="data:aerodynamics:horizontal_tail:cruise:CL_q",
                wrt=[
                    "data:geometry:wing:area",
                    "data:geometry:horizontal_tail:area",
                    "data:geometry:horizontal_tail:volume_coefficient",
                    "data:aerodynamics:horizontal_tail:cruise:CL_alpha",
                    "data:aerodynamics:horizontal_tail:efficiency",
                ],
                method="exact",
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        eta_h = inputs["data:aerodynamics:horizontal_tail:efficiency"]
        volume_coeff_ht = inputs["data:geometry:horizontal_tail:volume_coefficient"]

        # From the instructions section 10.2.7, it seems to suggest that we need the lift curve
        # coefficient with respect to the area of the horizontal tail hence the change in
        # reference surface.
        wing_area = inputs["data:geometry:wing:area"]
        ht_area = inputs["data:geometry:horizontal_tail:area"]

        if self.options["low_speed_aero"]:
            cl_alpha_ht = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]

            outputs["data:aerodynamics:horizontal_tail:low_speed:CL_q"] = (
                2.0 * cl_alpha_ht * eta_h * volume_coeff_ht * wing_area / ht_area
            )
        else:
            cl_alpha_ht = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]

            outputs["data:aerodynamics:horizontal_tail:cruise:CL_q"] = (
                2.0 * cl_alpha_ht * eta_h * volume_coeff_ht * wing_area / ht_area
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        eta_h = inputs["data:aerodynamics:horizontal_tail:efficiency"]
        volume_coeff_ht = inputs["data:geometry:horizontal_tail:volume_coefficient"]

        wing_area = inputs["data:geometry:wing:area"]
        ht_area = inputs["data:geometry:horizontal_tail:area"]

        if self.options["low_speed_aero"]:
            cl_alpha_ht = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]

            partials[
                "data:aerodynamics:horizontal_tail:low_speed:CL_q",
                "data:aerodynamics:horizontal_tail:efficiency",
            ] = (
                2.0 * cl_alpha_ht * volume_coeff_ht * wing_area / ht_area
            )
            partials[
                "data:aerodynamics:horizontal_tail:low_speed:CL_q",
                "data:geometry:horizontal_tail:volume_coefficient",
            ] = (
                2.0 * cl_alpha_ht * eta_h * wing_area / ht_area
            )
            partials[
                "data:aerodynamics:horizontal_tail:low_speed:CL_q",
                "data:aerodynamics:horizontal_tail:low_speed:CL_alpha",
            ] = (
                2.0 * volume_coeff_ht * eta_h * wing_area / ht_area
            )
            partials[
                "data:aerodynamics:horizontal_tail:low_speed:CL_q",
                "data:geometry:wing:area",
            ] = (
                2.0 * volume_coeff_ht * eta_h * cl_alpha_ht / ht_area
            )
            partials[
                "data:aerodynamics:horizontal_tail:low_speed:CL_q",
                "data:geometry:horizontal_tail:area",
            ] = (
                -2.0 * volume_coeff_ht * eta_h * cl_alpha_ht * wing_area / ht_area ** 2.0
            )
        else:
            cl_alpha_ht = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]

            partials[
                "data:aerodynamics:horizontal_tail:cruise:CL_q",
                "data:aerodynamics:horizontal_tail:efficiency",
            ] = (
                2.0 * cl_alpha_ht * volume_coeff_ht * wing_area / ht_area
            )
            partials[
                "data:aerodynamics:horizontal_tail:cruise:CL_q",
                "data:geometry:horizontal_tail:volume_coefficient",
            ] = (
                2.0 * cl_alpha_ht * eta_h * wing_area / ht_area
            )
            partials[
                "data:aerodynamics:horizontal_tail:cruise:CL_q",
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha",
            ] = (
                2.0 * volume_coeff_ht * eta_h * wing_area / ht_area
            )
            partials[
                "data:aerodynamics:horizontal_tail:cruise:CL_q",
                "data:geometry:wing:area",
            ] = (
                2.0 * volume_coeff_ht * eta_h * cl_alpha_ht / ht_area
            )
            partials[
                "data:aerodynamics:horizontal_tail:cruise:CL_q",
                "data:geometry:horizontal_tail:area",
            ] = (
                -2.0 * volume_coeff_ht * eta_h * cl_alpha_ht * wing_area / ht_area ** 2.0
            )
