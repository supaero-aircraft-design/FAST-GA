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

from fastga.models.aerodynamics.constants import SUBMODEL_CM_Q_HT


@oad.RegisterSubmodel(
    SUBMODEL_CM_Q_HT, "fastga.submodel.aerodynamics.horizontal_tail.cm_pitch_velocity.legacy"
)
class ComputeCMPitchVelocityHorizontalTail(om.ExplicitComponent):
    """
    Computation of the contribution of the horizontal tail to the increase in pitch moment due to
    a pitch velocity. This coefficient depends on the position of the CG, we will take it halfway
    between the aft and fwd CG. The convention from :cite:`roskampart6:1990` are used,
    meaning that, for the derivative with respect to a pitch rate, this rate is made
    dimensionless by multiplying it by the MAC and dividing it by 2 times the airspeed.

    Based on :cite:`roskampart6:1990` section 10.2.7. The formula uses the lift curve slope of
    the htp with respect to its own area, we will make the change since we have it with respect
    to the wing area.
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:volume_coefficient", val=np.nan)
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:aerodynamics:horizontal_tail:efficiency", val=np.nan)

        self.add_input("data:weight:aircraft:CG:fwd:x", units="m", val=np.nan)
        self.add_input("data:weight:aircraft:CG:aft:x", units="m", val=np.nan)

        if self.options["low_speed_aero"]:
            self.add_input(
                "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1"
            )

            self.add_output("data:aerodynamics:horizontal_tail:low_speed:Cm_q", units="rad**-1")

        else:
            self.add_input(
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan, units="rad**-1"
            )

            self.add_output("data:aerodynamics:horizontal_tail:cruise:Cm_q", units="rad**-1")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        l0_wing = inputs["data:geometry:wing:MAC:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        wing_area = inputs["data:geometry:wing:area"]
        ht_area = inputs["data:geometry:horizontal_tail:area"]
        eta_h = inputs["data:aerodynamics:horizontal_tail:efficiency"]
        volume_coeff_ht = inputs["data:geometry:horizontal_tail:volume_coefficient"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]

        # A CG position is necessary for the computation of this coefficient, we will thus assume
        # a CG between the two extrema
        x_cg_fwd = inputs["data:weight:aircraft:CG:fwd:x"]
        x_cg_aft = inputs["data:weight:aircraft:CG:aft:x"]
        x_cg_mid = (x_cg_fwd + x_cg_aft) / 2.0

        if self.options["low_speed_aero"]:
            cl_alpha_ht = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]

            outputs["data:aerodynamics:horizontal_tail:low_speed:Cm_q"] = (
                -2.0
                * cl_alpha_ht
                * wing_area
                / ht_area
                * eta_h
                * volume_coeff_ht
                * (lp_ht + fa_length - x_cg_mid)
                / l0_wing
            )
        else:
            cl_alpha_ht = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]

            outputs["data:aerodynamics:horizontal_tail:cruise:Cm_q"] = (
                -2.0
                * cl_alpha_ht
                * wing_area
                / ht_area
                * eta_h
                * volume_coeff_ht
                * (lp_ht + fa_length - x_cg_mid)
                / l0_wing
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        l0_wing = inputs["data:geometry:wing:MAC:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        wing_area = inputs["data:geometry:wing:area"]
        ht_area = inputs["data:geometry:horizontal_tail:area"]
        eta_h = inputs["data:aerodynamics:horizontal_tail:efficiency"]
        volume_coeff_ht = inputs["data:geometry:horizontal_tail:volume_coefficient"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]

        # A CG position is necessary for the computation of this coefficient, we will thus assume
        # a CG between the two extrema
        x_cg_fwd = inputs["data:weight:aircraft:CG:fwd:x"]
        x_cg_aft = inputs["data:weight:aircraft:CG:aft:x"]
        x_cg_mid = (x_cg_fwd + x_cg_aft) / 2.0

        if self.options["low_speed_aero"]:
            cl_alpha_ht = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]
            partials[
                "data:aerodynamics:horizontal_tail:low_speed:Cm_q",
                "data:aerodynamics:horizontal_tail:low_speed:CL_alpha",
            ] = (
                -2.0
                * wing_area
                / ht_area
                * eta_h
                * volume_coeff_ht
                * (lp_ht + fa_length - x_cg_mid)
                / l0_wing
            )
            partials[
                "data:aerodynamics:horizontal_tail:low_speed:Cm_q",
                "data:aerodynamics:horizontal_tail:efficiency",
            ] = (
                -2.0
                * cl_alpha_ht
                * wing_area
                / ht_area
                * volume_coeff_ht
                * (lp_ht + fa_length - x_cg_mid)
                / l0_wing
            )
            partials[
                "data:aerodynamics:horizontal_tail:low_speed:Cm_q",
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
            ] = (
                -2.0 * cl_alpha_ht * wing_area / ht_area * eta_h * volume_coeff_ht / l0_wing
            )
            partials[
                "data:aerodynamics:horizontal_tail:low_speed:Cm_q",
                "data:geometry:horizontal_tail:area",
            ] = (
                2.0
                * cl_alpha_ht
                * wing_area
                / ht_area ** 2.0
                * eta_h
                * volume_coeff_ht
                * (lp_ht + fa_length - x_cg_mid)
                / l0_wing
            )
            partials[
                "data:aerodynamics:horizontal_tail:low_speed:Cm_q", "data:geometry:wing:area"
            ] = (
                -2.0
                * cl_alpha_ht
                / ht_area
                * eta_h
                * volume_coeff_ht
                * (lp_ht + fa_length - x_cg_mid)
                / l0_wing
            )
            partials[
                "data:aerodynamics:horizontal_tail:low_speed:Cm_q",
                "data:geometry:wing:MAC:at25percent:x",
            ] = (
                -2.0 * cl_alpha_ht * wing_area / ht_area * eta_h * volume_coeff_ht / l0_wing
            )
            partials[
                "data:aerodynamics:horizontal_tail:low_speed:Cm_q",
                "data:geometry:horizontal_tail:volume_coefficient",
            ] = (
                -2.0
                * cl_alpha_ht
                * wing_area
                / ht_area
                * eta_h
                * (lp_ht + fa_length - x_cg_mid)
                / l0_wing
            )
            partials[
                "data:aerodynamics:horizontal_tail:low_speed:Cm_q",
                "data:weight:aircraft:CG:aft:x",
            ] = (
                cl_alpha_ht * wing_area / ht_area * eta_h * volume_coeff_ht / l0_wing
            )
            partials[
                "data:aerodynamics:horizontal_tail:low_speed:Cm_q",
                "data:weight:aircraft:CG:fwd:x",
            ] = (
                cl_alpha_ht * wing_area / ht_area * eta_h * volume_coeff_ht / l0_wing
            )
            partials[
                "data:aerodynamics:horizontal_tail:low_speed:Cm_q",
                "data:geometry:wing:MAC:length",
            ] = (
                2.0
                * cl_alpha_ht
                * wing_area
                / ht_area
                * eta_h
                * volume_coeff_ht
                * (lp_ht + fa_length - x_cg_mid)
                / l0_wing ** 2.0
            )
        else:
            cl_alpha_ht = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
            partials[
                "data:aerodynamics:horizontal_tail:cruise:Cm_q",
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha",
            ] = (
                -2.0
                * wing_area
                / ht_area
                * eta_h
                * volume_coeff_ht
                * (lp_ht + fa_length - x_cg_mid)
                / l0_wing
            )
            partials[
                "data:aerodynamics:horizontal_tail:cruise:Cm_q",
                "data:aerodynamics:horizontal_tail:efficiency",
            ] = (
                -2.0
                * cl_alpha_ht
                * wing_area
                / ht_area
                * volume_coeff_ht
                * (lp_ht + fa_length - x_cg_mid)
                / l0_wing
            )
            partials[
                "data:aerodynamics:horizontal_tail:cruise:Cm_q",
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
            ] = (
                -2.0 * cl_alpha_ht * wing_area / ht_area * eta_h * volume_coeff_ht / l0_wing
            )
            partials[
                "data:aerodynamics:horizontal_tail:cruise:Cm_q",
                "data:geometry:horizontal_tail:area",
            ] = (
                2.0
                * cl_alpha_ht
                * wing_area
                / ht_area ** 2.0
                * eta_h
                * volume_coeff_ht
                * (lp_ht + fa_length - x_cg_mid)
                / l0_wing
            )
            partials["data:aerodynamics:horizontal_tail:cruise:Cm_q", "data:geometry:wing:area"] = (
                -2.0
                * cl_alpha_ht
                / ht_area
                * eta_h
                * volume_coeff_ht
                * (lp_ht + fa_length - x_cg_mid)
                / l0_wing
            )
            partials[
                "data:aerodynamics:horizontal_tail:cruise:Cm_q",
                "data:geometry:wing:MAC:at25percent:x",
            ] = (
                -2.0 * cl_alpha_ht * wing_area / ht_area * eta_h * volume_coeff_ht / l0_wing
            )
            partials[
                "data:aerodynamics:horizontal_tail:cruise:Cm_q",
                "data:geometry:horizontal_tail:volume_coefficient",
            ] = (
                -2.0
                * cl_alpha_ht
                * wing_area
                / ht_area
                * eta_h
                * (lp_ht + fa_length - x_cg_mid)
                / l0_wing
            )
            partials[
                "data:aerodynamics:horizontal_tail:cruise:Cm_q",
                "data:weight:aircraft:CG:aft:x",
            ] = (
                cl_alpha_ht * wing_area / ht_area * eta_h * volume_coeff_ht / l0_wing
            )
            partials[
                "data:aerodynamics:horizontal_tail:cruise:Cm_q",
                "data:weight:aircraft:CG:fwd:x",
            ] = (
                cl_alpha_ht * wing_area / ht_area * eta_h * volume_coeff_ht / l0_wing
            )
            partials[
                "data:aerodynamics:horizontal_tail:cruise:Cm_q",
                "data:geometry:wing:MAC:length",
            ] = (
                2.0
                * cl_alpha_ht
                * wing_area
                / ht_area
                * eta_h
                * volume_coeff_ht
                * (lp_ht + fa_length - x_cg_mid)
                / l0_wing ** 2.0
            )
