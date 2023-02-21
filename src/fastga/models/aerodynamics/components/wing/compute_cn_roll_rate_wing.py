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

import fastoad.api as oad

from ..figure_digitization import FigureDigitization
from ...constants import SUBMODEL_CN_P_WING


@oad.RegisterSubmodel(
    SUBMODEL_CN_P_WING, "fastga.submodel.aerodynamics.wing.yaw_moment_roll_rate.legacy"
)
class ComputeCnRollRateWing(FigureDigitization):
    """
    Class to compute the contribution of the wing to the yaw moment coefficient due to roll rate.
    Depends on the lift coefficient of the wing, hence on the reference angle of attack,
    so the same remark as in ..compute_cy_yaw_rate.py holds. Flap deflection effect is neglected.
    The convention from :cite:`roskampart6:1985` are used, meaning that for lateral derivative,
    the reference length is the wing span. Another important point is that, for the derivative
    with respect to yaw and roll, the rotation speed are made dimensionless by multiplying them
    by the wing span and dividing them by 2 times the airspeed.

    Based on :cite:`roskampart6:1985` section 10.2.6. The reference point for the CG was taken to
    be equal to the wing quarter chord to match what is taken for other coefficient. The change
    in reference point is not easy for this coefficient as it only affect part of the coefficient
    (the wing lift contribution), this coefficient might thus need to be recomputed "on the fly"
    for future stability computation. This is has no influence for unswept wing as in any case it
    was multiplied by tan(sweep_25).
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input(
            "data:geometry:wing:twist",
            val=0.0,
            units="deg",
            desc="Negative twist means tip AOA is smaller than root",
        )

        if self.options["low_speed_aero"]:
            self.add_input(
                "settings:aerodynamics:reference_flight_conditions:low_speed:AOA",
                units="rad",
                val=5.0 * np.pi / 180.0,
            )
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan, units="rad**-1")

            self.add_output("data:aerodynamics:wing:low_speed:Cn_p", units="rad**-1")

        else:
            self.add_input(
                "settings:aerodynamics:reference_flight_conditions:cruise:AOA",
                units="rad",
                val=1.0 * np.pi / 180.0,
            )
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")

            self.add_output("data:aerodynamics:wing:cruise:Cn_p", units="rad**-1")

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        wing_taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]  # In rad !!!
        wing_twist = inputs["data:geometry:wing:twist"]  # In deg, not specified in the
        # formula

        if self.options["low_speed_aero"]:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:low_speed:AOA"]
            mach = inputs["data:aerodynamics:low_speed:mach"]
            cl_0_wing = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
            cl_alpha_wing = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
        else:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:cruise:AOA"]
            mach = inputs["data:aerodynamics:cruise:mach"]
            cl_0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
            cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]

        cl_w = cl_0_wing + cl_alpha_wing * aoa_ref

        cn_p_to_cl_mach_0 = (
            -1.0
            / 6.0
            * (
                wing_ar
                + 6.0 * (wing_ar + np.cos(wing_sweep_25)) * (np.tan(wing_sweep_25) ** 2.0 / 12.0)
            )
            / (wing_ar + 4.0 * np.cos(wing_sweep_25))
        )

        b_coeff = np.sqrt(1.0 - mach ** 2.0 * np.cos(wing_sweep_25) ** 2.0)

        cn_p_to_cl_mach = (
            (wing_ar + 4.0 * np.cos(wing_sweep_25))
            / (wing_ar * b_coeff + 4.0 * np.cos(wing_sweep_25))
            * (
                wing_ar * b_coeff
                + 1.0
                / 2.0
                * (wing_ar * b_coeff + np.cos(wing_sweep_25))
                * np.tan(wing_sweep_25) ** 2.0
            )
            / (
                wing_ar
                + 1.0 / 2.0 * (wing_ar + np.cos(wing_sweep_25)) * np.tan(wing_sweep_25) ** 2.0
            )
        ) * cn_p_to_cl_mach_0

        twist_contribution = self.cn_p_twist_contribution(wing_taper_ratio, wing_ar)

        # Flap contribution neglected
        cn_p_w = -cn_p_to_cl_mach * cl_w + twist_contribution * wing_twist

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:wing:low_speed:Cn_p"] = cn_p_w
        else:
            outputs["data:aerodynamics:wing:cruise:Cn_p"] = cn_p_w
