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
from ...constants import SUBMODEL_CN_R_WING


@oad.RegisterSubmodel(
    SUBMODEL_CN_R_WING, "fastga.submodel.aerodynamics.wing.yaw_moment_yaw_rate.legacy"
)
class ComputeCnYawRateWing(FigureDigitization):
    """
    Class to compute the contribution of the wing to the yaw moment coefficient due to yaw rate (
    yaw damping). Depends on the lift coefficient of the wing, hence on the reference angle of
    attack, so the same remark as in ..compute_cy_yaw_rate.py holds. The convention from
    :cite:`roskampart6:1985` are used, meaning that for lateral derivative, the reference length
    is the wing span. Another important point is that, for the derivative with respect to yaw and
    roll, the rotation speed are made dimensionless by multiplying them by the wing span and
    dividing them by 2 times the airspeed.

    Based on :cite:`roskampart6:1985` section 10.2.8
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:handling_qualities:stick_fixed_static_margin", val=np.nan)

        if self.options["low_speed_aero"]:
            self.add_input(
                "settings:aerodynamics:reference_flight_conditions:low_speed:AOA",
                units="rad",
                val=5.0 * np.pi / 180.0,
            )
            self.add_input("data:aerodynamics:wing:low_speed:CD0", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan, units="rad**-1")

            self.add_output("data:aerodynamics:wing:low_speed:Cn_r", units="rad**-1")

        else:
            self.add_input(
                "settings:aerodynamics:reference_flight_conditions:cruise:AOA",
                units="rad",
                val=1.0 * np.pi / 180.0,
            )
            self.add_input("data:aerodynamics:wing:cruise:CD0", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")

            self.add_output("data:aerodynamics:wing:cruise:Cn_r", units="rad**-1")

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        wing_taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]  # In rad !!!
        static_margin = inputs["data:handling_qualities:stick_fixed_static_margin"]

        if self.options["low_speed_aero"]:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:low_speed:AOA"]
            cd_0_wing = inputs["data:aerodynamics:wing:low_speed:CD0"]
            cl_0_wing = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
            cl_alpha_wing = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
        else:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:cruise:AOA"]
            cd_0_wing = inputs["data:aerodynamics:wing:cruise:CD0"]
            cl_0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
            cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]

        # Fuselage contribution neglected
        cl_w = cl_0_wing + cl_alpha_wing * aoa_ref

        lift_effect = self.cn_r_lift_effect(static_margin, wing_sweep_25, wing_ar, wing_taper_ratio)
        drag_effect = self.cn_r_drag_effect(static_margin, wing_sweep_25, wing_ar)

        cn_r_w = lift_effect * cl_w ** 2.0 + drag_effect * cd_0_wing

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:wing:low_speed:Cn_r"] = cn_r_w
        else:
            outputs["data:aerodynamics:wing:cruise:Cn_r"] = cn_r_w
