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
from ...constants import SUBMODEL_CL_R_WING


@oad.RegisterSubmodel(
    SUBMODEL_CL_R_WING, "fastga.submodel.aerodynamics.wing.roll_moment_yaw_rate.legacy"
)
class ComputeClYawRateWing(FigureDigitization):
    """
    Class to compute the contribution of the wing to the roll moment coefficient due to yaw rate.
    Depends on the lift coefficient of the wing, hence on the reference angle of attack,
    so the same remark as in ..compute_cy_yaw_rate.py holds. The convention from
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
        self.add_input("data:geometry:wing:dihedral", val=np.nan, units="deg")
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

            self.add_output("data:aerodynamics:wing:low_speed:Cl_r", units="rad**-1")

        else:
            self.add_input(
                "settings:aerodynamics:reference_flight_conditions:cruise:AOA",
                units="rad",
                val=1.0 * np.pi / 180.0,
            )
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")

            self.add_output("data:aerodynamics:wing:cruise:Cl_r", units="rad**-1")

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        wing_taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]  # In rad !!!
        wing_dihedral = inputs["data:geometry:wing:dihedral"]  # In deg
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

        # Fuselage contribution neglected
        cl_w = cl_0_wing + cl_alpha_wing * aoa_ref
        b_coeff = np.sqrt(1.0 - mach ** 2.0 * np.cos(wing_sweep_25) ** 2.0)

        lift_effect_mach_0 = self.cl_r_lifting_effect(wing_ar, wing_taper_ratio, wing_sweep_25)
        mach_correction = (
            1.0
            + (wing_ar * (1.0 - b_coeff))
            / (2.0 * b_coeff * (wing_ar * b_coeff + 2.0 * np.cos(wing_sweep_25)))
            + (wing_ar * b_coeff + 2.0 * np.cos(wing_sweep_25))
            / (wing_ar * b_coeff + 4.0 * np.cos(wing_sweep_25))
            * np.tan(wing_sweep_25) ** 2.0
            / 8.0
        ) / (
            1.0
            + (wing_ar * b_coeff + 2.0 * np.cos(wing_sweep_25))
            / (wing_ar * b_coeff + 4.0 * np.cos(wing_sweep_25))
            * np.tan(wing_sweep_25) ** 2.0
            / 8.0
        )

        dihedral_effect = (
            0.083
            * (np.pi * wing_ar * np.sin(wing_sweep_25))
            / (wing_ar + 4.0 * np.cos(wing_sweep_25))
        )

        twist_effect = self.cl_r_twist_effect(wing_taper_ratio, wing_ar)

        # Flap deflection effect neglected
        cl_r_w = (
            cl_w * mach_correction * lift_effect_mach_0
            + dihedral_effect * wing_dihedral
            + twist_effect * wing_twist
        )

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:wing:low_speed:Cl_r"] = cl_r_w
        else:
            outputs["data:aerodynamics:wing:cruise:Cl_r"] = cl_r_w
