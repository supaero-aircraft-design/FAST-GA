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

import logging
import numpy as np
import openmdao.api as om
import fastoad.api as oad

from ..figure_digitization import FigureDigitization
from ...constants import SUBMODEL_CL_R_WING

_LOGGER = logging.getLogger(__name__)


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
        b_coeff = np.sqrt(1.0 - mach**2.0 * np.cos(wing_sweep_25) ** 2.0)

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


class _Clw(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        aoa_ref_init = np.deg2rad(5.0) if self.options["low_speed_aero"] else np.deg2rad(1.0)

        self.add_input(
            "settings:aerodynamics:reference_flight_conditions:" + ls_tag + ":AOA",
            units="rad",
            val=aoa_ref_init,
        )
        self.add_input("data:aerodynamics:" + ls_tag + ":mach", val=np.nan)
        self.add_input("data:aerodynamics:wing:" + ls_tag + ":CL0_clean", val=np.nan)
        self.add_input(
            "data:aerodynamics:wing:" + ls_tag + ":CL_alpha", val=np.nan, units="rad**-1"
        )

        self.add_output("cl_w", val=0.001)

    def setup_partials(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.declare_partials("*", "*", method="exact")
        self.declare_partials("cl_w", "data:aerodynamics:wing:" + ls_tag + ":CL0_clean", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:" + ls_tag + ":AOA"]
        cl_0_wing = inputs["data:aerodynamics:wing:" + ls_tag + ":CL0_clean"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:" + ls_tag + ":CL_alpha"]

        outputs["cl_w"] = cl_0_wing + cl_alpha_wing * aoa_ref

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        partials["cl_w", "data:aerodynamics:wing:" + ls_tag + ":CL_alpha"] = inputs[
            "settings:aerodynamics:reference_flight_conditions:" + ls_tag + ":AOA"
        ]

        partials["cl_w", "settings:aerodynamics:reference_flight_conditions:" + ls_tag + ":AOA"] = (
            inputs["data:aerodynamics:wing:" + ls_tag + ":CL_alpha"]
        )


class _BCoeff(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input("data:aerodynamics:" + ls_tag + ":mach", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")

        self.add_output("b_coeff_mach", val=0.001)

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        outputs["b_coeff_mach"] = np.sqrt(
            1.0
            - inputs["data:aerodynamics:" + ls_tag + ":mach"] ** 2.0
            * np.cos(inputs["data:geometry:wing:sweep_25"]) ** 2.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]
        mach = inputs["data:aerodynamics:" + ls_tag + ":mach"]

        partials["b_coeff_mach", "data:geometry:wing:sweep_25"] = (
            mach**2.0
            * np.cos(wing_sweep_25)
            * np.sin(wing_sweep_25)
            / np.sqrt(1.0 - mach**2.0 * np.cos(wing_sweep_25) ** 2.0)
        )

        partials["b_coeff_mach", "data:aerodynamics:" + ls_tag + ":mach"] = (
            -mach
            * np.cos(wing_sweep_25) ** 2.0
            / np.sqrt(1.0 - mach**2.0 * np.cos(wing_sweep_25) ** 2.0)
        )


class _MachCorrection(om.ExplicitComponent):
    def setup(self):
        self.add_input("b_coeff_mach", val=np.nan)
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")

        self.add_output("mach_correction", val=0.99)

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        b_coeff = inputs["b_coeff_mach"]
        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]

        outputs["mach_correction"] = (
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

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        b_coeff = inputs["b_coeff_mach"]
        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]

        common_term = (
            wing_ar
            * (1.0 - b_coeff)
            / (2.0 * b_coeff * (wing_ar * b_coeff + 2.0 * np.cos(wing_sweep_25)))
        )
        common_denominator = (
            1.0
            + (
                (wing_ar * b_coeff + 2.0 * np.cos(wing_sweep_25))
                / (wing_ar * b_coeff + 4.0 * np.cos(wing_sweep_25))
            )
            * (np.tan(wing_sweep_25) ** 2.0)
            / 8.0
        )

        partials["mach_correction", "data:geometry:wing:aspect_ratio"] = (
            (
                (1.0 - b_coeff)
                * np.cos(wing_sweep_25)
                / (b_coeff * (wing_ar * b_coeff + 2.0 * np.cos(wing_sweep_25)) ** 2.0)
            )
            * common_denominator
            - (
                b_coeff
                * np.cos(wing_sweep_25)
                * np.tan(wing_sweep_25) ** 2
                / (4.0 * (wing_ar * b_coeff + 4.0 * np.cos(wing_sweep_25)) ** 2.0)
            )
            * common_term
        ) / common_denominator**2.0

        partials["mach_correction", "b_coeff_mach"] = (
            wing_ar
            * (wing_ar * b_coeff**2.0 - 2.0 * wing_ar * b_coeff - 2.0 * np.cos(wing_sweep_25))
            / (2.0 * b_coeff**2.0 * (wing_ar * b_coeff + 2.0 * np.cos(wing_sweep_25)) ** 2.0)
            * common_denominator
            - (
                wing_ar
                * np.cos(wing_sweep_25)
                * np.tan(wing_sweep_25) ** 2.0
                / (4.0 * (wing_ar * b_coeff + 4.0 * np.cos(wing_sweep_25)) ** 2.0)
            )
            * common_term
        ) / common_denominator**2.0


    # def mach_terms(A, b, s):
    #     c = np.cos(s)
    #     t = np.tan(s)
    #     T1 = A * (1.0 - b) / (2.0 * b * (A * b + 2.0 * c))
    #     T2 = ((A * b + 2.0 * c) / (A * b + 4.0 * c)) * (t ** 2) / 8.0
    #     N = 1.0 + T1 + T2
    #     D = 1.0 + T2
    #     return T1, T2, N, D, c, t
    #
    # def dmach_dwing_ar(A, b, s):
    #     T1, T2, N, D, c, t = mach_terms(A, b, s)
    #     # T1_A
    #     T1_A = (1.0 - b) * c / (b * (A * b + 2.0 * c) ** 2)
    #     # T2_A
    #     T2_A = b * c * t ** 2 / (4.0 * (A * b + 4.0 * c) ** 2)
    #     return (T1_A * D - T2_A * T1) / (D ** 2)
    #
    # def dmach_db_coeff(A, b, s):
    #     T1, T2, N, D, c, t = mach_terms(A, b, s)
    #     # T1_b
    #     T1_b = A * (A * b * b - 2.0 * A * b - 2.0 * c) / (2.0 * b * b * (A * b + 2.0 * c) ** 2)
    #     # T2_b
    #     T2_b = A * c * t ** 2 / (4.0 * (A * b + 4.0 * c) ** 2)
    #     return (T1_b * D - T2_b * T1) / (D ** 2)
    #
    # def dmach_dwing_sweep(A, b, s):
    #     T1, T2, N, D, c, t = mach_terms(A, b, s)
    #     sin_s = np.sin(s)
    #     cos_s = c
    #     tan_s = t
    #     # T1_s
    #     T1_s = -A * (b - 1.0) * sin_s / (b * (A * b + 2.0 * cos_s) ** 2)
    #     # T2_s (implemented from the compact expression above)
    #     # T2_s = ( (A^2 b^2 / cos^2 s - A b cos s + 7 A b / cos s + 8) * tan s )
    #     #        / (4 * (A b + 4 cos s)^2)
    #     T2_s = ((A * A * b * b / (cos_s ** 2)
    #              - A * b * cos_s
    #              + 7.0 * A * b / cos_s
    #              + 8.0)
    #             * tan_s) / (4.0 * (A * b + 4.0 * cos_s) ** 2)
    #     return (T1_s * D - T2_s * T1) / (D ** 2)


class _WingDihedralEffect(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")

        self.add_output("dihedral_effect", val=0.001)

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]

        outputs["dihedral_effect"] = (
            0.083
            * (np.pi * wing_ar * np.sin(wing_sweep_25))
            / (wing_ar + 4.0 * np.cos(wing_sweep_25))
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]

        partials["dihedral_effect", "data:geometry:wing:sweep_25"] = (
            0.083
            * np.pi
            * wing_ar
            * (wing_ar * np.cos(wing_sweep_25) + 4.0)
            / (4.0 * np.cos(wing_sweep_25) + wing_sweep_25) ** 2.0
        )

        partials["dihedral_effect", "data:geometry:wing:aspect_ratio"] = (
            0.332
            * np.pi
            * np.cos(wing_sweep_25)
            * np.sin(wing_sweep_25)
            / (4.0 * np.cos(wing_sweep_25) + wing_sweep_25) ** 2.0
        )


class _ClRollMomentFromTwist(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)

        self.add_output("cl_r_twist_effect", val=0.001)

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        unclipped_wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        unclipped_wing_taper_ratio = inputs["data:geometry:wing:taper_ratio"]

        wing_ar = np.clip(unclipped_wing_ar, 2.0, 10.0)
        wing_taper_ratio = np.clip(unclipped_wing_taper_ratio, 0.0, 1.0)

        if unclipped_wing_taper_ratio != wing_taper_ratio:
            _LOGGER.warning("Taper ratio is outside of the range in Roskam's book, value clipped")

        if unclipped_wing_ar != wing_ar:
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        outputs["cl_r_twist_effect"] = (
            -0.0001211945
            + 0.00643398 * wing_taper_ratio
            + 0.0036207415 * wing_ar
            - 0.0153251523 * wing_taper_ratio**2.0
            + 0.0024950401 * wing_ar * wing_taper_ratio
            - 0.0004995993 * wing_ar**2.0
            + 0.0098988825 * wing_taper_ratio**3.0
            - 0.0018342055 * wing_taper_ratio**2.0 * wing_ar
            - 1.401e-7 * wing_taper_ratio * wing_ar**2.0
            + 2.34128e-5 * wing_ar**3.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        unclipped_wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        unclipped_wing_taper_ratio = inputs["data:geometry:wing:taper_ratio"]

        wing_ar = np.clip(unclipped_wing_ar, 2.0, 10.0)
        wing_taper_ratio = np.clip(unclipped_wing_taper_ratio, 0.0, 1.0)

        partials["cl_r_twist_effect", "data:geometry:wing:aspect_ratio"] = np.where(
            wing_ar == unclipped_wing_ar,
            (
                0.0036207415
                + 0.0024950401 * wing_taper_ratio
                - 0.0009991986 * wing_ar
                - 0.0018342055 * wing_taper_ratio**2.0
                - 2.802e-7 * wing_taper_ratio * wing_ar
                + 7.02384e-5 * wing_ar**2.0
            ),
            1e-9,
        )

        partials["cl_r_twist_effect", "data:geometry:wing:taper_ratio"] = np.where(
            wing_taper_ratio == unclipped_wing_taper_ratio,
            (
                0.00643398
                - 0.0306503046 * wing_taper_ratio
                + 0.0024950401 * wing_ar
                + 0.0296966475 * wing_taper_ratio**2.0
                - 0.003668411 * wing_taper_ratio * wing_ar
                - 1.401e-7 * wing_ar**2.0
            ),
            1e-9,
        )
