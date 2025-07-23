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

from .compute_cl_wing import ComputeWingLiftCoefficient
from .compute_compressibility_correction_wing import ComputeCompressibilityCorrectionWing
from ..digitization.compute_cn_p_wing_twist_contribution import ComputeWingTwistContributionCnp

from ...constants import SUBMODEL_CN_P_WING


@oad.RegisterSubmodel(
    SUBMODEL_CN_P_WING, "fastga.submodel.aerodynamics.wing.yaw_moment_roll_rate.legacy"
)
class ComputeCnRollRateWing(om.Group):
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

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO initialize
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"
        self.add_subsystem(
            name="cl_w_" + ls_tag,
            subsys=ComputeWingLiftCoefficient(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["data:*", "settings:*"],
        )
        self.add_subsystem(
            name="compressibility_correction_" + ls_tag,
            subsys=ComputeCompressibilityCorrectionWing(
                low_speed_aero=self.options["low_speed_aero"]
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            name="cn_p_wing_mach_0_" + ls_tag,
            subsys=ComputeCnRollRateWithZeroMach(),
            promotes=["data:*"],
        )
        self.add_subsystem(
            name="cn_p_wing_mach_" + ls_tag,
            subsys=ComputeCnRollRateWithMach(),
            promotes=["data:*"],
        )
        self.add_subsystem(
            name="twist_contribution_" + ls_tag,
            subsys=ComputeWingTwistContributionCnp(),
            promotes=["data:*"],
        )
        self.add_subsystem(
            name="cn_roll_rate_wing_" + ls_tag,
            subsys=ComputeCnpWing(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["data:*"],
        )

        self.connect(
            "cn_p_wing_mach_0_" + ls_tag + ".cn_p_wing_mach_0",
            "cn_p_wing_mach_" + ls_tag + ".cn_p_wing_mach_0",
        )
        self.connect(
            "compressibility_correction_" + ls_tag + ".mach_correction_wing",
            "cn_p_wing_mach_" + ls_tag + ".mach_correction",
        )
        self.connect("cl_w_" + ls_tag + ".CL_wing", "cn_roll_rate_wing_" + ls_tag + ".CL_wing")
        self.connect(
            "twist_contribution_" + ls_tag + ".twist_contribution_cn_p",
            "cn_roll_rate_wing_" + ls_tag + ".twist_contribution_cn_p",
        )
        self.connect(
            "cn_p_wing_mach_" + ls_tag + ".cn_p_wing_mach",
            "cn_roll_rate_wing_" + ls_tag + ".cn_p_wing_mach",
        )


class ComputeCnRollRateWithZeroMach(om.ExplicitComponent):
    """
    Class to compute the wing yaw moment coefficient from the roll rate with lift coefficient
    and Mach number both equal to zero. This calculation is based on
    :cite:`roskampart6:1985` section 10.2.6,  equation 10.65.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")

        self.add_output("cn_p_wing_mach_0", val=-0.1)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]

        outputs["cn_p_wing_mach_0"] = (
            -1.0
            / 6.0
            * (
                wing_ar
                + 6.0 * (wing_ar + np.cos(wing_sweep_25)) * (np.tan(wing_sweep_25) ** 2.0 / 12.0)
            )
            / (wing_ar + 4.0 * np.cos(wing_sweep_25))
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]

        partials["cn_p_wing_mach_0", "data:geometry:wing:sweep_25"] = -(
            3.0 * wing_ar * np.sin(wing_sweep_25) * np.tan(wing_sweep_25) ** 2.0
            + (
                8.0 * np.cos(wing_sweep_25) ** 2.0
                + 10.0 * wing_ar * np.cos(wing_sweep_25)
                + 2.0 * wing_ar**2.0
            )
            * np.cos(wing_sweep_25) ** -2.0
            * np.tan(wing_sweep_25)
            + 8.0 * wing_ar * np.sin(wing_sweep_25)
        ) / (12.0 * (4.0 * np.cos(wing_sweep_25) + wing_ar) ** 2.0)

        partials["cn_p_wing_mach_0", "data:geometry:wing:aspect_ratio"] = -(
            np.cos(wing_sweep_25) * (3.0 * np.tan(wing_sweep_25) ** 2.0 + 8.0)
        ) / (12.0 * (wing_ar + 4.0 * np.cos(wing_sweep_25)) ** 2.0)


class ComputeCnRollRateWithMach(om.ExplicitComponent):
    """
    Class to compute the wing yaw moment coefficient from the roll rate when the
    lift coefficient is zero with compressibility correction. This calculation is based on
    :cite:`roskampart6:1985` section 10.2.6,  equation 10.63.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("cn_p_wing_mach_0", val=np.nan)
        self.add_input("mach_correction", val=np.nan)

        self.add_output("cn_p_wing_mach", val=-0.1)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        cn_p_to_cl_mach_0 = inputs["cn_p_wing_mach_0"]
        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]
        b_coeff = inputs["mach_correction"]

        outputs["cn_p_wing_mach"] = (
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

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        cn_p_to_cl_mach_0 = inputs["cn_p_wing_mach_0"]
        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]
        b_coeff = inputs["mach_correction"]

        frac_1n = wing_ar + 4.0 * np.cos(wing_sweep_25)
        frac_1d = wing_ar * b_coeff + 4.0 * np.cos(wing_sweep_25)
        frac_2n = (
            wing_ar * b_coeff
            + 1.0 / 2.0 * (wing_ar * b_coeff + np.cos(wing_sweep_25)) * np.tan(wing_sweep_25) ** 2.0
        )
        frac_2d = (
            wing_ar + 1.0 / 2.0 * (wing_ar + np.cos(wing_sweep_25)) * np.tan(wing_sweep_25) ** 2.0
        )

        frac1 = frac_1n / frac_1d
        frac2 = frac_2n / frac_2d

        partials["cn_p_wing_mach", "cn_p_wing_mach_0"] = frac1 * frac2

        partials["cn_p_wing_mach", "data:geometry:wing:sweep_25"] = (
            cn_p_to_cl_mach_0
            * (
                (((-4.0 * np.sin(wing_sweep_25) * (frac_1d - frac_1n)) / frac_1d**2.0) * frac2)
                + frac1
                * (
                    0.5
                    * frac_2d
                    * (
                        2.0
                        * (wing_ar * b_coeff + np.cos(wing_sweep_25))
                        * np.tan(wing_sweep_25)
                        * np.cos(wing_sweep_25) ** -2.0
                        - np.sin(wing_sweep_25) * np.tan(wing_sweep_25) ** 2.0
                    )
                    - 0.5
                    * frac_2n
                    * (
                        2.0
                        * (wing_ar + np.cos(wing_sweep_25))
                        * np.tan(wing_sweep_25)
                        * np.cos(wing_sweep_25) ** -2.0
                        - np.sin(wing_sweep_25) * np.tan(wing_sweep_25) ** 2.0
                    )
                )
            )
            / frac_2d**2.0
        )

        partials["cn_p_wing_mach", "data:geometry:wing:aspect_ratio"] = cn_p_to_cl_mach_0 * (
            (frac2 * (frac_1d - frac_1n * b_coeff) / frac_1d**2.0)
            + frac1
            * (
                frac_2d * (b_coeff + 0.5 * b_coeff * np.tan(wing_sweep_25) ** 2.0)
                - frac_2n * (1.0 + 0.5 * np.tan(wing_sweep_25) ** 2.0)
            )
            / frac_2d**2.0
        )

        partials["cn_p_wing_mach", "mach_correction"] = cn_p_to_cl_mach_0 * (
            frac1 * (wing_ar + 0.5 * wing_ar * np.tan(wing_sweep_25) ** 2.0) / frac_2d
            - frac2 * frac_1n * wing_ar / frac_1d**2.0
        )


class ComputeCnpWing(om.ExplicitComponent):
    """
    Class to compute the contribution of the wing to the yaw moment coefficient due to roll rate.
    Based on :cite:`roskampart6:1985` section 10.2.6, equation 10.62.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO initialize
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input(
            "data:geometry:wing:twist",
            val=0.0,
            units="deg",
            desc="Negative twist means tip AOA is smaller than root",
        )
        self.add_input("cn_p_wing_mach", val=np.nan)
        self.add_input("twist_contribution_cn_p", val=np.nan)
        self.add_input("CL_wing", val=np.nan, units="unitless")

        self.add_output("data:aerodynamics:wing:" + ls_tag + ":Cn_p", units="rad**-1")

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        wing_twist = inputs["data:geometry:wing:twist"]  # In deg, not specified in the
        cn_p_to_cl_mach = inputs["cn_p_wing_mach"]
        cl_w = inputs["CL_wing"]
        twist_contribution = inputs["twist_contribution_cn_p"]

        outputs["data:aerodynamics:wing:" + ls_tag + ":Cn_p"] = (
            -cn_p_to_cl_mach * cl_w + twist_contribution * wing_twist
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        wing_twist = inputs["data:geometry:wing:twist"]  # In deg, not specified in the
        cn_p_to_cl_mach = inputs["cn_p_wing_mach"]
        cl_w = inputs["CL_wing"]
        twist_contribution = inputs["twist_contribution_cn_p"]

        partials[
            "data:aerodynamics:wing:" + ls_tag + ":Cn_p",
            "CL_wing",
        ] = -cn_p_to_cl_mach

        partials["data:aerodynamics:wing:" + ls_tag + ":Cn_p", "cn_p_wing_mach"] = -cl_w

        partials["data:aerodynamics:wing:" + ls_tag + ":Cn_p", "data:geometry:wing:twist"] = (
            twist_contribution
        )

        partials["data:aerodynamics:wing:" + ls_tag + ":Cn_p", "twist_contribution_cn_p"] = (
            wing_twist
        )
