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

from ...constants import SUBMODEL_CL_R_WING

_LOGGER = logging.getLogger(__name__)


@oad.RegisterSubmodel(
    SUBMODEL_CL_R_WING, "fastga.submodel.aerodynamics.wing.roll_moment_yaw_rate.legacy"
)
class ComputeClYawRateWing(om.Group):
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

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO initialize
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_subsystem(
            name="b_coeff_" + ls_tag,
            subsys=_BCoeff(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["data:*"],
        )
        self.add_subsystem(
            name="mach_correction_" + ls_tag, subsys=_MachCorrection(), promotes=["data:*"]
        )
        self.add_subsystem(
            name="dihedral_effect_" + ls_tag,
            subsys=_WingDihedralEffect(),
            promotes=["data:*"],
        )
        self.add_subsystem(
            name="twist_effect_" + ls_tag,
            subsys=_ClRollMomentFromTwist(),
            promotes=["data:*"],
        )
        self.add_subsystem(
            name="lift_effect_part_a_" + ls_tag,
            subsys=_ClrLiftEffectPartA(),
            promotes=["data:*"],
        )
        self.add_subsystem(
            name="lift_effect_part_b_" + ls_tag,
            subsys=_ClrLiftEffectPartB(),
            promotes=["data:*"],
        )
        self.add_subsystem(
            name="Cl_w_" + ls_tag,
            subsys=_Clw(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["data:*", "settings:*"],
        )
        self.add_subsystem(
            name="Cl_r_wing_" + ls_tag,
            subsys=_ClrWing(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["data:*"],
        )

        self.connect(
            "b_coeff_" + ls_tag + ".b_coeff_mach",
            "mach_correction_" + ls_tag + ".b_coeff_mach",
        )
        self.connect(
            "lift_effect_part_a_" + ls_tag + ".k_coefficient",
            "lift_effect_part_b_" + ls_tag + ".k_coefficient",
        )
        self.connect(
            "mach_correction_" + ls_tag + ".mach_correction",
            "Cl_r_wing_" + ls_tag + ".mach_correction",
        )
        self.connect(
            "dihedral_effect_" + ls_tag + ".dihedral_effect",
            "Cl_r_wing_" + ls_tag + ".dihedral_effect",
        )
        self.connect(
            "twist_effect_" + ls_tag + ".cl_r_twist_effect",
            "Cl_r_wing_" + ls_tag + ".cl_r_twist_effect",
        )
        self.connect(
            "lift_effect_part_b_" + ls_tag + ".lift_effect_mach_0",
            "Cl_r_wing_" + ls_tag + ".lift_effect_mach_0",
        )
        self.connect(
            "Cl_w_" + ls_tag + ".cl_w",
            "Cl_r_wing_" + ls_tag + ".cl_w",
        )


class _Clw(om.ExplicitComponent):

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO initialize
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
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

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.declare_partials("*", "*", method="exact")
        self.declare_partials("cl_w", "data:aerodynamics:wing:" + ls_tag + ":CL0_clean", val=1.0)

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:" + ls_tag + ":AOA"]
        cl_0_wing = inputs["data:aerodynamics:wing:" + ls_tag + ":CL0_clean"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:" + ls_tag + ":CL_alpha"]

        outputs["cl_w"] = cl_0_wing + cl_alpha_wing * aoa_ref

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        partials["cl_w", "data:aerodynamics:wing:" + ls_tag + ":CL_alpha"] = inputs[
            "settings:aerodynamics:reference_flight_conditions:" + ls_tag + ":AOA"
        ]

        partials["cl_w", "settings:aerodynamics:reference_flight_conditions:" + ls_tag + ":AOA"] = (
            inputs["data:aerodynamics:wing:" + ls_tag + ":CL_alpha"]
        )


class _BCoeff(om.ExplicitComponent):

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO initialize
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input("data:aerodynamics:" + ls_tag + ":mach", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")

        self.add_output("b_coeff_mach", val=0.001)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        outputs["b_coeff_mach"] = np.sqrt(
            1.0
            - inputs["data:aerodynamics:" + ls_tag + ":mach"] ** 2.0
            * np.cos(inputs["data:geometry:wing:sweep_25"]) ** 2.0
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
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

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("b_coeff_mach", val=np.nan)
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")

        self.add_output("mach_correction", val=0.99)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
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

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
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

        partials["mach_correction", "data:geometry:wing:sweep_25"] = (
            -wing_ar
            * (b_coeff - 1.0)
            * np.sin(wing_sweep_25)
            / (b_coeff * (wing_ar * b_coeff + 2.0 * np.cos(wing_sweep_25)) ** 2.0)
            * common_denominator
            - (
                wing_ar**2.0 * b_coeff * b_coeff / (np.cos(wing_sweep_25) ** 2.0)
                - wing_ar * b_coeff * np.cos(wing_sweep_25)
                + 7.0 * wing_ar * b_coeff / np.cos(wing_sweep_25)
                + 8.0
            )
            * np.tan(wing_sweep_25)
            / (4.0 * (wing_ar * b_coeff + 4.0 * np.cos(wing_sweep_25)) ** 2.0)
            * common_term
        ) / common_denominator**2.0


class _ClrLiftEffectPartA(om.ExplicitComponent):

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)

        self.add_output("k_coefficient", val=0.001)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        unclipped_wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        unclipped_wing_taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        wing_ar = np.clip(unclipped_wing_ar, 1.0, 10.0)
        wing_taper_ratio = np.clip(unclipped_wing_taper_ratio, 0.0, 1.0)

        if unclipped_wing_ar != wing_ar:
            _LOGGER.warning(
                "Aspect ratio value outside of the range in Roskam's book, value clipped"
            )

        if unclipped_wing_taper_ratio != wing_taper_ratio:
            _LOGGER.warning(
                "Taper ratio value outside of the range in Roskam's book, value clipped"
            )

        outputs["k_coefficient"] = (
            -1.4015195785
            + 0.4384930191 * wing_taper_ratio
            + 2.0066238293 * wing_ar
            + 5.8435265827 * wing_taper_ratio**2.0
            + 1.2333250199 * wing_ar * wing_taper_ratio
            - 0.4127145467 * wing_ar**2.0
            + 0.4543034387 * wing_taper_ratio**3.0
            - 1.0655109692 * wing_taper_ratio**2.0 * wing_ar
            - 0.0929885638 * wing_taper_ratio * wing_ar**2.0
            + 0.0411290779 * wing_ar**3.0
            - 4.2632431147 * wing_taper_ratio**4.0
            + 0.3686232582 * wing_taper_ratio**3.0 * wing_ar
            + 0.0332327109 * wing_taper_ratio**2.0 * wing_ar**2.0
            + 0.0024559125 * wing_taper_ratio * wing_ar**3.0
            - 0.0015529689 * wing_ar**4.0
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        unclipped_wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        unclipped_wing_taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        wing_ar = np.clip(unclipped_wing_ar, 0.0, 10.0)
        wing_taper_ratio = np.clip(unclipped_wing_taper_ratio, 0.0, 1.0)

        partials["k_coefficient", "data:geometry:wing:aspect_ratio"] = np.where(
            unclipped_wing_ar == wing_ar,
            (
                2.0066238293
                + 1.2333250199 * wing_taper_ratio
                - 0.8254290934 * wing_ar
                - 1.0655109692 * wing_taper_ratio**2.0
                - 0.1859771276 * wing_taper_ratio * wing_ar
                + 0.1233872337 * wing_ar**2.0
                + 0.3686232582 * wing_taper_ratio**3.0
                + 0.0664654218 * wing_taper_ratio**2.0 * wing_ar
                + 0.0073677375 * wing_taper_ratio * wing_ar**2.0
                - 0.0062118756 * wing_ar**3.0
            ),
            1e-9,
        )

        partials["k_coefficient", "data:geometry:wing:taper_ratio"] = np.where(
            unclipped_wing_taper_ratio == wing_taper_ratio,
            (
                0.4384930191
                + 11.6870531654 * wing_taper_ratio
                + 1.2333250199 * wing_ar
                + 1.3629103161 * wing_taper_ratio**2.0
                - 2.1310219384 * wing_taper_ratio * wing_ar
                - 0.0929885638 * wing_ar**2.0
                - 17.0529724588 * wing_taper_ratio**3.0
                + 1.1058697746 * wing_taper_ratio**2.0 * wing_ar
                + 0.0664654218 * wing_taper_ratio * wing_ar**2.0
                + 0.0024559125 * wing_ar**3.0
            ),
            1e-9,
        )


class _ClrLiftEffectPartB(om.ExplicitComponent):

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("k_coefficient", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")

        self.add_output("lift_effect_mach_0", val=0.001)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        unclipped_k_coeff = inputs["k_coefficient"]
        k_coeff = np.clip(unclipped_k_coeff, 0.0, 8.0)
        unclipped_wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]
        wing_sweep_25 = np.clip(unclipped_wing_sweep_25, 0.0, 60.0)

        if unclipped_k_coeff != k_coeff:
            _LOGGER.warning(
                "Intermediate value outside of the range in Roskam's book, value clipped"
            )

        if unclipped_wing_sweep_25 != wing_sweep_25:
            _LOGGER.warning(
                "Sweep angle value outside of the range in Roskam's book, value clipped"
            )

        outputs["lift_effect_mach_0"] = (
            0.110577
            + 0.023436 * k_coeff
            - 0.000270 * wing_sweep_25
            - 0.000104 * k_coeff**2.0
            + 0.000537 * k_coeff * wing_sweep_25
            + 0.000040 * wing_sweep_25**2.0
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        unclipped_k_coeff = inputs["k_coefficient"]
        k_coeff = np.clip(unclipped_k_coeff, 0.0, 10.0)
        unclipped_wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]
        wing_sweep_25 = np.clip(unclipped_wing_sweep_25, 0.0, 60.0)

        partials["lift_effect_mach_0", "k_coefficient"] = np.where(
            unclipped_k_coeff == k_coeff,
            (0.023436 + 0.000537 * wing_sweep_25 - 0.000208 * k_coeff),
            1e-9,
        )

        partials["lift_effect_mach_0", "data:geometry:wing:sweep_25"] = np.where(
            unclipped_wing_sweep_25 == wing_sweep_25,
            (-0.00027 + 0.000537 * k_coeff + 0.00008 * wing_sweep_25),
            1e-9,
        )


class _WingDihedralEffect(om.ExplicitComponent):

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")

        self.add_output("dihedral_effect", val=0.001)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]

        outputs["dihedral_effect"] = (
            0.083
            * (np.pi * wing_ar * np.sin(wing_sweep_25))
            / (wing_ar + 4.0 * np.cos(wing_sweep_25))
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]

        partials["dihedral_effect", "data:geometry:wing:sweep_25"] = (
            0.083
            * np.pi
            * wing_ar
            * (wing_ar * np.cos(wing_sweep_25) + 4.0)
            / (4.0 * np.cos(wing_sweep_25) + wing_ar) ** 2.0
        )

        partials["dihedral_effect", "data:geometry:wing:aspect_ratio"] = (
            0.332
            * np.pi
            * np.cos(wing_sweep_25)
            * np.sin(wing_sweep_25)
            / (4.0 * np.cos(wing_sweep_25) + wing_ar) ** 2.0
        )


class _ClRollMomentFromTwist(om.ExplicitComponent):

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)

        self.add_output("cl_r_twist_effect", val=0.001)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
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

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
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


class _ClrWing(om.ExplicitComponent):

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO initialize
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input("cl_w", val=np.nan)
        self.add_input("mach_correction", val=np.nan)
        self.add_input("lift_effect_mach_0", val=np.nan)
        self.add_input("dihedral_effect", val=np.nan)
        self.add_input("data:geometry:wing:dihedral", val=np.nan, units="deg")
        self.add_input("cl_r_twist_effect", val=np.nan)
        self.add_input(
            "data:geometry:wing:twist",
            val=0.0,
            units="deg",
            desc="Negative twist means tip AOA is smaller than root",
        )

        self.add_output("data:aerodynamics:wing:" + ls_tag + ":Cl_r", units="rad**-1")

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        cl_w = inputs["cl_w"]
        mach_correction = inputs["mach_correction"]
        lift_effect_mach_0 = inputs["lift_effect_mach_0"]
        dihedral_effect = inputs["dihedral_effect"]
        wing_dihedral = inputs["data:geometry:wing:dihedral"]
        twist_effect = inputs["cl_r_twist_effect"]
        wing_twist = inputs["data:geometry:wing:twist"]

        outputs["data:aerodynamics:wing:" + ls_tag + ":Cl_r"] = (
            cl_w * mach_correction * lift_effect_mach_0
            + dihedral_effect * wing_dihedral
            + twist_effect * wing_twist
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        cl_w = inputs["cl_w"]
        mach_correction = inputs["mach_correction"]
        lift_effect_mach_0 = inputs["lift_effect_mach_0"]
        dihedral_effect = inputs["dihedral_effect"]
        wing_dihedral = inputs["data:geometry:wing:dihedral"]
        twist_effect = inputs["cl_r_twist_effect"]
        wing_twist = inputs["data:geometry:wing:twist"]

        partials["data:aerodynamics:wing:" + ls_tag + ":Cl_r", "cl_w"] = (
            mach_correction * lift_effect_mach_0
        )

        partials["data:aerodynamics:wing:" + ls_tag + ":Cl_r", "mach_correction"] = (
            cl_w * lift_effect_mach_0
        )

        partials["data:aerodynamics:wing:" + ls_tag + ":Cl_r", "lift_effect_mach_0"] = (
            cl_w * mach_correction
        )

        partials["data:aerodynamics:wing:" + ls_tag + ":Cl_r", "dihedral_effect"] = wing_dihedral

        partials["data:aerodynamics:wing:" + ls_tag + ":Cl_r", "data:geometry:wing:dihedral"] = (
            dihedral_effect
        )

        partials["data:aerodynamics:wing:" + ls_tag + ":Cl_r", "cl_r_twist_effect"] = wing_twist

        partials["data:aerodynamics:wing:" + ls_tag + ":Cl_r", "data:geometry:wing:twist"] = (
            twist_effect
        )
