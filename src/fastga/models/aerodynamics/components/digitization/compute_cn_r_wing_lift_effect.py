"""
Python module for wing lift effect on Cn_r calculation, part of the aerodynamic
component computation.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
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


_LOGGER = logging.getLogger(__name__)


class ComputeWingLiftEffectCnr(om.Group):
    """
    Roskam data :cite:`roskampart6:1985` to estimate the lift effect in the yaw moment
    computation result from yaw rate (yaw damping). (figure 10.48)
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            name="middle_coefficient",
            subsys=_ComputeMiddleCoefficient(),
            promotes=["*"],
        )
        self.add_subsystem(
            name="cn_r_lift_effect",
            subsys=_ComputeWingLiftEffectCnr(),
            promotes=["*"],
        )


class _ComputeMiddleCoefficient(om.ExplicitComponent):
    """
    Middle coefficient from Roskam's date :cite:`roskampart6:1985` to estimate the lift effect in
    the yaw moment computation result from yaw rate (yaw damping). (figure 10.48)

    :param static_margin: distance between aft cg and aircraft aerodynamic center divided by MAC
    :param sweep_25: the sweep at 25% of the lifting surface
    :param aspect_ratio: the aspect ratio of the lifting surface

    :return middle_coeff: the intermediate coefficient for lift effect calculation
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        self.add_input("data:handling_qualities:stick_fixed_static_margin", val=np.nan)

        self.add_output("middle_coeff", val=0.02)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        self.declare_partials(of="middle_coeff", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aspect_ratio = inputs["data:geometry:wing:aspect_ratio"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        static_margin = inputs["data:handling_qualities:stick_fixed_static_margin"]

        if static_margin != np.clip(static_margin, 0.0, 0.4):
            static_margin = np.clip(static_margin, 0.0, 0.4)
            _LOGGER.warning("Static margin is outside of the range in Roskam's book, value clipped")

        if sweep_25 != np.clip(sweep_25, 0.0, 60.0):
            sweep_25 = np.clip(sweep_25, 0.0, 60.0)
            _LOGGER.warning(
                "Sweep at 25% chord is outside of the range in Roskam's book, value clipped"
            )

        if aspect_ratio != np.clip(aspect_ratio, 0.7, 10.0):
            aspect_ratio = np.clip(aspect_ratio, 0.7, 10.0)
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        outputs["middle_coeff"] = (
            0.22600459
            + 5.44089461 * static_margin
            + 0.10640558 * sweep_25
            + 0.64416073 * aspect_ratio
            - 3.61131328 * static_margin**2.0
            + 0.00291060 * static_margin * sweep_25
            - 1.71233633 * static_margin * aspect_ratio
            - 0.00431660 * sweep_25**2.0
            - 0.00706544 * sweep_25 * aspect_ratio
            - 0.02266722 * aspect_ratio**2.0
            - 2.60205954 * static_margin**3.0
            + 0.12285429 * static_margin**2.0 * sweep_25
            + 0.34163936 * static_margin**2.0 * aspect_ratio
            + 0.00055887 * static_margin * sweep_25**2.0
            - 0.00839912 * static_margin * sweep_25 * aspect_ratio
            + 0.13509774 * static_margin * aspect_ratio**2.0
            + 0.00006179 * sweep_25**3.0
            - 0.00013279 * sweep_25**2.0 * aspect_ratio
            + 0.00109178 * sweep_25 * aspect_ratio**2.0
            - 0.00229362 * aspect_ratio**3.0
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        aspect_ratio = inputs["data:geometry:wing:aspect_ratio"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        static_margin = inputs["data:handling_qualities:stick_fixed_static_margin"]

        partials["middle_coeff", "data:geometry:wing:aspect_ratio"] = np.where(
            aspect_ratio == np.clip(aspect_ratio, 0.7, 10.0),
            (
                0.64416073
                - 0.04533444 * aspect_ratio
                - 1.71233633 * static_margin
                - 0.00706544 * sweep_25
                + 0.34163936 * static_margin**2.0
                - 0.00839912 * static_margin * sweep_25
                + 0.27019548 * static_margin * aspect_ratio
                - 0.00013279 * sweep_25**2.0
                + 0.00218356 * sweep_25 * aspect_ratio
                - 0.00688086 * aspect_ratio**2.0
            ),
            1e-6,
        )

        partials["middle_coeff", "data:geometry:wing:sweep_25"] = np.where(
            sweep_25 == np.clip(sweep_25, 0.0, 60.0),
            (
                0.10640558
                + 0.00291060 * static_margin
                - 0.0086332 * sweep_25
                - 0.00706544 * aspect_ratio
                + 0.12285429 * static_margin**2.0
                + 0.00111774 * static_margin * sweep_25
                - 0.00839912 * static_margin * aspect_ratio
                + 0.00018537 * sweep_25**2.0
                - 0.00026558 * sweep_25 * aspect_ratio
                + 0.00109178 * aspect_ratio**2.0
            ),
            1e-6,
        )

        partials["middle_coeff", "data:handling_qualities:stick_fixed_static_margin"] = np.where(
            static_margin == np.clip(static_margin, 0.0, 0.4),
            (
                5.44089461
                - 7.22262656 * static_margin
                + 0.00291060 * sweep_25
                - 1.71233633 * aspect_ratio
                - 7.80617862 * static_margin**2.0
                + 0.24570858 * static_margin * sweep_25
                + 0.68327872 * static_margin * aspect_ratio
                + 0.00055887 * sweep_25**2.0
                - 0.00839912 * sweep_25 * aspect_ratio
                + 0.13509774 * aspect_ratio**2.0
            ),
            1e-6,
        )


class _ComputeWingLiftEffectCnr(om.ExplicitComponent):
    """
    Roskam data :cite:`roskampart6:1985` to estimate the lift effect in the yaw moment
    computation result from yaw rate (yaw damping). (figure 10.48)

    :param middle_coeff: the intermediate coefficient for lift effect calculation derived from
    3rd-order regression model
    :param taper_ratio: the taper ratio of the lifting surface

    :return lift_effect: the effect of lift fot the computation of the yaw moment due to yaw
    rate
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("middle_coeff", val=np.nan)

        self.add_output("lift_effect", val=0.02)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        self.declare_partials(of="lift_effect", wrt="middle_coeff", val=0.5)
        self.declare_partials(of="lift_effect", wrt="data:geometry:wing:taper_ratio", val=-0.15)

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["lift_effect"] = 0.5 * (
            inputs["middle_coeff"] - 2.7 - 0.3 * inputs["data:geometry:wing:taper_ratio"]
        )
