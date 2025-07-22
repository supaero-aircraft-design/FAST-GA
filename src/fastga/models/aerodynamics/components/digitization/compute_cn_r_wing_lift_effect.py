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


class ComputeIntermediateParameter(om.ExplicitComponent):
    """
    Middle coefficient from Roskam's date :cite:`roskampart6:1985` to estimate the lift effect in
    the yaw moment computation result from yaw rate (yaw damping). (figure 10.44)
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("ln_ar", val=np.nan)
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
        ln_ar = inputs["ln_ar"]
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

        if ln_ar != np.clip(ln_ar, np.log(0.7), np.log(10.0)):
            ln_ar = np.clip(ln_ar, np.log(0.7), np.log(10.0))
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        outputs["middle_coeff"] = (
            0.44349717
            + 4.53744392 * static_margin
            + 0.08353393 * sweep_25
            + 2.01798623 * ln_ar
            - 3.58167734 * static_margin**2
            + 0.00871889 * static_margin * sweep_25
            - 5.14471767 * static_margin * ln_ar
            - 0.00407792 * sweep_25**2
            + 0.00001685 * sweep_25 * ln_ar
            - 0.38974151 * ln_ar**2
            - 2.51200192 * static_margin**3
            + 0.11946688 * static_margin**2 * sweep_25
            + 1.34092802 * static_margin**2 * ln_ar
            + 0.00096608 * static_margin * sweep_25**2
            - 0.04698288 * static_margin * sweep_25 * ln_ar
            + 1.62056258 * static_margin * ln_ar**2
            + 0.00006323 * sweep_25**3
            - 0.00075789 * sweep_25**2 * ln_ar
            + 0.01156277 * sweep_25 * ln_ar**2
            - 0.04624397 * ln_ar**3
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ln_ar = inputs["ln_ar"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        static_margin = inputs["data:handling_qualities:stick_fixed_static_margin"]

        lar = np.clip(ln_ar, np.log(0.7), np.log(10.0))
        sm = np.clip(static_margin, 0.0, 0.4)
        sw = np.clip(sweep_25, 0.0, 60.0)

        partials["middle_coeff", "ln_ar"] = np.where(
            ln_ar == np.clip(ln_ar, np.log(0.7), np.log(10.0)),
            (
                -0.13873191 * lar**2
                + 3.24112516 * lar * sm
                + 0.02312554 * lar * sw
                - 0.77948302 * lar
                + 1.34092802 * sm**2
                - 0.04698288 * sm * sw
                - 5.14471767 * sm
                - 0.00075789 * sw**2
                + 1.685e-5 * sw
                + 2.01798623
            ),
            1e-6,
        )

        partials["middle_coeff", "data:geometry:wing:sweep_25"] = np.where(
            sweep_25 == np.clip(sweep_25, 0.0, 60.0),
            (
                0.01156277 * lar**2
                - 0.04698288 * lar * sm
                - 0.00151578 * lar * sw
                + 1.685e-5 * lar
                + 0.11946688 * sm**2
                + 0.00193216 * sm * sw
                + 0.00871889 * sm
                + 0.00018969 * sw**2
                - 0.00815584 * sw
                + 0.08353393
            ),
            1e-6,
        )

        partials["middle_coeff", "data:handling_qualities:stick_fixed_static_margin"] = np.where(
            static_margin == np.clip(static_margin, 0.0, 0.4),
            (
                1.62056258 * lar**2
                + 2.68185604 * lar * sm
                - 0.04698288 * lar * sw
                - 5.14471767 * lar
                - 7.53600576 * sm**2
                + 0.23893376 * sm * sw
                - 7.16335468 * sm
                + 0.00096608 * sw**2
                + 0.00871889 * sw
                + 4.53744392
            ),
            1e-6,
        )


class ComputeWingLiftEffectCnr(om.ExplicitComponent):
    """
    Roskam data :cite:`roskampart6:1985` to estimate the lift effect in the yaw moment
    computation result from yaw rate (yaw damping). (figure 10.44)
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
