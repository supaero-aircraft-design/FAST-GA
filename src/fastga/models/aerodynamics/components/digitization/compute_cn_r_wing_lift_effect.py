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

        if sweep_25 != np.clip(sweep_25, 0.0, 50.0):
            sweep_25 = np.clip(sweep_25, 0.0, 50.0)
            _LOGGER.warning(
                "Sweep at 25% chord is outside of the range in Roskam's book, value clipped"
            )

        if ln_ar != np.clip(ln_ar, np.log(0.7), np.log(10.0)):
            ln_ar = np.clip(ln_ar, np.log(0.7), np.log(10.0))
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        outputs["middle_coeff"] = (
            0.46236794
            + 5.91939258 * static_margin
            - 0.00000138 * sweep_25
            + 1.80469105 * ln_ar
            - 6.62557472 * static_margin**2
            - 0.04095218 * static_margin * sweep_25
            - 5.91325304 * static_margin * ln_ar
            - 0.00003059 * sweep_25**2
            - 0.00142883 * sweep_25 * ln_ar
            - 0.19838867 * ln_ar**2
            - 4.44889624 * static_margin**3
            + 0.17317666 * static_margin**2 * sweep_25
            + 3.79534719 * static_margin**2 * ln_ar
            + 0.00160670 * static_margin * sweep_25**2
            - 0.04621831 * static_margin * sweep_25 * ln_ar
            + 1.54501070 * static_margin * ln_ar**2
            + 0.00001442 * sweep_25**3
            - 0.00060596 * sweep_25**2 * ln_ar
            + 0.00926858 * sweep_25 * ln_ar**2
            - 0.08765596 * ln_ar**3
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ln_ar = inputs["ln_ar"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        static_margin = inputs["data:handling_qualities:stick_fixed_static_margin"]

        lar = np.clip(ln_ar, np.log(0.7), np.log(10.0))
        sm = np.clip(static_margin, 0.0, 0.4)
        sw = np.clip(sweep_25, 0.0, 50.0)

        partials["middle_coeff", "ln_ar"] = np.where(
            ln_ar == np.clip(ln_ar, np.log(0.7), np.log(10.0)),
            (
                -0.26296788 * lar**2
                + 3.0900214 * lar * sm
                + 0.01853716 * lar * sw
                - 0.39677734 * lar
                + 3.79534719 * sm**2
                - 0.04621831 * sm * sw
                - 5.91325304 * sm
                - 0.00060596 * sw**2
                - 0.00142883 * sw
                + 1.80469105
            ),
            1e-6,
        )

        partials["middle_coeff", "data:geometry:wing:sweep_25"] = np.where(
            sweep_25 == np.clip(sweep_25, 0.0, 50.0),
            (
                0.00926858 * lar**2
                - 0.04621831 * lar * sm
                - 0.00121192 * lar * sw
                - 0.00142883 * lar
                + 0.17317666 * sm**2
                + 0.0032134 * sm * sw
                - 0.04095218 * sm
                + 4.326e-5 * sw**2
                - 6.118e-5 * sw
                - 1.38e-6
            ),
            1e-6,
        )

        partials["middle_coeff", "data:handling_qualities:stick_fixed_static_margin"] = np.where(
            static_margin == np.clip(static_margin, 0.0, 0.4),
            (
                1.5450107 * lar**2
                + 7.59069438 * lar * sm
                - 0.04621831 * lar * sw
                - 5.91325304 * lar
                - 13.34668872 * sm**2
                + 0.34635332 * sm * sw
                - 13.25114944 * sm
                + 0.0016067 * sw**2
                - 0.04095218 * sw
                + 5.91939258
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
