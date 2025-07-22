"""
Python module for wing drag effect on Cn_r calculation, part of the aerodynamic
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


class ComputeWingDragEffectCnr(om.ExplicitComponent):
    """
    Roskam data :cite:`roskampart6:1985` to estimate the drag effect in the yaw moment
    computation result from yaw rate (yaw damping). (figure 10.45)
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("ln_ar", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        self.add_input("data:handling_qualities:stick_fixed_static_margin", val=np.nan)

        self.add_output("drag_effect", val=0.02, units="unitless")

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        self.declare_partials(of="drag_effect", wrt="*", method="exact")

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

        if ln_ar != np.clip(ln_ar, np.log(1.0), np.log(8.0)):
            ln_ar = np.clip(ln_ar, np.log(1.0), np.log(8.0))
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        outputs["drag_effect"] = (
            -0.59726930
            - 0.83734303 * static_margin
            - 0.01801259 * sweep_25
            + 0.53191492 * ln_ar
            - 0.38000723 * static_margin**2
            - 0.00186004 * static_margin * sweep_25
            + 1.05520927 * static_margin * ln_ar
            + 0.00093863 * sweep_25**2
            - 0.00036945 * sweep_25 * ln_ar
            - 0.35723706 * ln_ar**2
            - 0.16101689 * static_margin**3
            + 0.00282320 * static_margin**2 * sweep_25
            + 0.33612886 * static_margin**2 * ln_ar
            + 0.00015169 * static_margin * sweep_25**2
            - 0.00285983 * static_margin * sweep_25 * ln_ar
            - 0.35821746 * static_margin * ln_ar**2
            - 0.00001259 * sweep_25**3
            - 0.00007595 * sweep_25**2 * ln_ar
            + 0.00101888 * sweep_25 * ln_ar**2
            + 0.08306687 * ln_ar**3
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ln_ar = inputs["ln_ar"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        static_margin = inputs["data:handling_qualities:stick_fixed_static_margin"]

        lar = np.clip(ln_ar, np.log(1.0), np.log(8.0))
        sm = np.clip(static_margin, 0.0, 0.4)
        sw = np.clip(sweep_25, 0.0, 60.0)

        partials["drag_effect", "ln_ar"] = np.where(
            ln_ar == np.clip(ln_ar, np.log(1.0), np.log(8.0)),
            (
                0.24920061 * lar**2
                - 0.71643492 * lar * sm
                + 0.00203776 * lar * sw
                - 0.71447412 * lar
                + 0.33612886 * sm**2
                - 0.00285983 * sm * sw
                + 1.05520927 * sm
                - 7.595e-5 * sw**2
                - 0.00036945 * sw
                + 0.53191492
            ),
            1e-6,
        )

        partials["drag_effect", "data:geometry:wing:sweep_25"] = np.where(
            sweep_25 == np.clip(sweep_25, 0.0, 60.0),
            (
                0.00101888 * lar**2
                - 0.00285983 * lar * sm
                - 0.0001519 * lar * sw
                - 0.00036945 * lar
                + 0.0028232 * sm**2
                + 0.00030338 * sm * sw
                - 0.00186004 * sm
                - 3.777e-5 * sw**2
                + 0.00187726 * sw
                - 0.01801259
            ),
            1e-6,
        )

        partials["drag_effect", "data:handling_qualities:stick_fixed_static_margin"] = np.where(
            static_margin == np.clip(static_margin, 0.0, 0.4),
            (
                -0.35821746 * lar**2
                + 0.67225772 * lar * sm
                - 0.00285983 * lar * sw
                + 1.05520927 * lar
                - 0.48305067 * sm**2
                + 0.0056464 * sm * sw
                - 0.76001446 * sm
                + 0.00015169 * sw**2
                - 0.00186004 * sw
                - 0.83734303
            ),
            1e-6,
        )
