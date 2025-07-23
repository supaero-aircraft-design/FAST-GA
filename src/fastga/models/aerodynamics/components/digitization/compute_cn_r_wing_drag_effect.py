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
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
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
        aspect_ratio = inputs["data:geometry:wing:aspect_ratio"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        static_margin = inputs["data:handling_qualities:stick_fixed_static_margin"]

        if static_margin != np.clip(static_margin, 0.0, 0.4):
            static_margin = np.clip(static_margin, 0.0, 0.4)
            _LOGGER.warning("Static margin is outside of the range in Roskam's book, value clipped")

        if sweep_25 != np.clip(sweep_25, 0.0, 50.0):
            sweep_25 = np.clip(sweep_25, 0.0, 50.0)
            # Wing sweep is limited to 50° since transonic wing designs are not relevant for
            # propeller aircraft. This also reduces the regression model complexity.
            _LOGGER.warning(
                "Sweep at 25% chord is outside of the range in Roskam's book, value clipped"
            )

        if aspect_ratio != np.clip(aspect_ratio, 1.0, 8.0):
            aspect_ratio = np.clip(aspect_ratio, 1.0, 8.0)
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        outputs["drag_effect"] = (
            -0.61144468
            - 0.70264474 * static_margin
            + 0.00962168 * sweep_25
            + 0.44131919 * np.log(aspect_ratio)
            - 0.05601423 * static_margin**2.0
            + 0.00328669 * static_margin * sweep_25
            + 0.37739783 * static_margin * np.log(aspect_ratio)
            - 0.00024876 * sweep_25**2.0
            - 0.00235158 * sweep_25 * np.log(aspect_ratio)
            - 0.14846284 * np.log(aspect_ratio) ** 2.0
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        aspect_ratio = inputs["data:geometry:wing:aspect_ratio"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        static_margin = inputs["data:handling_qualities:stick_fixed_static_margin"]

        aspect_ratio_clipped = np.clip(aspect_ratio, 1.0, 8.0)
        static_margin_clipped = np.clip(static_margin, 0.0, 0.4)
        sweep_25_clipped = np.clip(sweep_25, 0.0, 50.0)

        partials["drag_effect", "data:geometry:wing:aspect_ratio"] = np.where(
            aspect_ratio == np.clip(aspect_ratio, 1.0, 8.0),
            (
                0.44131919
                + 0.37739783 * static_margin_clipped
                - 0.29692568 * np.log(aspect_ratio_clipped)
                - 0.00235158 * sweep_25_clipped
            )
            / aspect_ratio_clipped,
            1e-6,
        )

        partials["drag_effect", "data:geometry:wing:sweep_25"] = np.where(
            sweep_25 == np.clip(sweep_25, 0.0, 50.0),
            (
                -0.00235158 * np.log(aspect_ratio_clipped)
                + 0.00328669 * static_margin_clipped
                - 0.00049752 * sweep_25_clipped
                + 0.00962168
            ),
            1e-6,
        )

        partials["drag_effect", "data:handling_qualities:stick_fixed_static_margin"] = np.where(
            static_margin == np.clip(static_margin, 0.0, 0.4),
            (
                0.37739783 * np.log(aspect_ratio_clipped)
                - 0.11202846 * static_margin_clipped
                + 0.00328669 * sweep_25_clipped
                - 0.70264474
            ),
            1e-6,
        )
