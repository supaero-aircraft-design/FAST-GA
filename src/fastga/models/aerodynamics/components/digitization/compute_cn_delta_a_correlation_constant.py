"""
Python module for deflection correlation constant of aileron Cn calculation, part of the aerodynamic
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


class ComputeAileronYawCorrelationConstant(om.ExplicitComponent):
    """
    Roskam data :cite:`roskampart6:1985` to estimate the correlation constant for the computation
    of the yaw moment due to aileron. (figure 10.48)
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:aileron:span_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)

        self.add_output("aileron_correlation_constant", val=0.02, units="unitless")

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        self.declare_partials(of="aileron_correlation_constant", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        aspect_ratio = inputs["data:geometry:wing:aspect_ratio"]
        aileron_inner_span_ratio = 1.0 - inputs["data:geometry:wing:aileron:span_ratio"]

        if taper_ratio != np.clip(taper_ratio, 0.25, 1.0):
            taper_ratio = np.clip(taper_ratio, 0.25, 1.0)
            _LOGGER.warning("Taper ratio is outside of the range in Roskam's book, value clipped")

        if aspect_ratio != np.clip(aspect_ratio, 3.0, 8.0):
            aspect_ratio = np.clip(aspect_ratio, 3.0, 8.0)
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        if aileron_inner_span_ratio != np.clip(aileron_inner_span_ratio, 0.0, 0.91):
            aileron_inner_span_ratio = np.clip(aileron_inner_span_ratio, 0.0, 0.91)
            _LOGGER.warning(
                "Aileron inboard location is outside of the range in Roskam's book, value clipped"
            )

        outputs["aileron_correlation_constant"] = (
            -0.63632822
            - 0.57304492 * taper_ratio
            + 0.21707743 * aspect_ratio
            + 0.10407542 * aileron_inner_span_ratio
            + 0.61587892 * taper_ratio**2.0
            + 0.04971284 * taper_ratio * aspect_ratio
            - 0.20067475 * taper_ratio * aileron_inner_span_ratio
            - 0.02997066 * aspect_ratio**2.0
            - 0.00550507 * aspect_ratio * aileron_inner_span_ratio
            - 0.00622554 * aileron_inner_span_ratio**2.0
            - 0.23509612 * taper_ratio**3.0
            - 0.02728276 * taper_ratio**2.0 * aspect_ratio
            + 0.12690424 * taper_ratio**2.0 * aileron_inner_span_ratio
            - 0.00050367 * taper_ratio * aspect_ratio**2.0
            - 0.01379183 * taper_ratio * aspect_ratio * aileron_inner_span_ratio
            + 0.04289006 * taper_ratio * aileron_inner_span_ratio**2.0
            + 0.00140254 * aspect_ratio**3.0
            + 0.00149011 * aspect_ratio**2.0 * aileron_inner_span_ratio
            - 0.00553262 * aspect_ratio * aileron_inner_span_ratio**2.0
            - 0.06766704 * aileron_inner_span_ratio**3.0
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        aspect_ratio = inputs["data:geometry:wing:aspect_ratio"]
        span_ratio = inputs["data:geometry:wing:aileron:span_ratio"]

        taper_ratio_clipped = np.clip(taper_ratio, 0.25, 1.0)
        aspect_ratio_clipped = np.clip(aspect_ratio, 3.0, 8.0)
        span_ratio_clipped = np.clip(span_ratio, 0.09, 1.0)

        partials["aileron_correlation_constant", "data:geometry:wing:taper_ratio"] = np.where(
            taper_ratio == np.clip(taper_ratio, 0.25, 1.0),
            (
                -0.00050367 * aspect_ratio_clipped**2.0
                + 0.01379183 * aspect_ratio_clipped * span_ratio_clipped
                - 0.05456552 * aspect_ratio_clipped * taper_ratio_clipped
                + 0.03592101 * aspect_ratio_clipped
                + 0.04289006 * span_ratio_clipped**2.0
                - 0.25380848 * span_ratio_clipped * taper_ratio_clipped
                + 0.11489463 * span_ratio_clipped
                - 0.70528836 * taper_ratio_clipped**2.0
                + 1.48556632 * taper_ratio_clipped
                - 0.73082961
            ),
            1e-6,
        )

        partials["aileron_correlation_constant", "data:geometry:wing:aspect_ratio"] = np.where(
            aspect_ratio == aspect_ratio_clipped,
            (
                0.00420762 * aspect_ratio_clipped**2.0
                - 0.00298022 * aspect_ratio_clipped * span_ratio_clipped
                - 0.00100734 * aspect_ratio_clipped * taper_ratio_clipped
                - 0.0569611 * aspect_ratio_clipped
                - 0.00553262 * span_ratio_clipped**2.0
                + 0.01379183 * span_ratio_clipped * taper_ratio_clipped
                + 0.01657031 * span_ratio_clipped
                - 0.02728276 * taper_ratio_clipped**2.0
                + 0.03592101 * taper_ratio_clipped
                + 0.20603974
            ),
            1e-6,
        )

        partials["aileron_correlation_constant", "data:geometry:wing:aileron:span_ratio"] = (
            np.where(
                span_ratio == span_ratio_clipped,
                (
                    -0.00149011 * aspect_ratio_clipped**2.0
                    - 0.01106524 * aspect_ratio_clipped * span_ratio_clipped
                    + 0.01379183 * aspect_ratio_clipped * taper_ratio_clipped
                    + 0.01657031 * aspect_ratio_clipped
                    + 0.20300112 * span_ratio_clipped**2.0
                    + 0.08578012 * span_ratio_clipped * taper_ratio_clipped
                    - 0.41845332 * span_ratio_clipped
                    - 0.12690424 * taper_ratio_clipped**2.0
                    + 0.11489463 * taper_ratio_clipped
                    + 0.11137678
                ),
                1e-6,
            )
        )
