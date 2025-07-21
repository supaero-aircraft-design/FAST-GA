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
    computation result from yaw rate (yaw damping). (figure 10.48)

    :param static_margin: distance between aft cg and aircraft aerodynamic center divided by MAC
    :param sweep_25: the sweep at 25% of the lifting surface
    :param aspect_ratio: the aspect ratio of the lifting surface

    :return drag_effect: the effect of drag for the computation of the yaw moment due to yaw rate
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

        if sweep_25 != np.clip(sweep_25, 0.0, 60.0):
            sweep_25 = np.clip(sweep_25, 0.0, 60.0)
            _LOGGER.warning(
                "Sweep at 25% chord is outside of the range in Roskam's book, value clipped"
            )

        if aspect_ratio != np.clip(aspect_ratio, 1.0, 8.0):
            aspect_ratio = np.clip(aspect_ratio, 1.0, 8.0)
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        outputs["drag_effect"] = (
            -0.86676535
            - 1.00338040 * static_margin
            - 0.01526337 * sweep_25
            + 0.36546427 * aspect_ratio
            - 0.28212678 * static_margin**2.0
            - 0.00177067 * static_margin * sweep_25
            + 0.36960477 * static_margin * aspect_ratio
            + 0.00090898 * sweep_25**2.0
            - 0.00123072 * sweep_25 * aspect_ratio
            - 0.07411481 * aspect_ratio**2.0
            - 0.08900564 * static_margin**3.0
            - 0.00114702 * static_margin**2.0 * sweep_25
            + 0.09569751 * static_margin**2.0 * aspect_ratio
            + 0.00016781 * static_margin * sweep_25**2.0
            - 0.00074218 * static_margin * sweep_25 * aspect_ratio
            - 0.03526457 * static_margin * aspect_ratio**2.0
            - 0.00001244 * sweep_25**3.0
            - 0.00001957 * sweep_25**2.0 * aspect_ratio
            + 0.00019188 * sweep_25 * aspect_ratio**2.0
            + 0.00470529 * aspect_ratio**3.0
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        aspect_ratio = inputs["data:geometry:wing:aspect_ratio"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        static_margin = inputs["data:handling_qualities:stick_fixed_static_margin"]

        partials["drag_effect", "data:geometry:wing:aspect_ratio"] = np.where(
            aspect_ratio == np.clip(aspect_ratio, 1.0, 8.0),
            (
                0.36546427
                - 0.14822962 * aspect_ratio
                + 0.36960477 * static_margin
                - 0.00123072 * sweep_25
                + 0.09569751 * static_margin**2.0
                - 0.00074218 * static_margin * sweep_25
                - 0.07052914 * static_margin * aspect_ratio
                - 0.00001957 * sweep_25**2.0
                + 0.00038376 * sweep_25 * aspect_ratio
                + 0.01411587 * aspect_ratio**2.0
            ),
            1e-6,
        )

        partials["drag_effect", "data:geometry:wing:sweep_25"] = np.where(
            sweep_25 == np.clip(sweep_25, 0.0, 60.0),
            (
                -0.01526337
                - 0.00177067 * static_margin
                + 0.00181796 * sweep_25
                - 0.00123072 * aspect_ratio
                - 0.00114702 * static_margin**2.0
                + 0.00033562 * static_margin * sweep_25
                - 0.00074218 * static_margin * aspect_ratio
                - 0.00003732 * sweep_25**2.0
                - 0.00003914 * sweep_25 * aspect_ratio
                + 0.00019188 * aspect_ratio**2.0
            ),
            1e-6,
        )

        partials["drag_effect", "data:handling_qualities:stick_fixed_static_margin"] = np.where(
            static_margin == np.clip(static_margin, 0.0, 0.4),
            (
                -1.00338040
                - 0.56425356 * static_margin
                - 0.00177067 * sweep_25
                + 0.36960477 * aspect_ratio
                - 0.26701692 * static_margin**2.0
                - 0.00229404 * static_margin * sweep_25
                + 0.19139502 * static_margin * aspect_ratio
                + 0.00016781 * sweep_25**2.0
                - 0.00074218 * sweep_25 * aspect_ratio
                - 0.03526457 * aspect_ratio**2.0
            ),
            1e-6,
        )
