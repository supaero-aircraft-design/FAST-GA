"""
Python module for wing twist contribution of Cn_p calculation, part of the aerodynamic
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


class ComputeWingTwistContributionCnp(om.ExplicitComponent):
    """
    Roskam data :cite:`roskampart6:1985` to estimate the contribution to the yaw moment of the
    twist of the lifting surface. (figure 10.37)
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)

        self.add_output("twist_contribution_cn_p", val=0.1)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        self.declare_partials(of="twist_contribution_cn_p", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        wing_taper_ratio = inputs["data:geometry:wing:taper_ratio"]

        if wing_taper_ratio != np.clip(wing_taper_ratio, 0.0, 1.0):
            wing_taper_ratio = np.clip(wing_taper_ratio, 0.0, 1.0)
            _LOGGER.warning("Taper ratio is outside of the range in Roskam's book, value clipped")

        if wing_ar != np.clip(wing_ar, 2.0, 12.0):
            wing_ar = np.clip(wing_ar, 2.0, 12.0)
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        outputs["twist_contribution_cn_p"] = (
            -4.845e-04
            + 2.411e-04 * wing_ar
            - 1.640e-05 * wing_ar**2.0
            + 5.393e-07 * wing_ar**3.0
            - 3.017e-03 * wing_taper_ratio
            + 2.460e-04 * wing_taper_ratio * wing_ar
            - 1.190e-06 * wing_taper_ratio * wing_ar**2
            + 2.768e-03 * wing_taper_ratio**2
            - 2.295e-04 * wing_taper_ratio**2 * wing_ar
            - 6.752e-04 * wing_taper_ratio**3
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        wing_taper_ratio = inputs["data:geometry:wing:taper_ratio"]

        partials["twist_contribution_cn_p", "data:geometry:wing:aspect_ratio"] = np.where(
            wing_ar == np.clip(wing_ar, 2.0, 12.0),
            (
                2.411e-04
                - 3.280e-05 * wing_ar
                + 2.460e-04 * wing_taper_ratio
                - 2.380e-06 * wing_taper_ratio * wing_ar
                - 2.295e-04 * wing_taper_ratio**2.0
                + 1.6179e-06 * wing_ar**2.0
            ),
            1e-9,
        )

        partials["twist_contribution_cn_p", "data:geometry:wing:taper_ratio"] = np.where(
            wing_taper_ratio == np.clip(wing_taper_ratio, 0.0, 1.0),
            (
                -3.017e-03
                + 2.460e-04 * wing_ar
                - 1.190e-06 * wing_ar**2.0
                + 5.536e-03 * wing_taper_ratio
                - 4.59e-04 * wing_taper_ratio * wing_ar
                - 2.0256e-03 * wing_taper_ratio**2.0
            ),
            1e-9,
        )
