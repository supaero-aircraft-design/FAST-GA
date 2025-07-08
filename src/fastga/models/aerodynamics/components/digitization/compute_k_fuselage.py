#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Airchord_ratioaft Design
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

import numpy as np
import openmdao.api as om
import logging

_LOGGER = logging.getLogger(__name__)


class ComputeFuselagePitchMomentFactor(om.ExplicitComponent):
    """
    Raymer data to estimate the empirical pitching moment factor K_fus (figure 16.14),
    from :cite:'raymer:2012'.

    :param root_quarter_chord_position_ratio: the position of the root quarter chord of the
    wing from the nose.
    divided by the total length of the fuselage.
    :return k_fus: the empirical pitching moment factor.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("x0_ratio", val=np.nan)

        self.add_output("fuselage_pitch_moment_factor", val=0.02)

        self.declare_partials(of="fuselage_pitch_moment_factor", wrt="x0_ratio", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x0_ratio = inputs["x0_ratio"]

        if x0_ratio != np.clip(inputs["x0_ratio"], 0.1, 0.62):
            x0_ratio = np.clip(inputs["x0_ratio"], 0.1, 0.62)
            _LOGGER.warning(
                "Thickness ratio value outside of the range in Raymer's book, value clipped"
            )

        outputs["fuselage_pitch_moment_factor"] = 0.01 - 0.063 * x0_ratio + 0.211 * x0_ratio**2.0

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        x0_ratio = inputs["x0_ratio"]

        partials["fuselage_pitch_moment_factor", "x0_ratio"] = np.where(
            x0_ratio == np.clip(x0_ratio, 0.1, 0.62),
            (-0.063 + 0.422 * x0_ratio),
            1e-6,
        )
