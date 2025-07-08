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

import numpy as np
import openmdao.api as om
import logging

_LOGGER = logging.getLogger(__name__)


class ComputeSingleSlottedLiftEffectiveness(om.ExplicitComponent):
    """
    Roskam data to estimate the lift effectiveness of a single slotted flap (figure 8.17),
    noted here k_prime to match the notation of the plain flap but is written alpha_delta in
    the book.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("flap_angle", val=0.0, units="deg")
        self.add_input("chord_ratio", val=np.nan)

        self.add_output("lift_effectiveness", val=0.1)

        self.declare_partials(of="lift_effectiveness", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        flap_angle = inputs["flap_angle"]
        chord_ratio = inputs["chord_ratio"]

        if flap_angle != np.clip(flap_angle, -0.08, 79.32):
            flap_angle = np.clip(flap_angle, -0.08, 79.32)
            _LOGGER.warning("Flap angle value outside of the range in Roskam's book, value clipped")

        if chord_ratio != np.clip(chord_ratio, 0.15, 0.4):
            chord_ratio = np.clip(chord_ratio, 0.15, 0.4)
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        outputs["lift_effectiveness"] = (
            0.0239
            + 0.006 * flap_angle
            + 2.6633 * chord_ratio
            - 0.0002 * flap_angle**2.0
            - 0.0121 * flap_angle * chord_ratio
            - 2.9929 * chord_ratio**2.0
            - 0.0002 * flap_angle**2.0 * chord_ratio
            + 0.0354 * flap_angle * chord_ratio**2.0
            - 0.5931 * chord_ratio**3.0
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        flap_angle = inputs["flap_angle"]
        chord_ratio = inputs["chord_ratio"]

        partials["lift_effectiveness", "flap_angle"] = np.where(
            flap_angle == np.clip(flap_angle, -0.08, 79.32),
            (
                0.006
                - 0.0004 * flap_angle
                - 0.0121 * chord_ratio
                - 0.0004 * flap_angle * chord_ratio
                + 0.0354 * chord_ratio**2.0
            ),
            1e-6,
        )

        partials["lift_effectiveness", "chord_ratio"] = np.where(
            chord_ratio == np.clip(chord_ratio, 0.15, 0.4),
            (
                2.6633
                - 0.0121 * flap_angle
                - 5.9858 * chord_ratio
                - 0.0002 * flap_angle**2.0
                + 0.0708 * flap_angle * chord_ratio
                - 1.7793 * chord_ratio**2.0
            ),
            1e-6,
        )
