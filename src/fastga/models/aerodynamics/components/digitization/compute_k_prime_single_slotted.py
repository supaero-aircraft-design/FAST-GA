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

    def setup(self):
        self.add_input("flap_angle", val=0.0, units="deg")
        self.add_input("chord_ratio", val=np.nan)

        self.add_output("lift_effectiveness", val=0.1)

        self.declare_partials(of="lift_effectiveness", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        d_angle = inputs["flap_angle"]
        cr = inputs["chord_ratio"]

        if d_angle != np.clip(d_angle, -0.08, 79.32):
            d_angle = np.clip(d_angle, -0.08, 79.32)
            _LOGGER.warning("Flap angle value outside of the range in Roskam's book, value clipped")

        if cr != np.clip(cr, 0.15, 0.4):
            cr = np.clip(cr, 0.15, 0.4)
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        outputs["lift_effectiveness"] = (
            0.0239
            + 0.006 * d_angle
            + 2.6633 * cr
            - 0.0002 * d_angle**2.0
            - 0.0121 * d_angle * cr
            - 2.9929 * cr**2.0
            - 0.0002 * d_angle**2.0 * cr
            + 0.0354 * d_angle * cr**2.0
            - 0.5931 * cr**3.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        d_angle = inputs["flap_angle"]
        cr = inputs["chord_ratio"]

        partials["lift_effectiveness", "flap_angle"] = np.where(
            d_angle == np.clip(d_angle, -0.08, 79.32),
            (0.006 - 0.0004 * d_angle - 0.0121 * cr - 0.0004 * d_angle * cr + 0.0354 * cr**2.0),
            1e-6,
        )

        partials["lift_effectiveness", "chord_ratio"] = np.where(
            cr == np.clip(cr, 0.15, 0.4),
            (
                2.6633
                - 0.0121 * d_angle
                - 5.9858 * cr
                - 0.0002 * d_angle**2.0
                + 0.0708 * d_angle * cr
                - 1.7793 * cr**2.0
            ),
            1e-6,
        )
