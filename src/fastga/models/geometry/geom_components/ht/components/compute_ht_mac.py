"""
Python module for horizontal tail mean aerodynamic chord calculation, part of the horizontal
tail geometry.
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

import numpy as np
import openmdao.api as om


class ComputeHTMAC(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Horizontal tail mean aerodynamic chord estimation.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:horizontal_tail:MAC:length", units="m")
        self.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:local", units="m")
        self.add_output("data:geometry:horizontal_tail:MAC:y", units="m")

        self.declare_partials(
            "data:geometry:horizontal_tail:MAC:length",
            ["data:geometry:horizontal_tail:root:chord", "data:geometry:horizontal_tail:tip:chord"],
            method="exact",
        )
        self.declare_partials(
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            [
                "data:geometry:horizontal_tail:root:chord",
                "data:geometry:horizontal_tail:tip:chord",
                "data:geometry:horizontal_tail:sweep_25",
                "data:geometry:horizontal_tail:span",
            ],
            method="exact",
        )
        self.declare_partials(
            "data:geometry:horizontal_tail:MAC:y",
            [
                "data:geometry:horizontal_tail:root:chord",
                "data:geometry:horizontal_tail:tip:chord",
                "data:geometry:horizontal_tail:span",
            ],
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        b_h = inputs["data:geometry:horizontal_tail:span"]

        chord_sum = root_chord + tip_chord

        mac_ht = (chord_sum**2.0 - root_chord * tip_chord) / (1.5 * chord_sum)

        x0_ht = (
            (root_chord * 0.5 + b_h * np.tan(sweep_25_ht) - tip_chord * 0.5)
            * (chord_sum + tip_chord)
            / (6.0 * chord_sum)
        )

        y0_ht = b_h * (chord_sum + tip_chord) / (6.0 * chord_sum)

        outputs["data:geometry:horizontal_tail:MAC:length"] = mac_ht
        outputs["data:geometry:horizontal_tail:MAC:at25percent:x:local"] = x0_ht
        outputs["data:geometry:horizontal_tail:MAC:y"] = y0_ht

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        b_h = inputs["data:geometry:horizontal_tail:span"]
        chord_sum = root_chord + tip_chord

        partials[
            "data:geometry:horizontal_tail:MAC:length", "data:geometry:horizontal_tail:root:chord"
        ] = (1.0 - (tip_chord / chord_sum) ** 2.0) / 1.5
        partials[
            "data:geometry:horizontal_tail:MAC:length", "data:geometry:horizontal_tail:tip:chord"
        ] = (1.0 - (root_chord / chord_sum) ** 2.0) / 1.5

        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            "data:geometry:horizontal_tail:root:chord",
        ] = (chord_sum**2.0 + 2.0 * tip_chord * (tip_chord - b_h * np.tan(sweep_25_ht))) / (
            12.0 * chord_sum**2.0
        )
        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            "data:geometry:horizontal_tail:tip:chord",
        ] = (root_chord * (root_chord + b_h * np.tan(sweep_25_ht)) - chord_sum**2.0) / (
            6.0 * chord_sum**2.0
        )
        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            "data:geometry:horizontal_tail:sweep_25",
        ] = b_h * (chord_sum + tip_chord) / (6.0 * chord_sum * np.cos(sweep_25_ht) ** 2.0)
        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            "data:geometry:horizontal_tail:span",
        ] = (chord_sum + tip_chord) * np.tan(sweep_25_ht) / (6.0 * chord_sum)

        partials["data:geometry:horizontal_tail:MAC:y", "data:geometry:horizontal_tail:span"] = (
            chord_sum + tip_chord
        ) / (6.0 * chord_sum)
        partials[
            "data:geometry:horizontal_tail:MAC:y", "data:geometry:horizontal_tail:root:chord"
        ] = -b_h * tip_chord / (6.0 * chord_sum**2.0)
        partials[
            "data:geometry:horizontal_tail:MAC:y", "data:geometry:horizontal_tail:tip:chord"
        ] = b_h * root_chord / (6.0 * chord_sum**2.0)
