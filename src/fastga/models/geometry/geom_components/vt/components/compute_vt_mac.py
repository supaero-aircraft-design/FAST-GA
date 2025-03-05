"""
Estimation of vertical tail mean aerodynamic chords
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


class ComputeVTMacFD(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Vertical tail mean aerodynamic chord estimation based on (F)ixed tail (D)istance.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:vertical_tail:MAC:length", units="m")
        self.add_output("data:geometry:vertical_tail:MAC:at25percent:x:local", units="m")
        self.add_output("data:geometry:vertical_tail:MAC:z", units="m")

        self.declare_partials(
            "data:geometry:vertical_tail:MAC:length",
            ["data:geometry:vertical_tail:root:chord", "data:geometry:vertical_tail:tip:chord"],
            method="exact",
        )
        self.declare_partials(
            "data:geometry:vertical_tail:MAC:at25percent:x:local", "*", method="exact"
        )
        self.declare_partials(
            "data:geometry:vertical_tail:MAC:z",
            [
                "data:geometry:vertical_tail:root:chord",
                "data:geometry:vertical_tail:tip:chord",
                "data:geometry:vertical_tail:span",
            ],
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        chord_sum = root_chord + tip_chord

        mac_vt = (chord_sum**2 - root_chord * tip_chord) / (1.5 * chord_sum)
        x0_vt = (
            (root_chord + 4 * b_v * np.tan(sweep_25_vt) - tip_chord) * (chord_sum + tip_chord)
        ) / (12 * chord_sum)
        z0_vt = b_v * (chord_sum + tip_chord) / (3 * chord_sum)

        outputs["data:geometry:vertical_tail:MAC:length"] = mac_vt
        outputs["data:geometry:vertical_tail:MAC:at25percent:x:local"] = x0_vt
        outputs["data:geometry:vertical_tail:MAC:z"] = z0_vt

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        chord_sum = root_chord + tip_chord

        partials[
            "data:geometry:vertical_tail:MAC:length", "data:geometry:vertical_tail:root:chord"
        ] = (1 - (tip_chord / chord_sum) ** 2) / 1.5
        partials[
            "data:geometry:vertical_tail:MAC:length", "data:geometry:vertical_tail:tip:chord"
        ] = (1 - (root_chord / chord_sum) ** 2) / 1.5

        partials["data:geometry:vertical_tail:MAC:z", "data:geometry:vertical_tail:span"] = (
            chord_sum + tip_chord
        ) / (3 * chord_sum)
        partials["data:geometry:vertical_tail:MAC:z", "data:geometry:vertical_tail:root:chord"] = (
            -b_v * tip_chord / (3 * chord_sum**2)
        )
        partials["data:geometry:vertical_tail:MAC:z", "data:geometry:vertical_tail:tip:chord"] = (
            b_v * root_chord / (3 * chord_sum**2)
        )

        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:local",
            "data:geometry:vertical_tail:root:chord",
        ] = (chord_sum**2 + 2 * tip_chord * (tip_chord - 2 * b_v * np.tan(sweep_25_vt))) / (
            12 * chord_sum**2
        )
        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:local",
            "data:geometry:vertical_tail:tip:chord",
        ] = -(chord_sum**2 - root_chord * (root_chord + 2 * b_v * np.tan(sweep_25_vt))) / (
            6 * chord_sum**2
        )
        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:local",
            "data:geometry:vertical_tail:sweep_25",
        ] = (b_v * (chord_sum + tip_chord)) / (3 * chord_sum * np.cos(sweep_25_vt) ** 2)
        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:local",
            "data:geometry:vertical_tail:span",
        ] = (chord_sum + tip_chord) * np.tan(sweep_25_vt) / (3 * chord_sum)


class ComputeVTMacFL(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Vertical tail mean aerodynamic chord estimation based on (F)ixed fuselage (L)ength (VTP
    distance computed).
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:vertical_tail:MAC:length", units="m")
        self.add_output("data:geometry:vertical_tail:MAC:z", units="m")

        self.declare_partials("data:geometry:vertical_tail:MAC:z", "*", method="exact")
        self.declare_partials(
            "data:geometry:vertical_tail:MAC:length",
            ["data:geometry:vertical_tail:root:chord", "data:geometry:vertical_tail:tip:chord"],
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        chord_sum = root_chord + tip_chord

        mac_vt = (chord_sum**2 - root_chord * tip_chord) / (1.5 * chord_sum)

        z0_vt = b_v * (chord_sum + tip_chord) / (3 * chord_sum)

        outputs["data:geometry:vertical_tail:MAC:length"] = mac_vt
        outputs["data:geometry:vertical_tail:MAC:z"] = z0_vt

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        chord_sum = root_chord + tip_chord

        partials[
            "data:geometry:vertical_tail:MAC:length", "data:geometry:vertical_tail:root:chord"
        ] = (1 - (tip_chord / chord_sum) ** 2) / 1.5
        partials[
            "data:geometry:vertical_tail:MAC:length", "data:geometry:vertical_tail:tip:chord"
        ] = (1 - (root_chord / chord_sum) ** 2) / 1.5

        partials["data:geometry:vertical_tail:MAC:z", "data:geometry:vertical_tail:span"] = (
            chord_sum + tip_chord
        ) / (3 * chord_sum)
        partials["data:geometry:vertical_tail:MAC:z", "data:geometry:vertical_tail:root:chord"] = (
            -b_v * tip_chord / (3 * chord_sum**2)
        )
        partials["data:geometry:vertical_tail:MAC:z", "data:geometry:vertical_tail:tip:chord"] = (
            b_v * root_chord / (3 * chord_sum**2)
        )
