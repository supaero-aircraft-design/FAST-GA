"""
Estimation of horizontal tail mean aerodynamic chords.
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


class ComputeHTMacFD(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Horizontal tail mean aerodynamic chord estimation based on (F)ixed tail (D)istance.
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

        mac_ht = (chord_sum**2 - root_chord * tip_chord) / (1.5 * chord_sum)

        x0_ht = (
            (root_chord * 0.5 + b_h * np.tan(sweep_25_ht) - tip_chord * 0.5)
            * (chord_sum + tip_chord)
            / (6 * chord_sum)
        )

        y0_ht = b_h * (chord_sum + tip_chord) / (6 * chord_sum)

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
        ] = (1 - (tip_chord / chord_sum) ** 2) / 1.5
        partials[
            "data:geometry:horizontal_tail:MAC:length", "data:geometry:horizontal_tail:tip:chord"
        ] = (1 - (root_chord / chord_sum) ** 2) / 1.5

        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            "data:geometry:horizontal_tail:root:chord",
        ] = (chord_sum**2 + 2 * tip_chord * (tip_chord - b_h * np.tan(sweep_25_ht))) / (
            12 * chord_sum**2
        )
        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            "data:geometry:horizontal_tail:tip:chord",
        ] = (root_chord * (root_chord + b_h * np.tan(sweep_25_ht)) - chord_sum**2) / (
            6 * chord_sum**2
        )
        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            "data:geometry:horizontal_tail:sweep_25",
        ] = b_h * (chord_sum + tip_chord) / (6 * chord_sum * np.cos(sweep_25_ht) ** 2)
        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            "data:geometry:horizontal_tail:span",
        ] = (chord_sum + tip_chord) * np.tan(sweep_25_ht) / (6 * chord_sum)

        partials["data:geometry:horizontal_tail:MAC:y", "data:geometry:horizontal_tail:span"] = (
            chord_sum + tip_chord
        ) / (6 * chord_sum)
        partials[
            "data:geometry:horizontal_tail:MAC:y", "data:geometry:horizontal_tail:root:chord"
        ] = -b_h * tip_chord / (6 * chord_sum**2)
        partials[
            "data:geometry:horizontal_tail:MAC:y", "data:geometry:horizontal_tail:tip:chord"
        ] = b_h * root_chord / (6 * chord_sum**2)


class ComputeHTMacFL(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Horizontal tail mean aerodynamic chord estimation based on (F)ixed fuselage (L)ength (HTP
    distance computed).
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:has_T_tail", val=np.nan)
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:absolute", val=np.nan, units="m"
        )
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        self.add_output("data:geometry:horizontal_tail:MAC:length", units="m")
        self.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:local", units="m")
        self.add_output("data:geometry:horizontal_tail:MAC:y", units="m")
        self.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", units="m")

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
        self.declare_partials(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
            [
                "data:geometry:horizontal_tail:root:chord",
                "data:geometry:horizontal_tail:tip:chord",
                "data:geometry:horizontal_tail:sweep_25",
                "data:geometry:horizontal_tail:span",
                "data:geometry:horizontal_tail:MAC:at25percent:x:absolute",
                "data:geometry:fuselage:length",
                "data:geometry:wing:MAC:at25percent:x",
            ],
            method="exact",
        )
        self.declare_partials(of="*", wrt="data:geometry:has_T_tail", method="fd")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        b_h = inputs["data:geometry:horizontal_tail:span"]
        fus_length = inputs["data:geometry:fuselage:length"]
        tail_type = inputs["data:geometry:has_T_tail"]
        x_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:absolute"]
        x_wing25 = inputs["data:geometry:wing:MAC:at25percent:x"]

        chord_sum = root_chord + tip_chord

        mac_ht = 2 * (chord_sum**2 - root_chord * tip_chord) / (3 * chord_sum)

        x0_ht = (
            (root_chord + 2 * b_h * np.tan(sweep_25_ht) - tip_chord)
            * (chord_sum + tip_chord)
            / (12 * chord_sum)
        )

        y0_ht = (b_h * (chord_sum + tip_chord)) / (6 * chord_sum)

        if tail_type == 1.0:
            ht_lp = (x_ht + x0_ht) - x_wing25
        else:
            ht_lp = (fus_length - root_chord + x0_ht) - x_wing25

        outputs["data:geometry:horizontal_tail:MAC:length"] = mac_ht
        outputs["data:geometry:horizontal_tail:MAC:at25percent:x:local"] = x0_ht
        outputs["data:geometry:horizontal_tail:MAC:y"] = y0_ht
        outputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"] = ht_lp

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        b_h = inputs["data:geometry:horizontal_tail:span"]
        tail_type = inputs["data:geometry:has_T_tail"]

        chord_sum = root_chord + tip_chord

        partials[
            "data:geometry:horizontal_tail:MAC:length", "data:geometry:horizontal_tail:root:chord"
        ] = (1 - (tip_chord / chord_sum) ** 2) / 1.5
        partials[
            "data:geometry:horizontal_tail:MAC:length", "data:geometry:horizontal_tail:tip:chord"
        ] = (1 - (root_chord / chord_sum) ** 2) / 1.5

        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            "data:geometry:horizontal_tail:root:chord",
        ] = (chord_sum**2 + 2 * tip_chord * (tip_chord - b_h * np.tan(sweep_25_ht))) / (
            12 * chord_sum**2
        )
        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            "data:geometry:horizontal_tail:tip:chord",
        ] = (root_chord * (root_chord + b_h * np.tan(sweep_25_ht)) - chord_sum**2) / (
            6 * chord_sum**2
        )
        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            "data:geometry:horizontal_tail:sweep_25",
        ] = b_h * (chord_sum + tip_chord) / (6 * chord_sum * np.cos(sweep_25_ht) ** 2)
        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            "data:geometry:horizontal_tail:span",
        ] = (chord_sum + tip_chord) * np.tan(sweep_25_ht) / (6 * chord_sum)

        partials["data:geometry:horizontal_tail:MAC:y", "data:geometry:horizontal_tail:span"] = (
            chord_sum + tip_chord
        ) / (6 * chord_sum)
        partials[
            "data:geometry:horizontal_tail:MAC:y", "data:geometry:horizontal_tail:root:chord"
        ] = -b_h * tip_chord / (6 * chord_sum**2)
        partials[
            "data:geometry:horizontal_tail:MAC:y", "data:geometry:horizontal_tail:tip:chord"
        ] = b_h * root_chord / (6 * chord_sum**2)

        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
            "data:geometry:horizontal_tail:tip:chord",
        ] = (root_chord * (root_chord + b_h * np.tan(sweep_25_ht)) - chord_sum**2) / (
            6 * chord_sum**2
        )
        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
            "data:geometry:horizontal_tail:sweep_25",
        ] = b_h * (chord_sum + tip_chord) / (6 * chord_sum * np.cos(sweep_25_ht) ** 2)
        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
            "data:geometry:horizontal_tail:span",
        ] = (chord_sum + tip_chord) * np.tan(sweep_25_ht) / (6 * chord_sum)

        if tail_type == 1.0:
            partials[
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:horizontal_tail:root:chord",
            ] = (chord_sum**2 + 2 * tip_chord * (tip_chord - b_h * np.tan(sweep_25_ht))) / (
                12 * chord_sum**2
            )
            partials[
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:horizontal_tail:MAC:at25percent:x:absolute",
            ] = 1.0
            partials[
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:fuselage:length",
            ] = 0.0
            partials[
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:wing:MAC:at25percent:x",
            ] = -1.0
        else:
            partials[
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:horizontal_tail:root:chord",
            ] = (chord_sum**2 + 2 * tip_chord * (tip_chord - b_h * np.tan(sweep_25_ht))) / (
                12 * chord_sum**2
            ) - 1.0
            partials[
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:horizontal_tail:MAC:at25percent:x:absolute",
            ] = 0.0
            partials[
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:fuselage:length",
            ] = 1.0
            partials[
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:wing:MAC:at25percent:x",
            ] = -1.0
