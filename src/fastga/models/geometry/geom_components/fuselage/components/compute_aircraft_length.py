"""
Python module for aircraft length calculation, part of the fuselage dimension.
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


class ComputeAircraftLength(om.ExplicitComponent):
    """
    Computes aircraft length.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:aircraft:length", units="m")

        self.declare_partials(of="*", wrt="*", method="exact")
        self.declare_partials("*", "data:geometry:wing:MAC:at25percent:x", val=1.0)

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        ht_lp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        vt_lp = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        ht_length = inputs["data:geometry:horizontal_tail:MAC:length"]
        vt_length = inputs["data:geometry:vertical_tail:MAC:length"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        b_h = inputs["data:geometry:horizontal_tail:span"]

        aircraft_length = fa_length + max(
            ht_lp + 0.75 * ht_length + b_h / 2.0 * np.tan(sweep_25_ht),
            vt_lp + 0.75 * vt_length + b_v * np.tan(sweep_25_vt),
        )

        outputs["data:geometry:aircraft:length"] = aircraft_length

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ht_lp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        vt_lp = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        ht_length = inputs["data:geometry:horizontal_tail:MAC:length"]
        vt_length = inputs["data:geometry:vertical_tail:MAC:length"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        b_h = inputs["data:geometry:horizontal_tail:span"]

        if (ht_lp + 0.75 * ht_length + b_h / 2.0 * np.tan(sweep_25_ht)) > (
            vt_lp + 0.75 * vt_length + b_v * np.tan(sweep_25_vt)
        ):
            partials[
                "data:geometry:aircraft:length",
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
            ] = 1.0
            partials[
                "data:geometry:aircraft:length", "data:geometry:horizontal_tail:MAC:length"
            ] = 0.75
            partials["data:geometry:aircraft:length", "data:geometry:horizontal_tail:sweep_25"] = (
                b_h * (np.tan(sweep_25_ht) ** 2.0 + 1.0) / 2.0
            )
            partials["data:geometry:aircraft:length", "data:geometry:horizontal_tail:span"] = (
                np.tan(sweep_25_ht) / 2.0
            )

            partials[
                "data:geometry:aircraft:length",
                "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            ] = 0.0

            partials["data:geometry:aircraft:length", "data:geometry:vertical_tail:MAC:length"] = (
                0.0
            )
            partials["data:geometry:aircraft:length", "data:geometry:vertical_tail:sweep_25"] = 0.0
            partials["data:geometry:aircraft:length", "data:geometry:vertical_tail:span"] = 0.0
        else:
            partials[
                "data:geometry:aircraft:length",
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
            ] = 0.0
            partials[
                "data:geometry:aircraft:length", "data:geometry:horizontal_tail:MAC:length"
            ] = 0.0
            partials["data:geometry:aircraft:length", "data:geometry:horizontal_tail:sweep_25"] = (
                0.0
            )
            partials["data:geometry:aircraft:length", "data:geometry:horizontal_tail:span"] = 0.0

            partials[
                "data:geometry:aircraft:length",
                "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            ] = 1.0
            partials["data:geometry:aircraft:length", "data:geometry:vertical_tail:MAC:length"] = (
                0.75
            )
            partials["data:geometry:aircraft:length", "data:geometry:vertical_tail:sweep_25"] = (
                b_v * (np.tan(sweep_25_vt) ** 2.0 + 1.0)
            )
            partials["data:geometry:aircraft:length", "data:geometry:vertical_tail:span"] = np.tan(
                sweep_25_vt
            )
