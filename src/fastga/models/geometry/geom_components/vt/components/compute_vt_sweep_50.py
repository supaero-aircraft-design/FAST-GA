"""
Python module for vertical tail sweep angle calculation at 50% of the MAC, part of the vertical tail
sweep angle.
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


class ComputeVTSweep50(om.ExplicitComponent):
    """Estimation of vertical tail sweep at 50% of the MAC."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:sweep_0", val=np.nan, units="rad")

        self.add_output("data:geometry:vertical_tail:sweep_50", units="rad")

        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]
        sweep_0 = inputs["data:geometry:vertical_tail:sweep_0"]

        sweep_50 = np.arctan(np.tan(sweep_0) - 2.0 / ar_vt * (1.0 - taper_vt) / (1.0 + taper_vt))

        outputs["data:geometry:vertical_tail:sweep_50"] = sweep_50

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]
        sweep_0 = inputs["data:geometry:vertical_tail:sweep_0"]

        common_denominator = (
            ar_vt * np.tan(sweep_0) - 2.0 * (1.0 - taper_vt) / (1.0 + taper_vt)
        ) ** 2.0 + ar_vt**2.0

        partials[
            "data:geometry:vertical_tail:sweep_50", "data:geometry:vertical_tail:aspect_ratio"
        ] = 2.0 * (1.0 - taper_vt) / (1.0 + taper_vt) / common_denominator
        partials[
            "data:geometry:vertical_tail:sweep_50", "data:geometry:vertical_tail:taper_ratio"
        ] = 4.0 * ar_vt / common_denominator / (taper_vt + 1.0) ** 2.0
        partials["data:geometry:vertical_tail:sweep_50", "data:geometry:vertical_tail:sweep_0"] = (
            ar_vt**2.0 / np.cos(sweep_0) ** 2.0 / common_denominator
        )
