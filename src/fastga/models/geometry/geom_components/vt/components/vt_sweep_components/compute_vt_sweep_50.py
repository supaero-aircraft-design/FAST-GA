"""Estimation of vertical tail sweeps."""
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


# TODO: HT and VT components are similar --> factorize
class ComputeVTSweep50(om.ExplicitComponent):
    """Estimation of vertical tail sweep at l/c=50%"""

    def setup(self):
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:sweep_0", val=np.nan, units="deg")

        self.add_output("data:geometry:vertical_tail:sweep_50", units="rad")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]
        sweep_0 = inputs["data:geometry:vertical_tail:sweep_0"]

        sweep_50 = np.arctan(
            np.tan(sweep_0 * np.pi / 180.0) - 2.0 / ar_vt * ((1.0 - taper_vt) / (1.0 + taper_vt))
        )

        outputs["data:geometry:vertical_tail:sweep_50"] = sweep_50

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]
        sweep_0 = inputs["data:geometry:vertical_tail:sweep_0"]

        tmp = (
            np.tan((np.pi * sweep_0) / 180.0)
            + (2.0 * (taper_vt - 1.0)) / (ar_vt * (taper_vt + 1.0))
        ) ** 2.0 + 1.0

        partials[
            "data:geometry:vertical_tail:sweep_50", "data:geometry:vertical_tail:aspect_ratio"
        ] = -(2.0 * (taper_vt - 1.0)) / (ar_vt**2.0 * (taper_vt + 1.0) * tmp)
        partials[
            "data:geometry:vertical_tail:sweep_50", "data:geometry:vertical_tail:taper_ratio"
        ] = (
            2.0 / (ar_vt * (taper_vt + 1.0))
            - (2.0 * (taper_vt - 1.0)) / (ar_vt * (taper_vt + 1.0) ** 2.0)
        ) / tmp
        partials["data:geometry:vertical_tail:sweep_50", "data:geometry:vertical_tail:sweep_0"] = (
            np.pi * (np.tan((np.pi * sweep_0) / 180.0) ** 2.0 + 1.0)
        ) / (180.0 * tmp)
