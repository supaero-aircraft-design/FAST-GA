#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
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

import openmdao.api as om
import numpy as np

POINTS_NB_WING = 50


class ComputeWingTankYArray(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:geometry:propulsion:tank:y_beginning", units="m", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:y_end", units="m", val=np.nan)

        self.add_output(
            "data:geometry:propulsion:tank:y_array",
            units="m",
            shape=POINTS_NB_WING,
            val=np.linspace(1.0, 6.0, POINTS_NB_WING),
        )

        self.declare_partials(
            of="data:geometry:propulsion:tank:y_array",
            wrt=[
                "data:geometry:propulsion:tank:y_beginning",
                "data:geometry:propulsion:tank:y_end",
            ],
            method="exact",
            rows=np.arange(POINTS_NB_WING),
            cols=np.zeros(POINTS_NB_WING),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # For reference, np.linspace(a, b, POINTS_NB) is equal to a + (b - a) *
        # np.arange(POINTS_NB) / (POINTS_NB - 1) which is more easily differentiable
        outputs["data:geometry:propulsion:tank:y_array"] = np.linspace(
            inputs["data:geometry:propulsion:tank:y_beginning"][0],
            inputs["data:geometry:propulsion:tank:y_end"][0],
            POINTS_NB_WING,
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials[
            "data:geometry:propulsion:tank:y_array", "data:geometry:propulsion:tank:y_beginning"
        ] = 1.0 - np.arange(POINTS_NB_WING) / (POINTS_NB_WING - 1)
        partials[
            "data:geometry:propulsion:tank:y_array", "data:geometry:propulsion:tank:y_end"
        ] = np.arange(POINTS_NB_WING) / (POINTS_NB_WING - 1)
