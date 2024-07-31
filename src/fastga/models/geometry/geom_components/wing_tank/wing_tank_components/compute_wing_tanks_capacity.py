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

from scipy.integrate import trapz

from .compute_wing_tank_y_array import POINTS_NB_WING


class ComputeWingTanksCapacity(om.ExplicitComponent):
    """Compute the capacity of the two wing tanks inside the aircraft wings."""

    def setup(self):

        self.add_input(
            "data:geometry:propulsion:tank:cross_section_array",
            shape=POINTS_NB_WING,
            units="m**2",
            val=np.nan,
        )
        self.add_input(
            "data:geometry:propulsion:tank:y_array",
            shape=POINTS_NB_WING,
            units="m",
            val=np.nan,
        )

        self.add_output(
            "data:geometry:propulsion:tank:capacity",
            units="m**3",
            val=1.0,
            desc="Capacity of both tanks on the aircraft",
        )

        self.declare_partials(
            of="data:geometry:propulsion:tank:capacity",
            wrt="*",
            method="exact",
            rows=np.zeros(POINTS_NB_WING),
            cols=np.arange(POINTS_NB_WING),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        y_array = inputs["data:geometry:propulsion:tank:y_array"]
        cross_section_array = inputs["data:geometry:propulsion:tank:cross_section_array"]

        # trapz should be equivalent to sum(
        #   (
        #       (cross_section_array[:-1] + cross_section_array[1:])
        #       / 2.0
        #       * (y_array[1:] - y_array[:-1])
        #   )
        #
        outputs["data:geometry:propulsion:tank:capacity"] = 2.0 * trapz(
            cross_section_array, y_array
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        y_array = inputs["data:geometry:propulsion:tank:y_array"]
        cross_section_array = inputs["data:geometry:propulsion:tank:cross_section_array"]

        first_order_delta_y = y_array[1:] - y_array[:-1]
        second_order_delta_y = y_array[2:] - y_array[:-2]

        first_order_sigma_cs = cross_section_array[:-1] + cross_section_array[1:]
        second_order_delta_cs = cross_section_array[:-2] - cross_section_array[2:]

        partials_cross_section = np.concatenate(
            (
                np.array([first_order_delta_y[0]]),
                second_order_delta_y,
                np.array([first_order_delta_y[-1]]),
            )
        )
        partials_y = np.concatenate(
            (
                np.array([-first_order_sigma_cs[0]]),
                second_order_delta_cs,
                np.array([first_order_sigma_cs[-1]]),
            )
        )

        partials[
            "data:geometry:propulsion:tank:capacity", "data:geometry:propulsion:tank:y_array"
        ] = partials_y
        partials[
            "data:geometry:propulsion:tank:capacity",
            "data:geometry:propulsion:tank:cross_section_array",
        ] = partials_cross_section
