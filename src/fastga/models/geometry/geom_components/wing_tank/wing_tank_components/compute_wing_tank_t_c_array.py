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

from .compute_wing_tank_y_array import POINTS_NB_WING


class ComputeWingTankRelativeThicknessArray(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input(
            "data:geometry:propulsion:tank:y_array",
            units="m",
            shape=POINTS_NB_WING,
            val=np.nan,
        )

        self.add_output(
            "data:geometry:propulsion:tank:relative_thickness_array",
            shape=POINTS_NB_WING,
            val=np.full(POINTS_NB_WING, 0.15),
        )

        self.declare_partials(
            of="data:geometry:propulsion:tank:relative_thickness_array",
            wrt=[
                "data:geometry:wing:root:thickness_ratio",
                "data:geometry:wing:tip:thickness_ratio",
                "data:geometry:wing:root:y",
                "data:geometry:wing:tip:y",
            ],
            method="exact",
            rows=np.arange(POINTS_NB_WING),
            cols=np.zeros(POINTS_NB_WING),
        )
        self.declare_partials(
            of="data:geometry:propulsion:tank:relative_thickness_array",
            wrt="data:geometry:propulsion:tank:y_array",
            method="exact",
            rows=np.arange(POINTS_NB_WING),
            cols=np.arange(POINTS_NB_WING),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        root_tc = inputs["data:geometry:wing:root:thickness_ratio"]
        tip_tc = inputs["data:geometry:wing:tip:thickness_ratio"]
        root_y = inputs["data:geometry:wing:root:y"]
        tip_y = inputs["data:geometry:wing:tip:y"]

        y_array = inputs["data:geometry:propulsion:tank:y_array"]

        relative_thickness_array = np.where(
            y_array < root_y,
            np.full_like(y_array, root_tc),
            root_tc + y_array * (tip_tc - root_tc) / (tip_y - root_y),
        )

        outputs["data:geometry:propulsion:tank:relative_thickness_array"] = relative_thickness_array

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        root_tc = inputs["data:geometry:wing:root:thickness_ratio"]
        tip_tc = inputs["data:geometry:wing:tip:thickness_ratio"]
        root_y = inputs["data:geometry:wing:root:y"]
        tip_y = inputs["data:geometry:wing:tip:y"]

        y_array = inputs["data:geometry:propulsion:tank:y_array"]

        partials[
            "data:geometry:propulsion:tank:relative_thickness_array",
            "data:geometry:wing:root:thickness_ratio",
        ] = np.where(
            y_array < root_y, np.full_like(y_array, 1e-6), 1.0 - y_array / (tip_y - root_y)
        )
        partials[
            "data:geometry:propulsion:tank:relative_thickness_array",
            "data:geometry:wing:tip:thickness_ratio",
        ] = np.where(y_array < root_y, np.full_like(y_array, 1e-6), y_array / (tip_y - root_y))

        partials[
            "data:geometry:propulsion:tank:relative_thickness_array", "data:geometry:wing:root:y"
        ] = np.where(
            y_array < root_y,
            np.full_like(y_array, 1e-6),
            y_array * (tip_tc - root_tc) / (tip_y - root_y) ** 2.0,
        )
        partials[
            "data:geometry:propulsion:tank:relative_thickness_array", "data:geometry:wing:tip:y"
        ] = np.where(
            y_array < root_y,
            np.full_like(y_array, 1e-6),
            -y_array * (tip_tc - root_tc) / (tip_y - root_y) ** 2.0,
        )

        partials[
            "data:geometry:propulsion:tank:relative_thickness_array",
            "data:geometry:propulsion:tank:y_array",
        ] = np.where(
            y_array < root_y,
            np.full_like(y_array, 1e-6),
            np.full_like(y_array, (tip_tc - root_tc) / (tip_y - root_y)),
        )
