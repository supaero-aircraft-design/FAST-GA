"""Estimation of landing gears height."""

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


import numpy as np
import openmdao.api as om
import fastoad.api as oad

from ...constants import SUBMODEL_LANDING_GEAR_GEOMETRY


@oad.RegisterSubmodel(
    SUBMODEL_LANDING_GEAR_GEOMETRY, "fastga.submodel.geometry.landing_gear.legacy"
)
class ComputeLGHeight(om.ExplicitComponent):
    """
    Landing gears height position estimation
    """

    def setup(self):

        self.add_input("data:geometry:propeller:diameter", val=np.nan, units="m")

        self.add_output("data:geometry:landing_gear:height", units="m")

        self.declare_partials(
            "data:geometry:landing_gear:height", "data:geometry:propeller:diameter", val=0.41
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prop_dia = inputs["data:geometry:propeller:diameter"]

        lg_height = 0.41 * prop_dia

        outputs["data:geometry:landing_gear:height"] = lg_height
