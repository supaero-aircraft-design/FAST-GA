"""Estimation of front landing gear center of gravity."""

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

from ..constants import SUBMODEL_FRONT_LANDING_GEAR_CG


@oad.RegisterSubmodel(
    SUBMODEL_FRONT_LANDING_GEAR_CG, "fastga.submodel.weight.cg.airframe.landing_gear.front.legacy"
)
class ComputeFrontLandingGearCG(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Front landing gear center of gravity estimation based on the ratio of weight supported by each
    gear.
    """

    def setup(self):

        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input(
            "settings:weight:airframe:landing_gear:front:front_fuselage_ratio",
            val=0.75,
        )

        self.add_output("data:weight:airframe:landing_gear:front:CG:x", units="m")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        lav = inputs["data:geometry:fuselage:front_length"]
        front_lg_fuselage = inputs[
            "settings:weight:airframe:landing_gear:front:front_fuselage_ratio"
        ]

        x_cg_a52 = lav * front_lg_fuselage

        outputs["data:weight:airframe:landing_gear:front:CG:x"] = x_cg_a52

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        front_lg_fuselage = inputs[
            "settings:weight:airframe:landing_gear:front:front_fuselage_ratio"
        ]
        lav = inputs["data:geometry:fuselage:front_length"]

        partials[
            "data:weight:airframe:landing_gear:front:CG:x", "data:geometry:fuselage:front_length"
        ] = front_lg_fuselage
        partials[
            "data:weight:airframe:landing_gear:front:CG:x",
            "settings:weight:airframe:landing_gear:front:front_fuselage_ratio",
        ] = lav
