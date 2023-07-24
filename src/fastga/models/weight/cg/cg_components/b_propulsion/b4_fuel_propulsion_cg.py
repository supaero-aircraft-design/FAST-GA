"""Estimation of fuel propulsion center of gravity."""
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

from ..constants import SUBMODEL_FUEL_PROPULSION_CG


@oad.RegisterSubmodel(
    SUBMODEL_FUEL_PROPULSION_CG, "fastga.submodel.weight.cg.propulsion.fuel_propulsion.legacy"
)
class ComputeFuelPropulsionCG(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:weight:propulsion:engine:CG:x", units="m", val=np.nan)
        self.add_input("data:weight:propulsion:fuel_lines:CG:x", units="m", val=np.nan)

        self.add_input("data:weight:propulsion:engine:mass", units="kg", val=np.nan)
        self.add_input("data:weight:propulsion:fuel_lines:mass", units="kg", val=np.nan)
        self.add_input("data:weight:propulsion:mass", units="kg", val=np.nan)

        self.add_output("data:weight:propulsion:CG:x", units="m")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        engine_cg = inputs["data:weight:propulsion:engine:CG:x"]
        fuel_lines_cg = inputs["data:weight:propulsion:fuel_lines:CG:x"]

        engine_mass = inputs["data:weight:propulsion:engine:mass"]
        fuel_lines_mass = inputs["data:weight:propulsion:fuel_lines:mass"]

        cg_propulsion = (engine_cg * engine_mass + fuel_lines_cg * fuel_lines_mass) / (
            engine_mass + fuel_lines_mass
        )

        outputs["data:weight:propulsion:CG:x"] = cg_propulsion

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        engine_cg = inputs["data:weight:propulsion:engine:CG:x"]
        fuel_lines_cg = inputs["data:weight:propulsion:fuel_lines:CG:x"]

        engine_mass = inputs["data:weight:propulsion:engine:mass"]
        fuel_lines_mass = inputs["data:weight:propulsion:fuel_lines:mass"]

        partials[
            "data:weight:propulsion:CG:x", "data:weight:propulsion:engine:CG:x"
        ] = engine_mass / (engine_mass + fuel_lines_mass)
        partials[
            "data:weight:propulsion:CG:x", "data:weight:propulsion:fuel_lines:CG:x"
        ] = fuel_lines_mass / (engine_mass + fuel_lines_mass)
        partials["data:weight:propulsion:CG:x", "data:weight:propulsion:engine:mass"] = (
            engine_cg / (engine_mass + fuel_lines_mass)
            - (engine_cg * engine_mass + fuel_lines_cg * fuel_lines_mass)
            / (engine_mass + fuel_lines_mass) ** 2
        )
        partials["data:weight:propulsion:CG:x", "data:weight:propulsion:fuel_lines:mass"] = (
            fuel_lines_cg / (engine_mass + fuel_lines_mass)
            - (engine_cg * engine_mass + fuel_lines_cg * fuel_lines_mass)
            / (engine_mass + fuel_lines_mass) ** 2
        )
