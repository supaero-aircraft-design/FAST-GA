"""Computes the mass of the engine support, adapted from Torenbeek by Lucas REMOND."""
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


class ComputeEngineSupport(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:weight:propulsion:engine:mass", val=np.nan, units="kg")
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)

        self.add_output("data:weight:airframe:fuselage:engine_support:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prop_layout = inputs["data:geometry:propulsion:engine:layout"]

        engine_mass = inputs["data:weight:propulsion:engine:mass"]

        if prop_layout == 2 or prop_layout == 3:
            mass_support_engine = 2.5 * engine_mass / 100.0
        else:
            mass_support_engine = 0

        outputs["data:weight:airframe:fuselage:engine_support:mass"] = mass_support_engine
