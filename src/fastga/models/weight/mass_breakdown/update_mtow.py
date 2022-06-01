"""
Main component for mass breakdown.
"""
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
from openmdao.core.explicitcomponent import ExplicitComponent

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain


@oad.RegisterOpenMDAOSystem("fastga.loop.mtow", domain=ModelDomain.OTHER)
class UpdateMTOW(ExplicitComponent):
    """
    Computes Maximum Take-Off Weight from Maximum Zero Fuel Weight and fuel weight.
    """

    def setup(self):
        self.add_input("data:weight:aircraft:ZFW", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:fuel", val=np.nan, units="kg")

        self.add_output("data:weight:aircraft:MTOW", 1500.0, units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        zfw = inputs["data:weight:aircraft:ZFW"]
        m_fuel = inputs["data:mission:sizing:fuel"]

        mtow = zfw + m_fuel

        outputs["data:weight:aircraft:MTOW"] = mtow
