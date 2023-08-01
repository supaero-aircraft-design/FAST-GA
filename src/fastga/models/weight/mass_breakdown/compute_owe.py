"""
Estimation of Operating Weight Empty (OWE).
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

import openmdao.api as om

import numpy as np


class ComputeOWE(om.ExplicitComponent):
    """
    Computes the aircraft operating empty weight.
    """

    def setup(self):

        self.add_input("data:weight:airframe:mass", val=np.nan, units="kg")
        self.add_input("data:weight:propulsion:mass", val=np.nan, units="kg")
        self.add_input("data:weight:systems:mass", val=np.nan, units="kg")
        self.add_input("data:weight:furniture:mass", val=np.nan, units="kg")

        self.add_output("data:weight:aircraft:OWE", units="kg")

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        m_airframe = inputs["data:weight:airframe:mass"]
        m_propulsion = inputs["data:weight:propulsion:mass"]
        m_systems = inputs["data:weight:systems:mass"]
        m_furniture = inputs["data:weight:furniture:mass"]

        outputs["data:weight:aircraft:OWE"] = m_airframe + m_propulsion + m_systems + m_furniture
