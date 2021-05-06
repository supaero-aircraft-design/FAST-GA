"""
Estimation of power systems weight
"""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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


class ComputePowerSystemsWeight(ExplicitComponent):
    """
    Weight estimation for power systems (generation and distribution)

    # TODO: Based on :????????????
    """

    def setup(self):
        
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:weight:propulsion:fuel_lines:mass", val=np.nan, units="lb")
        self.add_input("data:weight:systems:navigation:mass", val=np.nan, units="lb")

        self.add_output("data:weight:systems:power:electric_systems:mass", units="lb")
        self.add_output("data:weight:systems:power:hydraulic_systems:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        mtow = inputs["data:weight:aircraft:MTOW"]
        m_fuel_lines = inputs["data:weight:propulsion:fuel_lines:mass"]
        m_iae = inputs["data:weight:systems:navigation:mass"]
        
        c12 = 426. * ((m_fuel_lines+m_iae) / 1000.)**0.51  # mass formula in lb
        c13 = 0.007*mtow  # mass formula in lb
        
        outputs["data:weight:systems:power:electric_systems:mass"] = c12
        outputs["data:weight:systems:power:hydraulic_systems:mass"] = c13
