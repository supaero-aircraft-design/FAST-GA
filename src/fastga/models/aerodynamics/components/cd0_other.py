"""
    FAST - Copyright (c) 2016 ONERA ISAE
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


class Cd0Other(ExplicitComponent):
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        
        if self.options["low_speed_aero"]:
            self.add_output("data:aerodynamics:other:low_speed:CD0")
        else:
            self.add_output("data:aerodynamics:other:cruise:CD0")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        prop_layout = inputs["data:geometry:propulsion:layout"]
        wing_area = inputs["data:geometry:wing:area"]
        
        # COWLING (only if engine in fuselage): cx_cowl*wing_area assumed typical (Gudmunsson p739)
        if prop_layout == 3.0:
            cd0_cowling = 0.0267 / wing_area
        else:
            cd0_cowling = 0.0
        # Cooling (piston engine only)
        # Gudmunsson p715. Assuming cx_cooling*wing area/MTOW value of the book is typical
        cd0_cooling = 0.0005525  # (7.054E-6 / wing_area * mtow) FIXME: should come from propulsion model...
        # Gudmunnson p739. Sum of other components (not calculated here), cx_other*wing_area assumed typical
        cd0_components = 0.0253 / wing_area
        
        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:other:low_speed:CD0"] = cd0_cowling + cd0_cooling + cd0_components
        else:
            outputs["data:aerodynamics:other:cruise:CD0"] = cd0_cowling + cd0_cooling + cd0_components
