"""
    Estimation of landing gear center(s) of gravity
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


class ComputeLandingGearCG(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Landing gear center of gravity estimation """

    def setup(self):

        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")

        self.add_output("data:weight:airframe:landing_gear:front:CG:x", units="m")

        self.declare_partials("*", "*", method="fd")


    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        lav = inputs["data:geometry:fuselage:front_length"]
        
        # NLG gravity center
        x_cg_a52 = lav * 0.75

        outputs["data:weight:airframe:landing_gear:front:CG:x"] = x_cg_a52
