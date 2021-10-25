"""
    Estimation of fuselage wet area
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

import math

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent

from fastga.models.options import FUSELAGE_WET_AREA_OPTION


class ComputeFuselageWetArea(ExplicitComponent):

    """ Fuselage wet area estimation, based on the option it can either be determined based on :
    - a simple geometric description of the fuselage one cone at the front a cylinder in the middle and a cone at the
    back
    - Wells, Douglas P., Bryce L. Horvath, and Linwood A. McCullers. "The Flight Optimization System Weights Estimation
    Method." (2017). Equation 61"""

    def initialize(self):
        self.options.declare(FUSELAGE_WET_AREA_OPTION, types=float, default=0.0)

    def setup(self):

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")

        self.add_output("data:geometry:fuselage:wet_area", units="m**2")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]

        if self.options[FUSELAGE_WET_AREA_OPTION] == 1.0:
            # Using the formula from The Flight Optimization System Weights Estimation Method
            fus_dia = math.sqrt(b_f * h_f)  # equivalent diameter of the fuselage
            wet_area_fus = math.pi * (fus_length / fus_dia - 1.7) * fus_dia ** 2.0
        else:
            # Using the simple geometric description
            fus_dia = math.sqrt(b_f * h_f)  # equivalent diameter of the fuselage
            cyl_length = fus_length - lav - lar
            wet_area_nose = 2.45 * fus_dia * lav
            wet_area_cyl = math.pi * fus_dia * cyl_length
            wet_area_tail = 2.3 * fus_dia * lar
            wet_area_fus = wet_area_nose + wet_area_cyl + wet_area_tail

        outputs["data:geometry:fuselage:wet_area"] = wet_area_fus
