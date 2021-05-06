"""
Estimation of wing weight
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
import openmdao.api as om
import math


class ComputeWingWeight(om.ExplicitComponent):
    """
    Wing weight estimation

    Based on : Nicolai, Leland M., and Grant E. Carichner. Fundamentals of aircraft and airship design,
    Volume 1–Aircraft Design. American Institute of Aeronautics and Astronautics, 2010.

    Can also be found in : Gudmundsson, Snorri. General aviation aircraft design: Applied Methods and Procedures.
    Butterworth-Heinemann, 2013. Equation (6-19)
    """

    def setup(self):
        
        self.add_input("data:mission:sizing:cs23:sizing_factor_ultimate", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:TLAR:v_max_sl", val=np.nan, units="kn")
        
        self.add_output("data:weight:airframe:wing:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor_ultimate"]
        wing_area = inputs["data:geometry:wing:area"]
        taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        thickness_ratio = inputs["data:geometry:wing:thickness_ratio"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        aspect_ratio = inputs["data:geometry:wing:aspect_ratio"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        v_max_sl = inputs["data:TLAR:v_max_sl"]

        a1 = 96.948*(
                (mtow*sizing_factor_ultimate/10.0**5.0)**0.65
                * (aspect_ratio/(math.cos(sweep_25)**2.0))**0.57
                * (wing_area/100.0)**0.61 * ((1.0+taper_ratio) / (2.0*thickness_ratio))**0.36
                * (1.+v_max_sl/500.0)**0.5
        )**0.993  # mass formula in lb
            
        outputs["data:weight:airframe:wing:mass"] = a1
