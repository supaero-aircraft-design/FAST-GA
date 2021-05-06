"""
Estimation of tail weight
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

from fastoad.model_base import Atmosphere


class ComputeTailWeight(om.ExplicitComponent):
    """
    Weight estimation for tail weight (only horizontal)

    Based on : Raymer Daniel. Aircraft Design: A Conceptual Approach. AIAA
    Education Series 1996.

    Can also be found in : Gudmundsson, Snorri. General aviation aircraft design: Applied Methods and Procedures.
    Butterworth-Heinemann, 2013. Equation (6-20) and (6-22)
    """

    def setup(self):
        
        self.add_input("data:mission:sizing:cs23:sizing_factor_ultimate", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="kn")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")

        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:horizontal_tail:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)

        self.add_input("data:geometry:has_T_tail", val=np.nan)
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:vertical_tail:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)

        self.add_output("data:weight:airframe:horizontal_tail:mass", units="lb")
        self.add_output("data:weight:airframe:vertical_tail:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor_ultimate"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        v_cruise_ktas = inputs["data:TLAR:v_cruise"]
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]

        area_ht = inputs["data:geometry:horizontal_tail:area"]
        t_c_ht = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        ar_ht = inputs["data:geometry:horizontal_tail:aspect_ratio"]
        taper_ht = inputs["data:geometry:horizontal_tail:taper_ratio"]

        rho_cruise = Atmosphere(cruise_alt).density
        dynamic_pressure = 1./2.*rho_cruise*(v_cruise_ktas*0.5144)**2.*0.0208854
        # In lb/ft2

        a31 = 0.016 * (
                (sizing_factor_ultimate * mtow) ** 0.414 *
                dynamic_pressure ** 0.168 *
                area_ht ** 0.896 *
                (100*t_c_ht / math.cos(sweep_25_ht*math.pi/180.)) ** -0.12 *
                (ar_ht / (math.cos(sweep_25_ht*math.pi/180.))**2.0) ** 0.043 *
                taper_ht ** -0.02
        )
        # Mass formula in lb

        outputs["data:weight:airframe:horizontal_tail:mass"] = a31

        has_t_tail = inputs["data:geometry:has_T_tail"]
        area_vt = inputs["data:geometry:vertical_tail:area"]
        t_c_vt = inputs["data:geometry:vertical_tail:thickness_ratio"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]

        a32 = 0.073 * (1. + 0.2 * has_t_tail) * (
                (sizing_factor_ultimate * mtow) ** 0.376 *
                dynamic_pressure ** 0.122 *
                area_vt ** 0.873 *
                (100*t_c_vt / math.cos(sweep_25_vt*math.pi/180.)) ** -0.49 *
                (ar_vt / (math.cos(sweep_25_vt*math.pi/180.))**2.0) ** 0.357 *
                taper_vt ** 0.039
        )
        # Mass formula in lb

        outputs["data:weight:airframe:vertical_tail:mass"] = a32
