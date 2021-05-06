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


class Cd0Total(ExplicitComponent):
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:wing:low_speed:CD0", val=np.nan)
            self.add_input("data:aerodynamics:fuselage:low_speed:CD0", val=np.nan)
            self.add_input("data:aerodynamics:horizontal_tail:low_speed:CD0", val=np.nan)
            self.add_input("data:aerodynamics:vertical_tail:low_speed:CD0", val=np.nan)
            self.add_input("data:aerodynamics:nacelles:low_speed:CD0", val=np.nan)
            self.add_input("data:aerodynamics:landing_gear:low_speed:CD0", val=np.nan)
            self.add_input("data:aerodynamics:other:low_speed:CD0", val=np.nan)
            self.add_output("data:aerodynamics:aircraft:low_speed:CD0")
        else:
            self.add_input("data:aerodynamics:wing:cruise:CD0", val=np.nan)
            self.add_input("data:aerodynamics:fuselage:cruise:CD0", val=np.nan)
            self.add_input("data:aerodynamics:horizontal_tail:cruise:CD0", val=np.nan)
            self.add_input("data:aerodynamics:vertical_tail:cruise:CD0", val=np.nan)
            self.add_input("data:aerodynamics:nacelles:cruise:CD0", val=np.nan)
            self.add_input("data:aerodynamics:landing_gear:cruise:CD0", val=np.nan)
            self.add_input("data:aerodynamics:other:cruise:CD0", val=np.nan)
            self.add_output("data:aerodynamics:aircraft:cruise:CD0")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        if self.options["low_speed_aero"]:
            cd0_wing = inputs["data:aerodynamics:wing:low_speed:CD0"]
            cd0_fus = inputs["data:aerodynamics:fuselage:low_speed:CD0"]
            cd0_ht = inputs["data:aerodynamics:horizontal_tail:low_speed:CD0"]
            cd0_vt = inputs["data:aerodynamics:vertical_tail:low_speed:CD0"]
            cd0_nac = inputs["data:aerodynamics:nacelles:low_speed:CD0"]
            cd0_lg = inputs["data:aerodynamics:landing_gear:low_speed:CD0"]
            cd0_other = inputs["data:aerodynamics:other:low_speed:CD0"]
        else:
            cd0_wing = inputs["data:aerodynamics:wing:cruise:CD0"]
            cd0_fus = inputs["data:aerodynamics:fuselage:cruise:CD0"]
            cd0_ht = inputs["data:aerodynamics:horizontal_tail:cruise:CD0"]
            cd0_vt = inputs["data:aerodynamics:vertical_tail:cruise:CD0"]
            cd0_nac = inputs["data:aerodynamics:nacelles:cruise:CD0"]
            cd0_lg = inputs["data:aerodynamics:landing_gear:cruise:CD0"]
            cd0_other = inputs["data:aerodynamics:other:cruise:CD0"]

        # CRUD (other undesirable drag). Factor from Gudmunsson book
        crud_factor = 1.25

        cd0 = crud_factor * (cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_lg + cd0_nac + cd0_other)
        
        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:aircraft:low_speed:CD0"] = cd0
        else:
            outputs["data:aerodynamics:aircraft:cruise:CD0"] = cd0
