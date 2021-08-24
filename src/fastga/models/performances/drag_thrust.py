"""
    Estimation of fuselage aiframe mass
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
from fastoad.module_management._bundle_loader import BundleLoader
from fastga.command import api as api_cs23
import os.path as pth
import os

from fastga.models.propulsion.fuel_propulsion.base import FuelEngineSet


class ComputeThrustDrag(om.ExplicitComponent):
    """
        Thrust Drag.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", np.nan, units="m")
        self.add_input("data:geometry:wing:root:virtual_chord", np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", np.nan, units="m")
        self.add_input("data:geometry:wing:area", np.nan, units="m**2")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", np.nan, units="m"
        )
        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:cruise:CL0_clean", np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CM0_clean", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:cruise:CL_alpha", np.nan, units="rad**-1")

        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", np.nan, units="rad**-1"
        )
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:weight:aircraft:CG:aft:x", np.nan, units="m")
        self.add_input(
            "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment",
            np.nan,
            units="kg*m",
        )
        self.add_input("data:weight:propulsion:tank:CG:x", np.nan, units="m")
        self.add_input(
            "data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass", np.nan, units="kg"
        )

        self.add_output("data:thrust_drag:speed_array", units="m/s", shape=5)
        self.add_output("data:thrust_drag:thrust_array", units="N", shape=5)
        self.add_output("data:thrust_drag:drag_array", units="N", shape=5)

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mtow = inputs["data:weight:aircraft:mtow"]

        speed_array = []
        thrust_array = []
        drag_array = []

        outputs["data:thrust_drag:speed_array"] = speed_array
        outputs["data:thrust_drag:thrust_array"] = thrust_array
        outputs["data:thrust_drag:drag_array"] = drag_array
