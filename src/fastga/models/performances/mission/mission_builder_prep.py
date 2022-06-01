"""Simple module for prepping the use of the mission builder."""
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

from stdatm import Atmosphere
import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

from fastga.models.performances.mission.takeoff import TakeOffPhase
from fastga.models.weight.cg.cg_variation import InFlightCGVariation


@oad.RegisterOpenMDAOSystem(
    "fastga.performances.mission_builder_prep", domain=ModelDomain.PERFORMANCE
)
class PrepareMissionBuilder(om.Group):
    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self.add_subsystem(
            "takeoff", TakeOffPhase(propulsion_id=self.options["propulsion_id"]), promotes=["*"]
        )
        self.add_subsystem("in_flight_cg_variation", InFlightCGVariation(), promotes=["*"])
        self.add_subsystem("mission_builder_preparation", _PrepareMissionBuilder(), promotes=["*"])


class _PrepareMissionBuilder(om.ExplicitComponent):
    """
    Make some simple computation in order to enable the use of the mission builder in FAST-OAD-GA
    with relevant physic properties
    """

    def setup(self):
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("data:mission:sizing:cs23:min_climb_speed", units="m/s")
        self.add_output("data:mission:sizing:holding:v_holding", units="m/s")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mtow = inputs["data:weight:aircraft:MTOW"]
        wing_area = inputs["data:geometry:wing:area"]
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]

        altitude = 0.0
        g = 9.81

        atm = Atmosphere(altitude=altitude, altitude_in_feet=False)
        rho = atm.density

        v_climb_min = 1.3 * np.sqrt((mtow * g) / (0.5 * rho * wing_area * cl_max_clean))

        outputs["data:mission:sizing:cs23:min_climb_speed"] = v_climb_min

        # Based on the ratio between the speed of best L/D and the speed of best endurance
        # according to Gudmundsson

        outputs["data:mission:sizing:holding:v_holding"] = 0.75 * inputs["data:TLAR:v_cruise"]
