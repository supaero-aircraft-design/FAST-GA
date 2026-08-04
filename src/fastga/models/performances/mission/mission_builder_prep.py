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

import fastoad.api as oad
import numpy as np
import openmdao.api as om
from fastoad.module_management.constants import ModelDomain
from scipy.constants import g
from stdatm import Atmosphere

from fastga.models.performances.mission.takeoff import TakeOffPhase
from fastga.models.weight.cg.cg_variation import InFlightCGVariation

RHO_0 = Atmosphere(altitude=0).density


@oad.RegisterOpenMDAOSystem(
    "fastga.performances.mission_builder_prep", domain=ModelDomain.PERFORMANCE
)
class PrepareMissionBuilder(om.Group):
    def initialize(self):
        self.options.declare("propulsion_id", default=None, allow_none=True)

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
        self.add_input(
            "data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan, units="unitless"
        )
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("data:mission:sizing:cs23:min_climb_speed", units="m/s")
        self.add_output("data:mission:sizing:holding:v_holding", units="m/s")

    def setup_partials(self):
        self.declare_partials(
            of="data:mission:sizing:cs23:min_climb_speed",
            wrt=[
                "data:weight:aircraft:MTOW",
                "data:aerodynamics:wing:low_speed:CL_max_clean",
                "data:geometry:wing:area",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:holding:v_holding",
            wrt="data:TLAR:v_cruise",
            method="exact",
            val=0.75,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mtow = inputs["data:weight:aircraft:MTOW"]
        wing_area = inputs["data:geometry:wing:area"]
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]

        v_climb_min = 1.3 * np.sqrt((mtow * g) / (0.5 * RHO_0 * wing_area * cl_max_clean))

        outputs["data:mission:sizing:cs23:min_climb_speed"] = v_climb_min

        # Based on the ratio between the speed of best L/D and the speed of best endurance
        # according to Gudmundsson

        outputs["data:mission:sizing:holding:v_holding"] = 0.75 * inputs["data:TLAR:v_cruise"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mtow = inputs["data:weight:aircraft:MTOW"]
        wing_area = inputs["data:geometry:wing:area"]
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]

        partials["data:mission:sizing:cs23:min_climb_speed", "data:weight:aircraft:MTOW"] = (
            1.3 / 2.0 * np.sqrt(g / (0.5 * RHO_0 * wing_area * cl_max_clean * mtow))
        )
        partials["data:mission:sizing:cs23:min_climb_speed", "data:geometry:wing:area"] = -(
            1.3 / 2.0 * np.sqrt((mtow * g) / (0.5 * RHO_0 * wing_area**3.0 * cl_max_clean))
        )
        partials[
            "data:mission:sizing:cs23:min_climb_speed",
            "data:aerodynamics:wing:low_speed:CL_max_clean",
        ] = -(1.3 / 2.0 * np.sqrt((mtow * g) / (0.5 * RHO_0 * wing_area * cl_max_clean**3.0)))
