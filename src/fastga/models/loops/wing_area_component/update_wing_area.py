"""
Computation of wing area depending on which criteria between geometric and aerodynamic is the
most constraining.
"""
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

import logging

import numpy as np
import openmdao.api as om


_LOGGER = logging.getLogger(__name__)


class UpdateWingArea(om.ExplicitComponent):
    """
    Computes needed wing area to:
      - have enough lift at required approach speed
      - be able to load enough fuel to achieve the sizing mission.
    """

    def setup(self):

        self.add_input("wing_area:geometric", val=np.nan, units="m**2")
        self.add_input("wing_area:aerodynamic", val=np.nan, units="m**2")

        self.add_output("data:geometry:wing:area", val=10.0, units="m**2")

        self.declare_partials(
            "data:geometry:wing:area",
            [
                "wing_area:geometric",
                "wing_area:aerodynamic",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_area_mission = inputs["wing_area:geometric"]
        wing_area_approach = inputs["wing_area:aerodynamic"]

        _LOGGER.info(
            "Looping on wing area with new value equal to %f",
            max(wing_area_mission, wing_area_approach),
        )

        outputs["data:geometry:wing:area"] = max(wing_area_mission, wing_area_approach)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        wing_area_mission = inputs["wing_area:geometric"]
        wing_area_approach = inputs["wing_area:aerodynamic"]

        if wing_area_mission > wing_area_approach:
            partials["data:geometry:wing:area", "wing_area:geometric"] = 1.0
            partials["data:geometry:wing:area", "wing_area:aerodynamic"] = 0.0
        else:
            partials["data:geometry:wing:area", "wing_area:geometric"] = 0.0
            partials["data:geometry:wing:area", "wing_area:aerodynamic"] = 1.0
