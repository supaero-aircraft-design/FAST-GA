"""Estimation of the 3D maximum lift coefficients for wing and tail."""
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

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from fastga.models.aerodynamics.constants import SUBMODEL_CL_EXTREME


@oad.RegisterSubmodel(
    SUBMODEL_CL_EXTREME, "fastga.submodel.aerodynamics.aircraft.extreme_lift_coefficient.legacy"
)
class ComputeAircraftMaxCl(om.ExplicitComponent):
    """
    Add high-lift contribution (flaps)
    """

    def setup(self):

        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CL_max", val=np.nan)
        self.add_input("data:aerodynamics:flaps:landing:CL_max", val=np.nan)

        self.add_output("data:aerodynamics:aircraft:takeoff:CL_max")
        self.add_output("data:aerodynamics:aircraft:landing:CL_max")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        cl_max_takeoff = cl_max_clean + inputs["data:aerodynamics:flaps:takeoff:CL_max"]
        cl_max_landing = cl_max_clean + inputs["data:aerodynamics:flaps:landing:CL_max"]

        outputs["data:aerodynamics:aircraft:takeoff:CL_max"] = cl_max_takeoff
        outputs["data:aerodynamics:aircraft:landing:CL_max"] = cl_max_landing
