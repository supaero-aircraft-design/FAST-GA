"""
    Computation of the non-equilibrated aircraft polars
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

import numpy as np
from openmdao.core.group import Group
from openmdao.core.explicitcomponent import ExplicitComponent

from fastga.models.aerodynamics.constants import POLAR_POINT_COUNT

import fastoad.api as oad

from ..constants import SUBMODEL_NON_EQUILIBRATED_POLAR

@oad.RegisterSubmodel(
    SUBMODEL_NON_EQUILIBRATED_POLAR, "fastga.submodel.aerodynamics.cl_cd_polar.non_equilibrated.legacy"
)
class ComputeNonEquilibratedPolar(Group):

    def initialize(self):
        self.options.declare("low_speed_aero",  default=False, types=bool)

    def setup(self):

        if self.options["low_speed_aero"]:
            self.add_subsystem(
                "comp_CL", ComputeCL(low_speed_aero=self.options["low_speed_aero"]), promotes=["*"]
            )
            self.add_subsystem(
                "comp_CD", ComputeCD(low_speed_aero=self.options["low_speed_aero"]), promotes=["*"]
            )
        else:
            self.add_subsystem(
                "comp_CL", ComputeCL(low_speed_aero=self.options["low_speed_aero"]), promotes=["*"]
            )
            self.add_subsystem(
                "comp_CD", ComputeCD(low_speed_aero=self.options["low_speed_aero"]), promotes=["*"]
            )


class ComputeCL(ExplicitComponent):

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan, units="rad**-1")
            
            self.add_output("data:aerodynamics:aircraft:low_speed:CL", shape=POLAR_POINT_COUNT)
        else:
            self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
            
            self.add_output("data:aerodynamics:aircraft:cruise:CL", shape=POLAR_POINT_COUNT)

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        if self.options["low_speed_aero"]:
            cl_alpha = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
            cl0 = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
        else:
            cl_alpha = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
            cl0 = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
            
        alpha_array = np.linspace(0, 15, POLAR_POINT_COUNT) * np.pi / 180        
        cl_array = cl0 + alpha_array * cl_alpha

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:aircraft:low_speed:CL"] = cl_array
        else:
            outputs["data:aerodynamics:aircraft:cruise:CL"] = cl_array


class ComputeCD(ExplicitComponent):

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
    
    def setup(self):

        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:wing:low_speed:induced_drag_coefficient", val=np.nan)
            self.add_input("data:aerodynamics:aircraft:low_speed:CD0", val=np.nan)
            self.add_input(
                "data:aerodynamics:aircraft:low_speed:CL", val=np.nan, shape=POLAR_POINT_COUNT
            )

            self.add_output("data:aerodynamics:aircraft:low_speed:CD", shape=POLAR_POINT_COUNT) 
        else:
            self.add_input("data:aerodynamics:aircraft:cruise:CD0", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", val=np.nan)
            self.add_input("data:aerodynamics:aircraft:cruise:CL", val=np.nan, shape=POLAR_POINT_COUNT)

            self.add_output("data:aerodynamics:aircraft:cruise:CD", shape=POLAR_POINT_COUNT)

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        if self.options["low_speed_aero"]:
            coeff_k = inputs["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
            cd0 = inputs["data:aerodynamics:aircraft:low_speed:CD0"]
            cl_array = inputs["data:aerodynamics:aircraft:low_speed:CL"]
        else:
            coeff_k = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
            cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
            cl_array = inputs["data:aerodynamics:aircraft:cruise:CL"]

        cd_array = cd0 + coeff_k * cl_array ** 2

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:aircraft:low_speed:CD"] = cd_array
        else:
            outputs["data:aerodynamics:aircraft:cruise:CD"] = cd_array

