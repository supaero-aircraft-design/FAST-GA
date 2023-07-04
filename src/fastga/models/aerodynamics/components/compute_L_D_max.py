"""Estimation of the optimal aerodynamics configuration for the aircraft in cruise condition."""
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
import fastoad.api as oad
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.group import Group

from ..constants import SUBMODEL_MAX_L_D


@oad.RegisterSubmodel(SUBMODEL_MAX_L_D, "fastga.submodel.aerodynamics.aircraft.l_d_max.legacy")
class ComputeLDMax(Group):

    def setup(self):

        self.add_subsystem("comp_optimal_CL", ComputeOptimalCL(), promotes=["*"])
        self.add_subsystem("comp_optimal_CD", ComputeOptimalCD(), promotes=["*"])
        self.add_subsystem("comp_L_D_max", ComputeLDMaxValue(), promotes=["*"])
        self.add_subsystem("comp_optimal_alpha", ComputeOptimalAlpha(), promotes=["*"])


class ComputeOptimalCL(ExplicitComponent):

    def setup(self):

        self.add_input("data:aerodynamics:aircraft:cruise:CD0", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient",val=np.nan)
        
        self.add_output("data:aerodynamics:aircraft:cruise:optimal_CL")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        outputs["data:aerodynamics:aircraft:cruise:optimal_CL"] = (
            np.sqrt(inputs["data:aerodynamics:aircraft:cruise:CD0"] / 
                    inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"])
        )


class ComputeOptimalCD(ExplicitComponent):

    def setup(self):

        self.add_input("data:aerodynamics:aircraft:cruise:CD0", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:optimal_CL")

        self.add_output("data:aerodynamics:aircraft:cruise:optimal_CD")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        coeff_k = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        cl_opt = inputs["data:aerodynamics:aircraft:cruise:optimal_CL"]
        
        cd_opt = cd0 + coeff_k * cl_opt ** 2

        outputs["data:aerodynamics:aircraft:cruise:optimal_CD"] = cd_opt


class ComputeLDMaxValue(ExplicitComponent):

    def setup(self):

        self.add_input("data:aerodynamics:aircraft:cruise:optimal_CL")
        self.add_input("data:aerodynamics:aircraft:cruise:optimal_CD")

        self.add_output("data:aerodynamics:aircraft:cruise:L_D_max")

        self.declare_partials("*", "*", method="fd")
    
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["data:aerodynamics:aircraft:cruise:L_D_max"] = (
            inputs["data:aerodynamics:aircraft:cruise:optimal_CL"] / 
            inputs["data:aerodynamics:aircraft:cruise:optimal_CD"]
        )


class ComputeOptimalAlpha(ExplicitComponent):

    def setup(self):

        self.add_input("data:aerodynamics:aircraft:cruise:optimal_CL")
        self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
        
        self.add_output("data:aerodynamics:aircraft:cruise:optimal_alpha", units="deg")
        
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        cl_opt = inputs["data:aerodynamics:aircraft:cruise:optimal_CL"]
        cl0_clean = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
        cl_alpha = inputs["data:aerodynamics:wing:cruise:CL_alpha"]

        alpha = (cl_opt - cl0_clean) / cl_alpha * 180 / np.pi

        outputs["data:aerodynamics:aircraft:cruise:optimal_alpha"] = alpha
