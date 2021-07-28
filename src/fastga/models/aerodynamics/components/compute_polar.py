"""
    Computation of the aircraft polar
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
from ..constants import MACH_NB_PTS

NB_POINTS_ALPHA = 16


class ComputePolar(ExplicitComponent):

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector", shape=MACH_NB_PTS+1, val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:aircraft:mach_interpolation:mach_vector", shape=MACH_NB_PTS+1, val=np.nan)

        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:induced_drag_coefficient", val=np.nan)
            self.add_input("data:aerodynamics:aircraft:low_speed:CD0", val=np.nan)

            self.add_output("data:aerodynamics:aircraft:low_speed:cd_vector", shape=NB_POINTS_ALPHA)
            self.add_output("data:aerodynamics:aircraft:low_speed:cl_vector", shape=NB_POINTS_ALPHA)

        else:
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", val=np.nan)
            self.add_input("data:aerodynamics:aircraft:cruise:CD0", val=np.nan)

            self.add_output("data:aerodynamics:aircraft:cruise:cd_vector", shape=NB_POINTS_ALPHA)
            self.add_output("data:aerodynamics:aircraft:cruise:cl_vector", shape=NB_POINTS_ALPHA)

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cl_alpha_array = inputs["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
        mach_array = inputs["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]

        if self.options["low_speed_aero"]:
            coef_k = inputs["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
            cd0 = inputs["data:aerodynamics:aircraft:low_speed:CD0"]
            mach_number = inputs["data:aerodynamics:low_speed:mach"]
            cl0 = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
        else:
            coef_k = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
            cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
            mach_number = inputs["data:aerodynamics:cruise:mach"]
            cl0 = inputs["data:aerodynamics:wing:cruise:CL0_clean"]

        cl_alpha = np.interp(mach_number, mach_array, cl_alpha_array)

        alpha_array = np.linspace(0, 15, NB_POINTS_ALPHA) * np.pi / 180
        cl_array = cl0 + alpha_array * cl_alpha

        cd_array = cd0 + coef_k * cl_array ** 2

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:aircraft:low_speed:cd_vector"] = cd_array
            outputs["data:aerodynamics:aircraft:low_speed:cl_vector"] = cl_array
        else:
            outputs["data:aerodynamics:aircraft:cruise:cd_vector"] = cd_array
            outputs["data:aerodynamics:aircraft:cruise:cl_vector"] = cl_array
