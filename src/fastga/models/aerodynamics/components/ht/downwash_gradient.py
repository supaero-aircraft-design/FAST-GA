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

from ...constants import SUBMODEL_DOWNWASH


@oad.RegisterSubmodel(
    SUBMODEL_DOWNWASH, "fastga.submodel.aerodynamics.horizontal_tail.downwash.legacy"
)
class DownWashGradientComputation(om.ExplicitComponent):
    def initialize(self):
        """Declaring the low_speed_aero options so we can use low speed and cruise conditions."""
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)

        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan, units="rad**-1")
            self.add_output(
                "data:aerodynamics:horizontal_tail:low_speed:downwash_gradient",
                val=0.35,
            )
            self.declare_partials(
                of="data:aerodynamics:horizontal_tail:low_speed:downwash_gradient",
                wrt=[
                    "data:geometry:wing:aspect_ratio",
                    "data:aerodynamics:wing:low_speed:CL_alpha",
                ],
                method="exact",
            )
        else:
            self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
            self.add_output(
                "data:aerodynamics:horizontal_tail:cruise:downwash_gradient",
                val=0.35,
            )
            self.declare_partials(
                of="data:aerodynamics:horizontal_tail:cruise:downwash_gradient",
                wrt=["data:geometry:wing:aspect_ratio", "data:aerodynamics:wing:cruise:CL_alpha"],
                method="exact",
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_ar = inputs["data:geometry:wing:aspect_ratio"]

        if self.options["low_speed_aero"]:
            cl_alpha = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
        else:
            cl_alpha = inputs["data:aerodynamics:wing:cruise:CL_alpha"]

        downwash_gradient = 2 * cl_alpha / (np.pi * wing_ar)

        if self.options["low_speed_aero"]:
            outputs[
                "data:aerodynamics:horizontal_tail:low_speed:downwash_gradient"
            ] = downwash_gradient
        else:
            outputs[
                "data:aerodynamics:horizontal_tail:cruise:downwash_gradient"
            ] = downwash_gradient

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        wing_ar = inputs["data:geometry:wing:aspect_ratio"]

        if self.options["low_speed_aero"]:
            cl_alpha = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]

            partials[
                "data:aerodynamics:horizontal_tail:low_speed:downwash_gradient",
                "data:geometry:wing:aspect_ratio",
            ] = (
                -2 * cl_alpha / (np.pi * wing_ar ** 2.0)
            )
            partials[
                "data:aerodynamics:horizontal_tail:low_speed:downwash_gradient",
                "data:aerodynamics:wing:low_speed:CL_alpha",
            ] = 2 / (np.pi * wing_ar)
        else:

            cl_alpha = inputs["data:aerodynamics:wing:cruise:CL_alpha"]

            partials[
                "data:aerodynamics:horizontal_tail:cruise:downwash_gradient",
                "data:geometry:wing:aspect_ratio",
            ] = (
                -2 * cl_alpha / (np.pi * wing_ar ** 2.0)
            )
            partials[
                "data:aerodynamics:horizontal_tail:cruise:downwash_gradient",
                "data:aerodynamics:wing:cruise:CL_alpha",
            ] = 2 / (np.pi * wing_ar)
