"""FAST - Copyright (c) 2022 ONERA ISAE."""

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

from ..constants import SUBMODEL_DEP_EFFECT

oad.RegisterSubmodel.active_models[
    SUBMODEL_DEP_EFFECT
] = "fastga.submodel.performances.dep_effect.none"


@oad.RegisterSubmodel(SUBMODEL_DEP_EFFECT, "fastga.submodel.performances.dep_effect.none")
class NoDEPEffect(om.ExplicitComponent):
    """
    Using Submodels on not connected inputs will cause the code to crash, this is why we
    created this components that returns the deltas at 0 and have the same inputs as a workaround
    for that problem.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            "flaps_position",
            default="cruise",
            desc="position of the flaps for the computation of the equilibrium",
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("data:geometry:propeller:diameter", val=np.nan, units="m")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input(
            "data:aerodynamics:wing:low_speed:Y_vector",
            val=np.nan,
            shape_by_conn=True,
            units="m",
        )
        self.add_input(
            "data:aerodynamics:wing:low_speed:chord_vector",
            val=np.nan,
            shape_by_conn=True,
            copy_shape="data:aerodynamics:wing:low_speed:Y_vector",
            units="m",
        )
        self.add_input(
            "data:aerodynamics:wing:low_speed:CL_vector",
            val=np.nan,
            shape_by_conn=True,
            copy_shape="data:aerodynamics:wing:low_speed:Y_vector",
        )
        self.add_input("data:geometry:propulsion:engine:y_ratio", shape_by_conn=True, val=np.nan)
        self.add_input(
            "data:geometry:propulsion:nacelle:from_LE",
            shape_by_conn=True,
            copy_shape="data:geometry:propulsion:engine:y_ratio",
            val=np.nan,
            units="m",
        )
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)

        self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:cruise:CM0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CD0", val=np.nan)

        self.add_input("altitude", val=np.full(number_of_points, np.nan), units="m")
        self.add_input("true_airspeed", val=np.full(number_of_points, np.nan), units="m/s")

        self.add_input("alpha", val=np.full(number_of_points, np.nan), units="deg")
        self.add_input("thrust", val=np.full(number_of_points, np.nan), units="N")

        self.add_output("delta_Cl", val=np.full(number_of_points, 0.0))
        self.add_output("delta_Cd", val=np.full(number_of_points, 0.0))
        self.add_output("delta_Cm", val=np.full(number_of_points, 0.0))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["delta_Cl"] = 0.0
        outputs["delta_Cd"] = 0.0
        outputs["delta_Cm"] = 0.0
