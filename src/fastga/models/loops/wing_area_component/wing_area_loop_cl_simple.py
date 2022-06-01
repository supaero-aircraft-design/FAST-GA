"""
Computation of wing area update and constraints based on the lift required in low speed
conditions with simple computation.
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
import openmdao.api as om

from scipy.constants import g

import fastoad.api as oad

from ..constants import SUBMODEL_WING_AREA_AERO_LOOP, SUBMODEL_WING_AREA_AERO_CONS

oad.RegisterSubmodel.active_models[
    SUBMODEL_WING_AREA_AERO_LOOP
] = "fastga.submodel.loop.wing_area.update.aero.simple"
oad.RegisterSubmodel.active_models[
    SUBMODEL_WING_AREA_AERO_CONS
] = "fastga.submodel.loop.wing_area.constraint.aero.simple"


@oad.RegisterSubmodel(
    SUBMODEL_WING_AREA_AERO_LOOP, "fastga.submodel.loop.wing_area.update.aero.simple"
)
class UpdateWingAreaLiftSimple(om.ExplicitComponent):
    """
    Computes needed wing area to have enough lift at required approach speed.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)

    def setup(self):

        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)

        self.add_output("wing_area", val=10.0, units="m**2")

        self.declare_partials(
            "wing_area",
            [
                "data:TLAR:v_approach",
                "data:weight:aircraft:MLW",
                "data:aerodynamics:aircraft:landing:CL_max",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        stall_speed = inputs["data:TLAR:v_approach"] / 1.3
        mlw = inputs["data:weight:aircraft:MLW"]
        max_cl = inputs["data:aerodynamics:aircraft:landing:CL_max"]
        wing_area_approach = 2 * mlw * g / (stall_speed ** 2) / (1.225 * max_cl)

        outputs["wing_area"] = wing_area_approach

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        stall_speed = inputs["data:TLAR:v_approach"] / 1.3
        mlw = inputs["data:weight:aircraft:MLW"]
        max_cl = inputs["data:aerodynamics:aircraft:landing:CL_max"]

        constant = 2 * g / 1.225

        d_wing_area_d_stall_speed = -4.0 * mlw * g / (stall_speed ** 3) / (1.225 * max_cl)

        partials["wing_area", "data:weight:aircraft:MLW"] = constant / (stall_speed ** 2 * max_cl)
        partials["wing_area", "data:TLAR:v_approach"] = d_wing_area_d_stall_speed / 1.3
        partials["wing_area", "data:aerodynamics:aircraft:landing:CL_max"] = (
            -2.0 * mlw * g / (stall_speed ** 2) / (1.225 * max_cl ** 2.0)
        )


@oad.RegisterSubmodel(
    SUBMODEL_WING_AREA_AERO_CONS, "fastga.submodel.loop.wing_area.constraint.aero.simple"
)
class ConstraintWingAreaLiftSimple(om.ExplicitComponent):
    """
    Computes the difference between the lift coefficient required for the low speed conditions
    and the what the wing can provide.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)

    def setup(self):

        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("data:constraints:wing:additional_CL_capacity")

        self.declare_partials(
            "data:constraints:wing:additional_CL_capacity",
            [
                "data:TLAR:v_approach",
                "data:weight:aircraft:MLW",
                "data:aerodynamics:aircraft:landing:CL_max",
                "data:geometry:wing:area",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        v_stall = inputs["data:TLAR:v_approach"] / 1.3
        cl_max = inputs["data:aerodynamics:aircraft:landing:CL_max"]
        mlw = inputs["data:weight:aircraft:MLW"]
        wing_area = inputs["data:geometry:wing:area"]

        outputs["data:constraints:wing:additional_CL_capacity"] = cl_max - mlw * g / (
            0.5 * 1.225 * v_stall ** 2 * wing_area
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        v_stall = inputs["data:TLAR:v_approach"] / 1.3
        mlw = inputs["data:weight:aircraft:MLW"]
        wing_area = inputs["data:geometry:wing:area"]

        partials[
            "data:constraints:wing:additional_CL_capacity",
            "data:aerodynamics:aircraft:landing:CL_max",
        ] = 1.0
        partials[
            "data:constraints:wing:additional_CL_capacity",
            "data:weight:aircraft:MLW",
        ] = -g / (0.5 * 1.225 * v_stall ** 2 * wing_area)
        partials["data:constraints:wing:additional_CL_capacity", "wing_area"] = (
            mlw * g / (0.5 * 1.225 * v_stall ** 2 * wing_area ** 2.0)
        )
        partials["data:constraints:wing:additional_CL_capacity", "data:TLAR:v_approach"] = (
            2.0 * mlw * g / (0.5 * 1.225 * v_stall ** 3.0 * wing_area)
        ) / 1.3
