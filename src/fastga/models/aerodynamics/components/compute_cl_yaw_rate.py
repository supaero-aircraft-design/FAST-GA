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

from ..constants import (
    SUBMODEL_CL_R_WING,
    SUBMODEL_CL_R_VT,
    SUBMODEL_CL_R,
)


@oad.RegisterSubmodel(SUBMODEL_CL_R, "submodel.aerodynamics.aircraft.roll_moment_yaw_rate.legacy")
class ComputeClYawRateAircraft(om.Group):
    """
    Computation of the increase in roll moment due to a yaw rate. Assumes the coefficient at
    aircraft level can be obtained by summing the contribution of the individual components. Some
    of these computations depend on the aircraft flying conditions, see the warnings in each
    file. The convention from :cite:`roskampart6:1985` are used, meaning that for lateral
    derivative, the reference length is the wing span. Another important point is that,
    for the derivative with respect to yaw and roll, the rotation speed are made dimensionless by
    multiplying them by the wing span and dividing them by 2 times the airspeed.

    Based on :cite:`roskampart6:1985` section 10.2.8
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        options = {
            "low_speed_aero": self.options["low_speed_aero"],
        }
        self.add_subsystem(
            "wing_contribution",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CL_R_WING, options=options),
            promotes=["*"],
        )
        self.add_subsystem(
            "vt_contribution",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CL_R_VT, options=options),
            promotes=["*"],
        )
        self.add_subsystem(
            "sum",
            _SumCLYawRateContributions(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["*"],
        )


class _SumCLYawRateContributions(om.ExplicitComponent):
    """
    Sums the contribution of the various components to the increase in roll moment due to a yaw
    rate.
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        if self.options["low_speed_aero"]:

            self.add_input("data:aerodynamics:wing:low_speed:Cl_r", val=np.nan, units="rad**-1")
            self.add_input(
                "data:aerodynamics:vertical_tail:low_speed:Cl_r", val=np.nan, units="rad**-1"
            )

            self.add_output("data:aerodynamics:aircraft:low_speed:Cl_r", units="rad**-1")

        else:

            self.add_input("data:aerodynamics:wing:cruise:Cl_r", val=np.nan, units="rad**-1")
            self.add_input(
                "data:aerodynamics:vertical_tail:cruise:Cl_r", val=np.nan, units="rad**-1"
            )

            self.add_output("data:aerodynamics:aircraft:cruise:Cl_r", units="rad**-1")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:aircraft:low_speed:Cl_r"] = (
                inputs["data:aerodynamics:wing:low_speed:Cl_r"]
                + inputs["data:aerodynamics:vertical_tail:low_speed:Cl_r"]
            )
        else:
            outputs["data:aerodynamics:aircraft:cruise:Cl_r"] = (
                inputs["data:aerodynamics:wing:cruise:Cl_r"]
                + inputs["data:aerodynamics:vertical_tail:cruise:Cl_r"]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        if self.options["low_speed_aero"]:
            partials[
                "data:aerodynamics:aircraft:low_speed:Cl_r",
                "data:aerodynamics:wing:low_speed:Cl_r",
            ] = 1.0
            partials[
                "data:aerodynamics:aircraft:low_speed:Cl_r",
                "data:aerodynamics:vertical_tail:low_speed:Cl_r",
            ] = 1.0
        else:
            partials[
                "data:aerodynamics:aircraft:cruise:Cl_r", "data:aerodynamics:wing:cruise:Cl_r"
            ] = 1.0
            partials[
                "data:aerodynamics:aircraft:cruise:Cl_r",
                "data:aerodynamics:vertical_tail:cruise:Cl_r",
            ] = 1.0
