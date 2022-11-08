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

from ..constants import SUBMODEL_CL_Q, SUBMODEL_CL_Q_WING, SUBMODEL_CL_Q_HT


@oad.RegisterSubmodel(SUBMODEL_CL_Q, "submodel.aerodynamics.aircraft.cl_pitch_velocity.legacy")
class ComputeCLPitchVelocityAircraft(om.Group):
    """
    Computation of the increase in lift due to a a pitch velocity. Assumes the coefficient at
    aircraft level can be obtained by summing the contribution of the individual components. Not
    destined for the computation of the equilibrium since they are assumed quasi-steady but
    rather for future interface with flight simulator. The convention from
    :cite:`roskampart6:1985` are used, meaning that, for the derivative with respect to a pitch
    rate, this rate is made dimensionless by multiplying it by the MAC and dividing it by 2 times
    the airspeed.

    Based on :cite:`roskampart6:1985` section 10.2.7
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        options = {
            "low_speed_aero": self.options["low_speed_aero"],
        }
        self.add_subsystem(
            "wing_contribution",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CL_Q_WING, options=options),
            promotes=["*"],
        )
        self.add_subsystem(
            "ht_contribution",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CL_Q_HT, options=options),
            promotes=["*"],
        )
        self.add_subsystem(
            "sum",
            _SumCLPitchVelocityContributions(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["*"],
        )


class _SumCLPitchVelocityContributions(om.ExplicitComponent):
    """
    Sums the contribution of the various components to the increase in lift due to a a pitch
    velocity.
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        if self.options["low_speed_aero"]:

            self.add_input("data:aerodynamics:wing:low_speed:CL_q", val=np.nan, units="rad**-1")
            self.add_input(
                "data:aerodynamics:horizontal_tail:low_speed:CL_q", val=np.nan, units="rad**-1"
            )

            self.add_output("data:aerodynamics:aircraft:low_speed:CL_q", units="rad**-1")

        else:

            self.add_input("data:aerodynamics:wing:cruise:CL_q", val=np.nan, units="rad**-1")
            self.add_input(
                "data:aerodynamics:horizontal_tail:cruise:CL_q", val=np.nan, units="rad**-1"
            )

            self.add_output("data:aerodynamics:aircraft:cruise:CL_q", units="rad**-1")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:aircraft:low_speed:CL_q"] = (
                inputs["data:aerodynamics:wing:low_speed:CL_q"]
                + inputs["data:aerodynamics:horizontal_tail:low_speed:CL_q"]
            )
        else:
            outputs["data:aerodynamics:aircraft:cruise:CL_q"] = (
                inputs["data:aerodynamics:wing:cruise:CL_q"]
                + inputs["data:aerodynamics:horizontal_tail:cruise:CL_q"]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        if self.options["low_speed_aero"]:
            partials[
                "data:aerodynamics:aircraft:low_speed:CL_q", "data:aerodynamics:wing:low_speed:CL_q"
            ] = 1.0
            partials[
                "data:aerodynamics:aircraft:low_speed:CL_q",
                "data:aerodynamics:horizontal_tail:low_speed:CL_q",
            ] = 1.0
        else:
            partials[
                "data:aerodynamics:aircraft:cruise:CL_q", "data:aerodynamics:wing:cruise:CL_q"
            ] = 1.0
            partials[
                "data:aerodynamics:aircraft:cruise:CL_q",
                "data:aerodynamics:horizontal_tail:cruise:CL_q",
            ] = 1.0
