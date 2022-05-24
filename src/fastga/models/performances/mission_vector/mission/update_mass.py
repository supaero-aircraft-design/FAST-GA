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


class UpdateMass(om.ExplicitComponent):
    """Update mass for next iteration."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input(
            "fuel_consumed_t",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="kg",
        )
        self.add_input("data:mission:sizing:taxi_out:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", val=np.nan, units="kg")

        self.add_output(
            "mass", shape=number_of_points, val=np.full(number_of_points, 1500.0), units="kg"
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mtow = inputs["data:weight:aircraft:MTOW"]
        fuel_taxi_out = inputs["data:mission:sizing:taxi_out:fuel"]
        fuel_takeoff = inputs["data:mission:sizing:takeoff:fuel"]
        fuel_initial_climb = inputs["data:mission:sizing:initial_climb:fuel"]
        number_of_points = self.options["number_of_points"]

        # The + inputs["fuel_consumed_t"][0] is due to the fact that the mass is the one we use
        # to compute the equilibrium so, in a sense it is before the fuel is burned so the first
        # point of the mass vector is gonna be a t the very start of climb which means the first
        # kg of fuel will not have been consumed. Only the fuel for taxi out, takeoff and initial
        # climb is considered
        outputs["mass"] = (
            np.full(number_of_points, mtow)
            - np.cumsum(inputs["fuel_consumed_t"])
            - fuel_taxi_out
            - fuel_takeoff
            - fuel_initial_climb
            + inputs["fuel_consumed_t"][0]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]

        partials["mass", "data:weight:aircraft:MTOW"] = np.full(number_of_points, 1.0)
        partials["mass", "data:mission:sizing:taxi_out:fuel"] = np.full(number_of_points, -1.0)
        partials["mass", "data:mission:sizing:initial_climb:fuel"] = np.full(number_of_points, -1.0)
        partials["mass", "data:mission:sizing:takeoff:fuel"] = np.full(number_of_points, -1.0)
        partials["mass", "fuel_consumed_t"] = -(
            np.tri(number_of_points, number_of_points) - np.eye(number_of_points)
        )
