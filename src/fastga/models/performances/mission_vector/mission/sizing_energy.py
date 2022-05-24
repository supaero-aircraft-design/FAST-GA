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


class SizingEnergy(om.ExplicitComponent):
    """Computes the fuel consumed during the whole sizing mission."""

    def setup(self):

        self.add_input("data:mission:sizing:main_route:climb:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:energy", val=np.nan, units="W*h")
        self.add_input("data:mission:sizing:main_route:cruise:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:cruise:energy", val=np.nan, units="W*h")
        self.add_input("data:mission:sizing:main_route:descent:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:descent:energy", val=np.nan, units="W*h")
        self.add_input("data:mission:sizing:main_route:reserve:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:reserve:energy", val=np.nan, units="W*h")
        self.add_input("data:mission:sizing:taxi_out:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_out:energy", val=np.nan, units="W*h")
        self.add_input("data:mission:sizing:takeoff:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:energy", val=np.nan, units="W*h")
        self.add_input("data:mission:sizing:initial_climb:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:energy", val=np.nan, units="W*h")
        self.add_input("data:mission:sizing:taxi_in:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_in:energy", val=np.nan, units="W*h")

        self.add_output("data:mission:sizing:fuel", val=250, units="kg")
        self.add_output("data:mission:sizing:energy", val=200e3, units="W*h")

    def setup_partials(self):

        self.declare_partials(of="data:mission:sizing:fuel", wrt="*:fuel", method="exact")
        self.declare_partials(of="data:mission:sizing:energy", wrt="*:energy", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["data:mission:sizing:fuel"] = (
            inputs["data:mission:sizing:main_route:climb:fuel"]
            + inputs["data:mission:sizing:main_route:cruise:fuel"]
            + inputs["data:mission:sizing:main_route:descent:fuel"]
            + inputs["data:mission:sizing:main_route:reserve:fuel"]
            + inputs["data:mission:sizing:taxi_out:fuel"]
            + inputs["data:mission:sizing:taxi_in:fuel"]
            + inputs["data:mission:sizing:takeoff:fuel"]
            + inputs["data:mission:sizing:initial_climb:fuel"]
        )

        outputs["data:mission:sizing:energy"] = (
            inputs["data:mission:sizing:main_route:climb:energy"]
            + inputs["data:mission:sizing:main_route:cruise:energy"]
            + inputs["data:mission:sizing:main_route:descent:energy"]
            + inputs["data:mission:sizing:main_route:reserve:energy"]
            + inputs["data:mission:sizing:taxi_out:energy"]
            + inputs["data:mission:sizing:taxi_in:energy"]
            + inputs["data:mission:sizing:takeoff:energy"]
            + inputs["data:mission:sizing:initial_climb:energy"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["data:mission:sizing:fuel", "data:mission:sizing:main_route:climb:fuel"] = 1
        partials["data:mission:sizing:fuel", "data:mission:sizing:main_route:cruise:fuel"] = 1
        partials["data:mission:sizing:fuel", "data:mission:sizing:main_route:descent:fuel"] = 1
        partials["data:mission:sizing:fuel", "data:mission:sizing:main_route:reserve:fuel"] = 1
        partials["data:mission:sizing:fuel", "data:mission:sizing:taxi_out:fuel"] = 1
        partials["data:mission:sizing:fuel", "data:mission:sizing:taxi_in:fuel"] = 1
        partials["data:mission:sizing:fuel", "data:mission:sizing:takeoff:fuel"] = 1
        partials["data:mission:sizing:fuel", "data:mission:sizing:initial_climb:fuel"] = 1

        partials["data:mission:sizing:energy", "data:mission:sizing:main_route:climb:energy"] = 1
        partials["data:mission:sizing:energy", "data:mission:sizing:main_route:cruise:energy"] = 1
        partials["data:mission:sizing:energy", "data:mission:sizing:main_route:descent:energy"] = 1
        partials["data:mission:sizing:energy", "data:mission:sizing:main_route:reserve:energy"] = 1
        partials["data:mission:sizing:energy", "data:mission:sizing:taxi_out:energy"] = 1
        partials["data:mission:sizing:energy", "data:mission:sizing:taxi_in:energy"] = 1
        partials["data:mission:sizing:energy", "data:mission:sizing:takeoff:energy"] = 1
        partials["data:mission:sizing:energy", "data:mission:sizing:initial_climb:energy"] = 1
