"""FAST - Copyright (c) 2021 ONERA ISAE."""

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
import openmdao.api as om


class InitializeHorizontalSpeed(om.ExplicitComponent):
    """Computes the fuel consumed at each time step."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("true_airspeed", shape=n, val=np.full(n, np.nan), units="m/s")
        self.add_input("gamma", shape=n, val=np.full(n, 0.0), units="deg")

        self.add_output("horizontal_speed", val=np.full(n, 50.0), units="m/s")

    def setup_partials(self):

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        true_airspeed = inputs["true_airspeed"]
        gamma = inputs["gamma"] * np.pi / 180.0

        outputs["horizontal_speed"] = true_airspeed * np.cos(gamma)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        true_airspeed = inputs["true_airspeed"]
        gamma = inputs["gamma"] * np.pi / 180.0

        partials["horizontal_speed", "gamma"] = (
            -np.diag(true_airspeed * np.sin(gamma)) * np.pi / 180.0
        )
        partials["horizontal_speed", "true_airspeed"] = np.diag(np.cos(gamma))
