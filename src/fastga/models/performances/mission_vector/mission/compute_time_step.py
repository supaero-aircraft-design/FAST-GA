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

import openmdao.api as om
import numpy as np


class ComputeTimeStep(om.ExplicitComponent):
    """Computes the time step size for the energy consumption later."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("time", val=np.full(n, np.nan), shape=n, units="s")

        self.add_output("time_step", shape=n, units="s")

    def setup_partials(self):

        self.declare_partials(of="time_step", wrt="time", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        time = inputs["time"]

        time_step = time[1:] - time[:-1]
        time_step = np.append(time_step, time_step[-1])

        outputs["time_step"] = time_step

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        n = self.options["number_of_points"]

        middle_diagonal = -np.eye(n)
        upper_diagonal = np.diagflat(np.full(n - 1, 1), 1)
        d_ts_dt = middle_diagonal + upper_diagonal
        d_ts_dt[-1, -1] = 1.0
        d_ts_dt[-1, -2] = -1.0
        partials["time_step", "time"] = d_ts_dt
