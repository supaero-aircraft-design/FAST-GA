"""Simple module for reserve computation."""
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

import logging
import numpy as np
import openmdao.api as om

import fastoad.api as oad
from ..constants import SUBMODEL_RESERVES

_LOGGER = logging.getLogger(__name__)

POINTS_NB_CLIMB = 100
MAX_CALCULATION_TIME = 15  # time in seconds

oad.RegisterSubmodel.active_models[
    SUBMODEL_RESERVES
] = "fastga.submodel.performances.mission.reserves.legacy"


@oad.RegisterSubmodel(SUBMODEL_RESERVES, "fastga.submodel.performances.mission.reserves.legacy")
class ComputeReserve(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:mission:sizing:main_route:cruise:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:cruise:duration", np.nan, units="s")
        self.add_input("data:mission:sizing:main_route:reserve:duration", np.nan, units="s")

        self.add_input("settings:mission:sizing:main_route:reserve:k_factor", val=1.0)

        self.add_output("data:mission:sizing:main_route:reserve:fuel", units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        m_reserve = (
            inputs["data:mission:sizing:main_route:cruise:fuel"]
            * inputs["data:mission:sizing:main_route:reserve:duration"]
            / max(
                1e-6, inputs["data:mission:sizing:main_route:cruise:duration"]
            )  # avoid 0 division
        ) * inputs["settings:mission:sizing:main_route:reserve:k_factor"]
        outputs["data:mission:sizing:main_route:reserve:fuel"] = m_reserve
