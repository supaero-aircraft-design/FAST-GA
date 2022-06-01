"""
    Computation of the aircraft polars
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

from openmdao.api import Group

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

from fastga.models.aerodynamics.components.compute_equilibrated_polar import (
    ComputeEquilibratedPolar,
)
from fastga.models.aerodynamics.components.compute_non_equilibrated_polar import (
    ComputeNonEquilibratedPolar,
)


@oad.RegisterOpenMDAOSystem("fastga.aerodynamics.cl_cd_polar", domain=ModelDomain.AERODYNAMICS)
class ComputePolar(Group):
    def initialize(self):
        self.options.declare("cg_ratio", default=-0.0, types=float)

    def setup(self):
        self.add_subsystem(
            "equilibrated_polar_cruise",
            ComputeEquilibratedPolar(low_speed_aero=False, cg_ratio=self.options["cg_ratio"]),
            promotes=["*"],
        )
        self.add_subsystem(
            "non_equilibrated_polar_cruise",
            ComputeNonEquilibratedPolar(low_speed_aero=False),
            promotes=["*"],
        )
        self.add_subsystem(
            "equilibrated_polar_ls",
            ComputeEquilibratedPolar(low_speed_aero=True, cg_ratio=self.options["cg_ratio"]),
            promotes=["*"],
        )
        self.add_subsystem(
            "non_equilibrated_polar_ls",
            ComputeNonEquilibratedPolar(low_speed_aero=True),
            promotes=["*"],
        )
