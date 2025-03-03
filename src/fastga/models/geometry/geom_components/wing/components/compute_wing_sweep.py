"""Estimation of wing sweeps."""
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

import openmdao.api as om
import fastoad.api as oad

from .wing_sweep_components import (
    ComputeWingSweep0,
    ComputeWingSweep50,
    ComputeWingSweep100Inner,
    ComputeWingSweep100Outer,
)
from ..constants import SUBMODEL_WING_SWEEP


@oad.RegisterSubmodel(SUBMODEL_WING_SWEEP, "fastga.submodel.geometry.wing.sweep.legacy")
class ComputeWingSweep(om.Group):
    # TODO: Document equations. Cite sources
    """Wing sweeps estimation."""

    def setup(self):
        self.add_subsystem(
            "wing_sweep_0",
            ComputeWingSweep0(),
            promotes=["*"],
        )

        self.add_subsystem(
            "wing_sweep_50",
            ComputeWingSweep50(),
            promotes=["*"],
        )

        self.add_subsystem(
            "wing_sweep_100_inner",
            ComputeWingSweep100Inner(),
            promotes=["*"],
        )

        self.add_subsystem(
            "wing_sweep_100_outer",
            ComputeWingSweep100Outer(),
            promotes=["*"],
        )
