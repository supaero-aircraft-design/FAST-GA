"""
Python module for wing sweep angle calculations, part of the wing geometry.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
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

from .compute_wing_sweep_0 import ComputeWingSweep0
from .compute_wing_sweep_50 import ComputeWingSweep50
from .compute_wing_sweep_100_inner import ComputeWingSweep100Inner
from .compute_wing_sweep_100_outer import ComputeWingSweep100Outer
from ..constants import SERVICE_WING_SWEEP, SUBMODEL_WING_SWEEP_LEGACY


# pylint: disable=too-few-public-methods
@oad.RegisterSubmodel(SERVICE_WING_SWEEP, SUBMODEL_WING_SWEEP_LEGACY)
class ComputeWingSweep(om.Group):
    """Wing sweeps estimation, obtained from :cite:`supaero:2014`."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
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
