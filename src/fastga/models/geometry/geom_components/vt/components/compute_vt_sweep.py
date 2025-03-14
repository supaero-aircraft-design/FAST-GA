"""
Python module for vertical tail sweep angle calculations, part of the vertical tail geometry.
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

from .compute_vt_sweep_0 import ComputeVTSweep0
from .compute_vt_sweep_50 import ComputeVTSweep50
from .compute_vt_sweep_100 import ComputeVTSweep100
from ..constants import SERVICE_VT_SWEEP, SUBMODEL_VT_SWEEP_LEGACY


# pylint: disable=too-few-public-methods
@oad.RegisterSubmodel(SERVICE_VT_SWEEP, SUBMODEL_VT_SWEEP_LEGACY)
class ComputeVTSweep(om.Group):
    # TODO: Document equations. Cite sources
    """Vertical tail sweeps estimation."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            "VT_sweep_0",
            ComputeVTSweep0(),
            promotes=["*"],
        )

        self.add_subsystem(
            "VT_sweep_50",
            ComputeVTSweep50(),
            promotes=["*"],
        )

        self.add_subsystem(
            "VT_sweep_100",
            ComputeVTSweep100(),
            promotes=["*"],
        )
