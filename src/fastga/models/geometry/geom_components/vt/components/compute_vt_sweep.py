"""Estimation of vertical tail sweeps."""
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

import fastoad.api as oad
import openmdao.api as om

from .vt_sweep_components import ComputeVTSweep0, ComputeVTSweep50, ComputeVTSweep100
from ..constants import SUBMODEL_VT_SWEEP


# TODO: HT and VT components are similar --> factorize
@oad.RegisterSubmodel(SUBMODEL_VT_SWEEP, "fastga.submodel.geometry.vertical_tail.sweep.legacy")
class ComputeVTSweep(om.Group):
    # TODO: Document equations. Cite sources
    """Vertical tail sweeps estimation."""

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
