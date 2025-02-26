"""
Estimation of horizontal tail sweeps and aspect ratio.
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

import fastoad.api as oad
import openmdao.api as om

from .ht_sweep_components import ComputeHTSweep0, ComputeHTSweep50, ComputeHTSweep100

from ..constants import SUBMODEL_HT_SWEEP


@oad.RegisterSubmodel(SUBMODEL_HT_SWEEP, "fastga.submodel.geometry.horizontal_tail.sweep.legacy")
class ComputeHTSweep(om.Group):
    # TODO: Document equations. Cite sources
    """Horizontal tail sweeps and aspect ratio estimation"""

    def setup(self):
        self.add_subsystem(
            "HT_sweep_0",
            ComputeHTSweep0(),
            promotes=["*"],
        )

        self.add_subsystem(
            "HT_sweep_50",
            ComputeHTSweep50(),
            promotes=["*"],
        )

        self.add_subsystem(
            "HT_sweep_100",
            ComputeHTSweep100(),
            promotes=["*"],
        )
