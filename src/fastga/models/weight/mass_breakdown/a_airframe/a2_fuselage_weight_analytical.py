"""Computes the mass of the fuselage using a method adapted from TASOPT by Lucas REMOND."""
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

from .fuselage_components.compute_shell_mass import ComputeShell
from .fuselage_components.compute_cone_mass import ComputeTailCone
from .fuselage_components.compute_windows_mass import ComputeWindows
from .fuselage_components.compute_insulation_mass import ComputeInsulation
from .fuselage_components.compute_floor_mass import ComputeFloor
from .fuselage_components.compute_nlg_hatch_mass import ComputeNLGHatch
from .fuselage_components.compute_doors_mass import ComputeDoors
from .fuselage_components.compute_wing_fuselage_connection_mass import ComputeWingFuselageConnection
from .fuselage_components.compute_engine_support_mass import ComputeEngineSupport
from .fuselage_components.compute_bulkhead_mass import ComputeBulkhead
from .fuselage_components.compute_additional_bending_material_mass_h import (
    ComputeAddBendingMassHorizontal,
)
from .fuselage_components.compute_additional_bending_material_mass_v import (
    ComputeAddBendingMassVertical,
)
from .fuselage_components.update_fuselage_mass import UpdateFuselageMass

from .constants import SUBMODEL_FUSELAGE_MASS


@oad.RegisterSubmodel(
    SUBMODEL_FUSELAGE_MASS, "fastga.submodel.weight.mass.airframe.fuselage.analytical"
)
class ComputeFuselageMassAnalytical(om.Group):
    """Computes analytically the mass of each fuselage component and add them to get total mass."""

    def setup(self):
        self.add_subsystem("compute_shell", ComputeShell(), promotes=["*"])
        self.add_subsystem("compute_tail_cone", ComputeTailCone(), promotes=["*"])
        self.add_subsystem("compute_windows", ComputeWindows(), promotes=["*"])
        self.add_subsystem("compute_insulation", ComputeInsulation(), promotes=["*"])
        self.add_subsystem("compute_floor", ComputeFloor(), promotes=["*"])
        self.add_subsystem("compute_nlg_hatch", ComputeNLGHatch(), promotes=["*"])
        self.add_subsystem("compute_doors", ComputeDoors(), promotes=["*"])
        self.add_subsystem(
            "compute_wing_fuselage_connection", ComputeWingFuselageConnection(), promotes=["*"]
        )
        self.add_subsystem("compute_engine_support", ComputeEngineSupport(), promotes=["*"])
        self.add_subsystem("compute_bulkhead", ComputeBulkhead(), promotes=["*"])
        self.add_subsystem(
            "compute_add_mass_horizontal", ComputeAddBendingMassHorizontal(), promotes=["*"]
        )
        self.add_subsystem(
            "compute_add_mass_vertical", ComputeAddBendingMassVertical(), promotes=["*"]
        )
        self.add_subsystem("update_fuselage", UpdateFuselageMass(), promotes=["*"])
