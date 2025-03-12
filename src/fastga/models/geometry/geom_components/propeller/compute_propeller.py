"""
Python module for the calculation of propeller position and effective advance ratio, part of the
geometry component.
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

from ...constants import SERVICE_PROPELLER_GEOMETRY, SUBMODEL_PROPELLER_GEOMETRY_LEGACY
from .constants import SERVICE_PROPELLER_POSITION, SERVICE_PROPELLER_INSTALLATION


# pylint: disable=too-few-public-methods
@oad.RegisterSubmodel(SERVICE_PROPELLER_GEOMETRY, SUBMODEL_PROPELLER_GEOMETRY_LEGACY)
class ComputePropellerGeometry(om.Group):
    """Propeller position with respect to the leading edge estimation."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            "propeller_position",
            oad.RegisterSubmodel.get_submodel(SERVICE_PROPELLER_POSITION),
            promotes=["*"],
        )
        self.add_subsystem(
            "propeller_effective_advance_ratio",
            oad.RegisterSubmodel.get_submodel(SERVICE_PROPELLER_INSTALLATION),
            promotes=["*"],
        )
