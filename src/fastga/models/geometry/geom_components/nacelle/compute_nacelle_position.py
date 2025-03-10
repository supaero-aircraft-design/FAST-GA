"""
Python module for nacelle position calculations, part of the geometry component.
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

from .components import ComputeNacelleXPosition, ComputeNacelleYPosition
from ...constants import SERVICE_NACELLE_POSITION, SUBMODEL_NACELLE_POSITION_LEGACY


# pylint: disable=too-few-public-methods
@oad.RegisterSubmodel(SERVICE_NACELLE_POSITION, SUBMODEL_NACELLE_POSITION_LEGACY)
class ComputeNacellePosition(om.Group):
    """Nacelle and pylon geometry estimation, obtained from :cite:`supaero:2014`."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            "nacelle_x_position",
            ComputeNacelleXPosition(),
            promotes=["*"],
        )

        self.add_subsystem(
            "nacelle_y_position",
            ComputeNacelleYPosition(),
            promotes=["*"],
        )
