"""
Python package for estimations of each fuselage geometry component.
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

# pylint: disable=unused-import
# flake8: noqa

from .compute_aircraft_length import ComputeAircraftLength
from .compute_fuselage_cabin_length import ComputeFuselageCabinLength
from .compute_fuselage_depth import ComputeFuselageDepth
from .compute_fuselage_dimensions import (
    ComputeFuselageGeometryBasic,
    ComputeFuselageGeometryCabinSizingFD,
    ComputeFuselageGeometryCabinSizingFL,
)
from .compute_fuselage_length_fd import ComputeFuselageLengthFD
from .compute_fuselage_length_fl import ComputeFuselageLengthFL
from .compute_fuselage_luggage_length import ComputeFuselageLuggageLength
from .compute_fuselage_master_cross_section import ComputeFuselageMasterCrossSection
from .compute_fuselage_max_height import ComputeFuselageMaxHeight
from .compute_fuselage_max_width import ComputeFuselageMaxWidth
from .compute_fuselage_nose_length import ComputeFuselageNoseLength
from .compute_fuselage_npax import ComputeFuselageNPAX
from .compute_fuselage_pax_length import ComputeFuselagePAXLength
from .compute_fuselage_rear_length import ComputeFuselageRearLength
from .compute_fuselage_volume import ComputeFuselageVolume
from .compute_fuselage_wet_area import ComputeFuselageWetArea, ComputeFuselageWetAreaFLOPS
