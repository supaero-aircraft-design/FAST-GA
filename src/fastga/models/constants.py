"""
Module for management of options and factorizing their definition.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2026  ONERA & ISAE-SUPAERO
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

from aenum import IntEnum

CABIN_SIZING_OPTION = "cabin_sizing"
PAYLOAD_FROM_NPAX = "payload_from_npax"


class PropulsionLayout(IntEnum):
    """
    Enumeration of possible propulsion layout.
    """

    UNDER_THE_WING = 1
    IN_THE_REAR = 2
    IN_THE_NOSE = 3


class AircraftCategory(IntEnum):
    """
    Enumeration for possible aircraft category in the CS-23
    """

    AEROBATIC = 1
    UTILITY = 2
    NORMAL = 3
    COMMUTER = 4


class FlapType(IntEnum):
    """
    Enumeration for implemented flap type
    """

    PLAIN_FLAP = 0
    SINGLE_SLOTTED = 1


class WingLayout(IntEnum):
    """
    Enumeration for possible position of the wing
    """

    LOW_WING = 1
    MID_WING = 2
    HIGH_WING = 3


class FuelType(IntEnum):
    """
    Enumeration for fuel type
    """

    AVGAS = 1
    DIESEL = 2
    JET_A1 = 3
