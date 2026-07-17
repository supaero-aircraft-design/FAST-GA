"""
Module for options validating functions.
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


def check_propulsion_id(name: str, value):
    """
    This is a function for validating propulsion_id options. It checks whether the option is set
    to 'None' or is not a string. It is not factorised on purpose so that the error message is
    slightly more explicit. This will also help with debugging. If the option is 'None',
    this will likely be because the option was not passed down correctly by parent group. If it
    is not a string, it will probably be because the user has not set the right type of option.
    """
    if value is None:
        raise ValueError(f"Option '{name}' is expected to be a string, not None")
    if not isinstance(value, str):
        raise ValueError(
            f"Option '{name}' is expected to be a string, not a {type(value).__name__}"
        )
