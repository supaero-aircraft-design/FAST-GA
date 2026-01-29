"""
Defines the utility functions for postprocessing.
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

import numpy as np
import openmdao.api as om


def _unit_conversion(variable, new_unit: str):
    """
    Returns the value of the requested variable expressed in the new unit.

    :param variable: instance containing variable information
    :param new_unit: new given units defined by user or functions
    :return: values of the requested variables expressed in the new unit
    """
    original_unit = variable.metadata["units"]

    if len(variable.metadata["val"]) > 1:
        values = variable.metadata["val"]
        return np.array([om.convert_units(value, original_unit, new_unit) for value in values])

    value = variable.metadata["val"][0]
    return om.convert_units(value, original_unit, new_unit)
