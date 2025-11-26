"""Test module for aerodynamics externals."""
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

from pathlib import Path

from fastga.command.api import string_to_array

from ..xfoil.xfoil_polar import XfoilPolar


def test_interpolation_data_type():
    test_file = Path(__file__).parent.parent / "xfoil" / "resources" / "naca23012_20S.csv"

    _, data_frame = XfoilPolar._interpolation_for_exist_data(
        result_file=test_file, mach=0.11, reynolds=6e6
    )

    for label in data_frame.index:
        for index, entry in enumerate(data_frame.loc[label]):
            try:
                array = string_to_array(entry)

            except Exception:
                assert False, "Could not convert data extracted from the csv to an array"

            if label in ["alpha", "cl", "cd", "cdp", "cm"]:
                assert len(array) > 1

            else:
                assert len(array) == 1
