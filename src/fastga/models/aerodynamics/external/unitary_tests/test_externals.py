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

import numpy as np
from pathlib import Path

from fastga.command.api import string_to_array

from ..xfoil.xfoil_polar import XfoilPolar


def test_interpolation_data_type():
    resources_dir = Path(__file__).parent.parent / "xfoil" / "resources"
    csv_files_paths = resources_dir / "naca23012_20S.csv"

    _, data_frame = XfoilPolar._interpolation_for_exist_data(
        result_file=csv_files_paths, mach=0.11, reynolds=6e6
    )

    for label in data_frame.index:
        for index, entry in enumerate(data_frame.loc[label]):
            array = string_to_array(entry)
            # Check list entries
            if label in ["alpha", "cl", "cd", "cdp", "cm"]:
                assert len(array) > 1 and isinstance(array, np.ndarray)

            else:
                assert len(array) == 1 and isinstance(array, np.ndarray)
