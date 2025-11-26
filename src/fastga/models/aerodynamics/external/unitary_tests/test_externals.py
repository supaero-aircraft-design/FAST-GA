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
import ast
from ..xfoil.xfoil_polar import XfoilPolar


def test_interpolation_data_type():
    resources_dir = Path(__file__).parent.parent / "xfoil" / "resources"
    csv_files_paths = [str(f) for f in resources_dir.glob("*.csv")]

    for path in csv_files_paths:
        _, data_frame = XfoilPolar._interpolation_for_exist_data(
            result_file=path, mach=0.11, reynolds=6e6
        )

        for label in data_frame.index:
            for index, entry in enumerate(data_frame.loc[label]):
                # Check list entries
                if label in ["alpha", "cl", "cd", "cdp", "cm"]:
                    assert (
                        len(ast.literal_eval(entry)) > 1
                    ), f"Expected list, got {entry} from {path}"

                else:
                    # Check if there is any alphabets
                    assert not any(element.isalpha() for element in entry), (
                        f"Expected numeric, " f"got {entry} from {path}"
                    )
