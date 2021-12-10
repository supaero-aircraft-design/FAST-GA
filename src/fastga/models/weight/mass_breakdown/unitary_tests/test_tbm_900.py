"""
Test module for mass breakdown functions.
"""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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

import pytest

from ..a_airframe import ComputeFuselageMassAnalytical

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "tbm_900.xml"


def _test_compute_fuselage_mass():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeFuselageMassAnalytical()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageMassAnalytical(), ivc)
    assert problem["data:weight:airframe:fuselage:mass"] == pytest.approx(365.0, abs=1e-2)
