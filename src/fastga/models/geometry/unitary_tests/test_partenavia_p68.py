"""
Test module for geometry functions of the different components.
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

import openmdao.api as om
import pytest

from ..geom_components.wing.components import ComputeWingZ

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "partenavia_p68.xml"


def test_geometry_wing_z():
    """Tests computation of the wing Zs"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingZ()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingZ(), ivc)
    wing_z2 = problem.get_val("data:geometry:wing:root:z", units="m")
    assert wing_z2 == pytest.approx(-0.523, rel=1e-2)
    wing_z4 = problem.get_val("data:geometry:wing:tip:z", units="m")
    assert wing_z4 == pytest.approx(-0.617, rel=1e-2)

    problem.check_partials(compact_print=True)
