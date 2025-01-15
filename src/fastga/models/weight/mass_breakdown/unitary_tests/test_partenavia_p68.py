"""
Test module for mass breakdown functions.
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

import pytest

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from ..a_airframe import (
    ComputeFuselageWeight,
    ComputeFuselageWeightRaymer,
    ComputeFuselageWeightRoskam,
    ComputeFuselageMassAnalytical,
)

XML_FILE = "partenavia_p68.xml"


def test_compute_fuselage_weight():
    """Tests fuselage weight computation from sample XML data."""
    # Research independent input value in .xml file
    inputs_list = [
        "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        "data:weight:aircraft:MTOW",
        "data:weight:airframe:fuselage:k_factor",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:fuselage:maximum_height",
        "data:geometry:fuselage:length",
        "data:TLAR:v_max_sl",
        "data:mission:sizing:main_route:cruise:altitude",
    ]
    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageWeight(), ivc)
    weight_a2 = problem.get_val("data:weight:airframe:fuselage:mass", units="kg")
    assert weight_a2 == pytest.approx(161.65, abs=1e-2)


def test_compute_fuselage_weight_raymer():
    """Tests fuselage weight computation from sample XML data."""
    # Research independent input value in .xml file
    inputs_list = [
        "data:geometry:fuselage:length",
        "data:geometry:fuselage:front_length",
        "data:geometry:fuselage:rear_length",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:fuselage:maximum_height",
        "data:geometry:fuselage:wet_area",
        "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        "data:weight:aircraft:MTOW",
        "data:weight:airframe:fuselage:k_factor",
        "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
        "data:mission:sizing:main_route:cruise:altitude",
        "data:TLAR:v_cruise",
    ]
    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageWeightRaymer(), ivc)
    weight_a2 = problem.get_val("data:weight:airframe:fuselage:mass", units="kg")
    assert weight_a2 == pytest.approx(190.81, abs=1e-2)


def test_compute_fuselage_weight_roskam():
    """Tests fuselage weight computation from sample XML data."""
    # Research independent input value in .xml file
    inputs_list = [
        "data:geometry:fuselage:length",
        "data:geometry:fuselage:front_length",
        "data:weight:aircraft:MTOW",
        "data:geometry:cabin:seats:passenger:NPAX_max",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:fuselage:maximum_height",
        "data:geometry:wing_configuration",
    ]
    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageWeightRoskam(), ivc)
    weight_a2 = problem.get_val("data:weight:airframe:fuselage:mass", units="kg")
    assert weight_a2 == pytest.approx(328.71, abs=1e-2)

    problem.check_partials(compact_print=True)


def test_compute_fuselage_mass_analytical():
    """Tests fuselage weight analytical computation from sample XML data."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeFuselageMassAnalytical()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageMassAnalytical(), ivc)
    assert problem["data:weight:airframe:fuselage:mass"] == pytest.approx(240.12, abs=1e-2)

    problem.check_partials(compact_print=True)
